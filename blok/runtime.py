import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


class BlokRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


KIMI_TEXT = {
    "model_type": "kimi_k2",
    "num_hidden_layers": 61,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "first_k_dense_replace": 1,
    "num_attention_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 384,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "moe_intermediate_size": 2048,
    "routed_scaling_factor": 2.827,
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "rms_norm_eps": 1e-5,
    "rope_theta": 50000.0,
    "max_position_embeddings": 262144,
    "vocab_size": 163840,
}
KIMI_PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)
MAX_CONTEXT = 262144
KV_BYTES_PER_TOKEN = 61 * (64 * 192 + 64 * 128) * 4


def generate(*, model_dir: str | os.PathLike[str], prompt: str, max_tokens: int, max_time: float) -> str:
    return generate_report(model_dir=model_dir, prompt=prompt, max_tokens=max_tokens, max_time=max_time).text


def generate_report(*, model_dir: str | os.PathLike[str], prompt: str, max_tokens: int, max_time: float) -> GenerationResult:
    if not prompt or max_tokens <= 0 or max_time <= 0:
        raise BlokRuntimeError("prompt must be non-empty and limits must be positive")
    index = runtime_index(model_dir)
    tokenizer = index.parent / "tokenizer.blok"
    executor = Path(os.getenv("BLOK_KIMI_EXEC_BIN", Path(__file__).parents[1] / "build/blok-kimi-exec"))
    required = [index, tokenizer, executor]
    missing = [str(path) for path in required if not path.is_file()]
    for key in ("BLOK_UGDS_DEVICE", "BLOK_UGDS_MAP"):
        value = os.getenv(key, "")
        valid = value and (Path(value).exists() if key.endswith("DEVICE") else Path(value).is_file())
        if not valid: missing.append(f"{key}={value or '<unset>'}")
    if missing:
        raise BlokRuntimeError("missing required file(s): " + ", ".join(missing))
    for key in ("BLOK_KV_UGDS_BASE", "BLOK_KV_UGDS_BYTES"):
        if not os.getenv(key, "").isdigit() or int(os.getenv(key, "0")) <= 0:
            raise BlokRuntimeError(f"{key} must be a positive integer")
    try:
        encoding = tokenizer_encoding(tokenizer)
        prompt_ids = encode_prompt(tokenizer, prompt, encoding)
        if len(prompt_ids) + max_tokens > MAX_CONTEXT:
            raise BlokRuntimeError(f"requested {len(prompt_ids) + max_tokens} tokens exceeds Kimi's {MAX_CONTEXT}-token context")
        required = required_kv_bytes(len(prompt_ids), max_tokens)
        capacity = int(os.environ["BLOK_KV_UGDS_BYTES"])
        if capacity < required:
            raise BlokRuntimeError(f"BLOK_KV_UGDS_BYTES={capacity} is smaller than the required {required} bytes")
        result = subprocess.run(
            [
                str(executor),
                "--index",
                str(index),
                "--prompt-tokens",
                ",".join(map(str, prompt_ids)),
                "--tokens",
                str(max_tokens),
            ],
            text=True,
            capture_output=True,
            timeout=max_time,
        )
    except subprocess.TimeoutExpired as error:
        raise BlokRuntimeError(f"generation exceeded {max_time:g} seconds") from error
    if result.returncode:
        raise BlokRuntimeError(result.stderr.strip() or result.stdout.strip())
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise BlokRuntimeError(f"executor returned invalid JSON: {result.stdout!r}") from error
    ids = report.get("token_ids")
    finish_reason = report.get("finish_reason")
    if (report.get("status") != "ok" or finish_reason not in {"eos", "length"} or not isinstance(ids, list) or
            not all(type(i) is int and 0 <= i < KIMI_TEXT["vocab_size"] for i in ids)):
        raise BlokRuntimeError(f"executor failed: {report}")
    return GenerationResult(encoding.decode(ids), len(prompt_ids), len(ids), finish_reason)


def required_kv_bytes(input_tokens: int, output_tokens: int) -> int:
    if input_tokens <= 0 or output_tokens <= 0 or input_tokens + output_tokens > MAX_CONTEXT:
        raise BlokRuntimeError("token counts must be positive and fit the Kimi context")
    return (input_tokens + output_tokens) * KV_BYTES_PER_TOKEN


def runtime_index(model: str | os.PathLike[str]) -> Path:
    value = str(model)
    if value in {"<kimi k2 model directory", "<kimi k2 model directory>"}:
        value = os.getenv("BLOK_MODEL", "")
    candidates = []
    if value:
        path = Path(value).expanduser().resolve()
        candidates += [path, path / "runtime-index.blok"]
        config = path / "config.json"
        if config.is_file():
            raw = json.loads(config.read_text())
            text = raw.get("text_config", raw)
            wrong = [key for key, expected in KIMI_TEXT.items() if text.get(key) != expected]
            if wrong:
                raise BlokRuntimeError("unsupported Kimi config fields: " + ", ".join(wrong))
    home = Path(os.getenv("BLOK_HOME", Path.home() / ".blok"))
    candidates.append(Path(os.getenv("BLOK_META_ROOT", home / "metadata")) / "moonshotai/Kimi-K2.6/runtime-index.blok")
    for path in candidates:
        if path.is_file() and path.name == "runtime-index.blok":
            return path
    raise BlokRuntimeError("runtime-index.blok not found; run scripts/model_fetch.py kimi-k2.6 materialize")


def tokenizer_encoding(tokenizer: Path):
    try:
        import tiktoken
    except ImportError as error:
        raise BlokRuntimeError("install requirements.txt") from error
    ranks, special = {}, {}
    lines = tokenizer.read_text().splitlines()
    if not lines or lines[0] != "blok-tokenizer-v2":
        raise BlokRuntimeError(f"bad tokenizer: {tokenizer}")
    for line in lines[1:]:
        kind, token_id, encoded = line.split()
        value = bytes.fromhex(encoded)
        (ranks if kind == "tok" else special)[value if kind == "tok" else value.decode()] = int(token_id)
    return tiktoken.Encoding("kimi-k2.6", pat_str=KIMI_PATTERN, mergeable_ranks=ranks, special_tokens=special)


def encode_prompt(tokenizer: Path, prompt: str, encoding=None) -> list[int]:
    encoding = encoding or tokenizer_encoding(tokenizer)
    text = f"<|im_user|>user<|im_middle|>{prompt}<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>"
    return encoding.encode(text, allowed_special="all")


def answer(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    if text.startswith("<") and ">" in text:
        text = text[1 : text.index(">")]
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).strip()
