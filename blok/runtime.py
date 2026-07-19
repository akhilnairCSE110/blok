from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class BlokRuntimeError(RuntimeError):
    pass


class Planning(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


high = Planning.HIGH


KIMI_K26_TEXT = {
    "model_type": "kimi_k2",
    "num_hidden_layers": 61,
    "hidden_size": 7168,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 384,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
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


@dataclass(frozen=True)
class TextValue:
    value: str

    def asstr(self) -> str:
        return self.value


@dataclass(frozen=True)
class PowerReport:
    watts: float | None

    def low(self) -> bool:
        return self.watts is not None and self.watts <= 250.0


@dataclass(frozen=True)
class PlanReport:
    native: bool
    predicted_tokens_per_second: float | None

    def predicted(self) -> bool:
        return self.predicted_tokens_per_second is not None and self.predicted_tokens_per_second > 0


@dataclass(frozen=True)
class GenerationResponse:
    text: TextValue
    ttft: float
    min_tps: float
    max_tps: float
    power: PowerReport
    plan: PlanReport


@dataclass(frozen=True)
class KimiThread:
    model_dir: Path
    max_tokens: int
    max_time: float
    prompt: str
    planning: Planning

    def run(self) -> GenerationResponse:
        model_dir = resolve_model_dir(self.model_dir)
        check_ready(model_dir)
        manifest = ensure_manifest(model_dir)
        started = time.perf_counter()
        report = run_native_generate(manifest, self.prompt, self.max_tokens, self.max_time)
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        text = normalize_answer(report["text"])
        tokens = float(report["tokens"])
        tps = tokens / elapsed if tokens else 0.0
        return GenerationResponse(
            text=TextValue(text),
            ttft=elapsed,
            min_tps=tps,
            max_tps=tps,
            power=PowerReport(watts=report.get("watts")),
            plan=PlanReport(native=True, predicted_tokens_per_second=report.get("predicted_tps")),
        )


def new_threadi(
    *,
    model_dir: str | os.PathLike[str],
    max_tokens: int,
    max_time: float,
    prompt: str,
    planning: Planning = Planning.HIGH,
) -> KimiThread:
    return new_thread(
        model_dir=model_dir,
        max_tokens=max_tokens,
        max_time=max_time,
        prompt=prompt,
        planning=planning,
    )


def new_thread(
    *,
    model_dir: str | os.PathLike[str],
    max_tokens: int,
    max_time: float,
    prompt: str,
    planning: Planning = Planning.HIGH,
) -> KimiThread:
    if max_tokens <= 0:
        raise BlokRuntimeError("max_tokens must be greater than zero")
    if max_time <= 0:
        raise BlokRuntimeError("max_time must be greater than zero")
    if not prompt:
        raise BlokRuntimeError("prompt must not be empty")
    return KimiThread(Path(model_dir), max_tokens, float(max_time), prompt, planning)


def resolve_model_dir(model_dir: Path) -> Path:
    if str(model_dir) in {"<kimi k2 model directory", "<kimi k2 model directory>"}:
        env = os.getenv("BLOK_KIMI_MODEL_DIR") or os.getenv("BLOK_MODEL")
        if env:
            model_dir = Path(env)
        else:
            home = Path(os.getenv("BLOK_HOME", Path.home() / ".blok"))
            model_dir = (
                home
                / "models"
                / "moonshotai"
                / "Kimi-K2.6"
                / "source"
                / "hf"
                / "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
            )
    model_dir = model_dir.expanduser().resolve()
    if model_dir.is_file():
        model_dir = model_dir.parent
    if not model_dir.is_dir():
        raise BlokRuntimeError(f"model directory does not exist: {model_dir}")
    return model_dir


def validate_kimi_config(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise BlokRuntimeError(f"missing config.json: {config_path}")
    config = json.loads(config_path.read_text())
    text = config.get("text_config", config)
    mismatches = [
        f"{key}={text.get(key)!r}, expected {expected!r}"
        for key, expected in KIMI_K26_TEXT.items()
        if text.get(key) != expected
    ]
    if mismatches:
        raise BlokRuntimeError("unsupported Kimi config: " + "; ".join(mismatches))


def check_ready(model_dir: str | os.PathLike[str]) -> None:
    model_dir = resolve_model_dir(Path(model_dir))
    issues: list[str] = []
    if (model_dir / "config.json").is_file():
        try:
            validate_kimi_config(model_dir)
        except BlokRuntimeError as error:
            issues.append(str(error))
        for name in ("tiktoken.model", "tokenizer_config.json"):
            require_file(model_dir / name, issues)
        try:
            ensure_complete_model(model_dir)
        except BlokRuntimeError as error:
            issues.append(str(error))
    try:
        sidecars = ensure_manifest(model_dir).parent
    except BlokRuntimeError as error:
        issues.append(str(error))
        sidecars = model_dir
    for name in ("manifest.blok", "runtime-index.blok", "tokenizer.blok"):
        require_file(sidecars / name, issues, "; run scripts/model_fetch.py kimi-k2.6 materialize")
    if not _blok_bin().is_file():
        issues.append("native blok binary is missing; run cargo build --release")
    if not Path(os.getenv("BLOK_KIMI_EXEC_BIN", "build/blok-kimi-exec")).is_file():
        issues.append("CUDA executor is missing; run cmake -S . -B build && cmake --build build")
    for key in ("BLOK_UGDS_DEVICE", "BLOK_UGDS_MAP"):
        value = os.getenv(key)
        if not value:
            issues.append(f"missing {key}")
        elif not Path(value).exists():
            issues.append(f"{key} does not exist: {value}")
    for key in ("BLOK_KV_UGDS_BASE", "BLOK_KV_UGDS_BYTES"):
        value = os.getenv(key)
        if not value:
            issues.append(f"missing {key}")
        elif not value.isdigit() or int(value) <= 0:
            issues.append(f"{key} must be a positive byte count/offset")
    if issues:
        raise BlokRuntimeError("model is not ready:\n- " + "\n- ".join(issues))


def require_file(path: Path, issues: list[str], suffix: str = "") -> None:
    if not path.is_file():
        issues.append(f"missing {path.name}: {path}{suffix}")


def ensure_complete_model(model_dir: Path) -> None:
    shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if not shards:
        raise BlokRuntimeError(f"no safetensor shards found in {model_dir}")
    expected = int(shards[0].name.split("-of-", 1)[1].split(".", 1)[0])
    if len(shards) != expected:
        raise BlokRuntimeError(f"incomplete model: found {len(shards)}/{expected} shards")


def ensure_manifest(model_dir: Path) -> Path:
    env = os.getenv("BLOK_MODEL")
    candidates = [
        Path(env).expanduser().resolve() if env else model_dir / "manifest.blok",
        model_dir / "blok" / "manifest.blok",
        Path(os.getenv("BLOK_META_ROOT", Path.home() / ".blok" / "metadata"))
        / "moonshotai"
        / "Kimi-K2.6"
        / "manifest.blok",
        model_dir / "manifest.blok",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise BlokRuntimeError("missing manifest.blok; run scripts/model_fetch.py kimi-k2.6 materialize")


def run_native_generate(manifest: Path, prompt: str, max_tokens: int, max_time: float | None = None) -> dict:
    exe = _blok_bin()
    if not exe.is_file():
        raise BlokRuntimeError("native blok binary is missing; run cargo build --release")
    try:
        proc = subprocess.run(
            [
                str(exe),
                "generate",
                "--model",
                str(manifest),
                "--prompt",
                prompt,
                "--prompt-tokens",
                ",".join(map(str, encode_kimi_prompt(manifest.parent / "tokenizer.blok", prompt))),
                "--tokens",
                str(max_tokens),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max_time,
        )
    except subprocess.TimeoutExpired as error:
        raise BlokRuntimeError(f"generation exceeded {max_time:g} seconds") from error
    if proc.returncode != 0:
        raise BlokRuntimeError(proc.stderr.strip() or proc.stdout.strip())
    report = json.loads(proc.stdout)
    if report.get("status") != "ok" or "text" not in report:
        raise BlokRuntimeError(f"native generation did not emit text: {proc.stdout.strip()}")
    return report


def encode_kimi_prompt(tokenizer_path: Path, prompt: str) -> list[int]:
    try:
        import tiktoken
    except ImportError as error:
        raise BlokRuntimeError("tiktoken is required; run python3 -m pip install -r requirements.txt") from error
    ranks, special = {}, {}
    for line in tokenizer_path.read_text().splitlines()[1:]:
        kind, token_id, encoded = line.split()
        value = bytes.fromhex(encoded)
        if kind == "tok":
            ranks[value] = int(token_id)
        elif kind == "special":
            special[value.decode()] = int(token_id)
    encoding = tiktoken.Encoding("kimi-k2.6", pat_str=KIMI_PATTERN, mergeable_ranks=ranks, special_tokens=special)
    formatted = (
        "<|im_user|>user<|im_middle|>"
        + prompt
        + "<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>"
    )
    return encoding.encode(formatted, allowed_special="all")


def _blok_bin() -> Path:
    exe = Path(os.getenv("BLOK_BIN", Path(__file__).resolve().parents[1] / "target" / "release" / "blok"))
    return exe if exe.is_file() else Path(__file__).resolve().parents[1] / "target" / "debug" / "blok"


def normalize_answer(text: str) -> str:
    text = text.strip()
    if text.startswith("<") and ">" in text:
        text = text[1 : text.index(">")]
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).strip()
