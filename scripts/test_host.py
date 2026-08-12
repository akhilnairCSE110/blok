#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import blok.runtime as runtime
import scripts.check_kimi_contract as contract
import scripts.model_fetch as fetch
import scripts.plan_ugds_layout as layout


class Encoding:
    def __init__(self): self.encoded = ""
    def encode(self, text, **_): self.encoded = text; return [11, 12]
    def decode(self, ids): assert ids == [13]; return "<Paris>"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root, meta = Path(tmp) / "model", Path(tmp) / "meta"
        root.mkdir()
        shard = root / "model-00001-of-00064.safetensors"
        shard.write_bytes(b"x" * 4096)
        context = {"model": "kimi-k2.6", "local_dir": root, "meta_dir": meta}
        row = ("language_model.model.norm.weight", "resident", "bf16", "1", str(shard), 8, 2)
        with mock.patch.object(fetch, "status", return_value={"complete": True, "safetensors": 64}), \
             mock.patch.object(fetch, "tokenizer", return_value="blok-tokenizer-v2\n"), \
             mock.patch.object(fetch, "tensors", return_value=[row]):
            report = fetch.materialize(context)
        index = Path(report["index"])
        tensor_line = index.read_text().splitlines()[1]
        assert len(tensor_line.split()) == 14
        expected = {
            "hello": [163587, 2482, 163601, 22931, 163586, 163588, 69702, 163601, 163606, 163607],
            "1234567890": [163587, 2482, 163601, 6694, 12972, 16242, 15, 163586, 163588, 69702, 163601, 163606, 163607],
        }
        with mock.patch.object(contract, "encode_prompt", side_effect=lambda _, prompt: expected[prompt]):
            assert contract.load(index)[(-1, "resident", "norm", -1)][2:] == ((1,), 2)
        assert layout.read_required_ranges(index, 4096) == {shard.resolve(): [(0, 4096)]}

        executor, device, map_file = (Path(tmp) / name for name in ("exec", "device", "map"))
        for path in (executor, device, map_file): path.touch()
        encoding = Encoding()
        env = {"BLOK_KIMI_EXEC_BIN": str(executor), "BLOK_UGDS_DEVICE": str(device),
               "BLOK_UGDS_MAP": str(map_file), "BLOK_KV_UGDS_BASE": "4096", "BLOK_KV_UGDS_BYTES": "100000000"}
        result = subprocess.CompletedProcess([], 0, '{"status":"ok","token_ids":[13],"finish_reason":"length"}', "")
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(runtime, "tokenizer_encoding", return_value=encoding), \
             mock.patch.object(runtime.subprocess, "run", return_value=result) as run:
            assert runtime.answer(runtime.generate(model_dir=meta, prompt="Paris?", max_tokens=1, max_time=2)) == "paris"
        assert encoding.encoded == "<|im_user|>user<|im_middle|>Paris?<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>"
        assert run.call_args.args[0][-4:] == ["--prompt-tokens", "11,12", "--tokens", "1"]
        assert runtime.required_kv_bytes(10_000, 10_000) == 99_942_400_000
    print("ok: host index, contract, layout, tokenizer, executor, and public API")


if __name__ == "__main__": main()
