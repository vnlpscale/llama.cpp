#!/usr/bin/env python3
"""Create synthetic GDN64 fixtures and test loading, graph reuse and state reset. No model weights are downloaded."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf  # noqa: E402
import numpy as np  # noqa: E402


def make_model(path: Path, vocab: Path, stale_mtp: bool) -> None:
    writer = gguf.GGUFWriter(path, "qwen35")
    reader = gguf.GGUFReader(vocab)
    n_vocab = len(reader.fields["tokenizer.ggml.tokens"].contents())
    for name, field in reader.fields.items():
        if name.startswith("tokenizer."):
            writer.add_key_value(name, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
    writer.add_name("SYNTHETIC GDN64 regression fixture - not a language model")
    writer.add_block_count(65 if stale_mtp else 64)
    if stale_mtp:
        writer.add_nextn_predict_layers(1)
    writer.add_context_length(128)
    writer.add_embedding_length(32)
    writer.add_feed_forward_length(64)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(8)
    writer.add_rope_dimension_sections([1, 1, 2, 0])
    writer.add_full_attention_interval(4)  # Deliberately stale: tensor evidence must win.
    writer.add_ssm_conv_kernel(4)
    writer.add_ssm_state_size(128)
    writer.add_ssm_group_count(1)
    writer.add_ssm_time_step_rank(2)
    writer.add_ssm_inner_size(256)
    rng = np.random.default_rng(3764)

    def weight(name: str, shape: tuple[int, ...], constant: float | None = None) -> None:
        data = (rng.standard_normal(shape) * 0.02).astype(np.float32) if constant is None else np.full(shape, constant, np.float32)
        writer.add_tensor(name, data)

    weight("token_embd.weight", (n_vocab, 32))
    weight("output_norm.weight", (32,), 1.0)
    for i in range(64):
        prefix = f"blk.{i}."
        for name in ("attn_norm.weight", "post_attention_norm.weight"):
            weight(prefix + name, (32,), 1.0)
        weight(prefix + "attn_qkv.weight", (512, 32))
        weight(prefix + "attn_gate.weight", (256, 32))
        weight(prefix + "ssm_conv1d.weight", (512, 4))
        weight(prefix + "ssm_dt.bias", (2,), -2.0)
        weight(prefix + "ssm_a", (2,), -0.5)
        weight(prefix + "ssm_beta.weight", (2, 32))
        weight(prefix + "ssm_alpha.weight", (2, 32))
        weight(prefix + "ssm_norm.weight", (128,), 1.0)
        weight(prefix + "ssm_out.weight", (32, 256))
        weight(prefix + "ffn_gate.weight", (64, 32))
        weight(prefix + "ffn_up.weight", (64, 32))
        weight(prefix + "ffn_down.weight", (32, 64))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--gpu-layers", type=int, default=0)
    args = parser.parse_args()
    suffix = ".exe" if sys.platform == "win32" else ""
    candidates = [args.build_dir / "bin" / f"test-gdn64-model{suffix}", args.build_dir / "bin" / "Release" / f"test-gdn64-model{suffix}"]
    binary = next((p.resolve() for p in candidates if p.is_file()), None)
    if binary is None:
        parser.error("build the test-gdn64-model target first")
    with tempfile.TemporaryDirectory(prefix="gdn64-smoke-") as tmp:
        for stale_mtp in (False, True):
            path = Path(tmp) / f"synthetic-gdn64-mtp-{int(stale_mtp)}.gguf"
            make_model(path, ROOT / "models" / "ggml-vocab-gpt-2.gguf", stale_mtp)
            subprocess.run([str(binary), str(path), str(args.gpu_layers)], check=True)


if __name__ == "__main__":
    main()
