#!/usr/bin/env python3
"""Run correctness-gated GDN CUDA A/B benchmarks without silently testing a missing backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def binary(build: Path, name: str) -> Path:
    name += ".exe" if sys.platform == "win32" else ""
    for path in (build / "bin" / name, build / "bin" / "Release" / name):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Build target {name} first")


def run(args: list[str], env: dict[str, str], path: Path) -> str:
    result = subprocess.run(args, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    path.write_text(result.stdout, encoding="utf-8")
    path.with_suffix(path.suffix + ".stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {path} and its .stderr file")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("gdn64-bench-results"))
    parser.add_argument("--backend", default="CUDA0")
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if not args.model.is_file() or args.repetitions < 2 or not args.backend.startswith("CUDA"):
        parser.error("provide an existing GGUF, at least two repetitions, and a CUDA backend name")
    args.out.mkdir(parents=True, exist_ok=False)
    ops = str(binary(args.build_dir, "test-backend-ops"))
    bench = str(binary(args.build_dir, "llama-bench"))
    digest = hashlib.sha256()
    with args.model.open("rb") as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            digest.update(block)
    metadata = {"model": str(args.model.resolve()), "sha256": digest.hexdigest(),
                "backend": args.backend, "repetitions": args.repetitions,
                "warning": "These results do not establish language-model quality or bitwise equivalence."}
    try:
        metadata["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True)
    except (OSError, subprocess.CalledProcessError):
        metadata["gpu"] = "nvidia-smi unavailable"
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for mode in ("0", "1"):
        env = dict(os.environ, GGML_CUDA_GDN_TILED=mode)
        log = run([ops, "test", "-b", args.backend, "-o", "GATED_DELTA_NET.*"], env,
                  args.out / f"correctness-{mode}.log")
        plain = re.sub(r"\x1b\[[0-9;]*m", "", log)
        counts = re.search(r"(\d+)/(\d+) tests passed", plain)
        # Upstream's backend filter can skip every backend and still exit zero.
        if (f"Backend {args.backend}: OK" not in plain or counts is None or
                int(counts[1]) < 57 or counts[1] != counts[2] or
                "GATED_DELTA_NET_REFERENCE" not in plain):
            raise RuntimeError("CUDA correctness cases did not actually run and pass; refusing to benchmark")
        run([ops, "perf", "-b", args.backend, "-o", "GATED_DELTA_NET"], env,
            args.out / f"kernel-{mode}.log")
        output = run([bench, "-m", str(args.model.resolve()), "-ngl", "99", "-p", "512", "-n", "128",
                      "-r", str(args.repetitions), "-o", "json"], env, args.out / f"model-{mode}.json")
        rows = json.loads(output)
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("No model benchmark results")
    print(f"A/B logs and benchmark JSON: {args.out.resolve()}")
    print("Compare prompt and generation results separately; repeat with reversed order to check thermal bias.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(str(exc))
