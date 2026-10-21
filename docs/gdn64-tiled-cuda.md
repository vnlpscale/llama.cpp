# GDN64: upstream integration and precision-preserving CUDA experiment

## Scope and status

This fork integrates upstream `86a24a182bd3c6c877aaf3734d99f9a1820f8cbd`
with the existing SpeedX27/GDN64 compatibility changes. GDN64 means **64 recurrent
layers**, not a 64-element head. Existing converter aliases, physical-tensor layer
inference, and the fork's CI/release policy are retained. The initial integration
used `08618ff8`; the subsequent one-line upstream Jinja build fix is also merged.

The new CUDA path is **opt-in**: `GGML_CUDA_GDN_TILED=1`. It is an optimization
candidate, not a measured speedup claim. No model weights are requantized. The
upstream path remains the default until correctness and throughput are measured
on the intended NVIDIA GPU and model. CPU regression tests do not validate GPU
execution or real-model perplexity.

## Kernel changes

`ggml/src/ggml-cuda/gated_delta_net_tiled.cu` implements a 128-thread tile:

- A warp processes two state columns, reusing Q/K registers. Eight columns per
  block halve the block count relative to the four-column upstream kernel.
- Aligned `float4` (head size 128) or `float2` (head size 64) state/Q/K loads and
  stores replace scalar access. State traffic itself is not halved.
- State, dot products, and outputs remain FP32. Explicit round-to-nearest FMA
  and FP32 operations avoid introducing FP16/BF16/TF32 recurrent state.
- Decay is computed as `float(exp(double(g)))` once per warp per token and
  broadcast. The source file overrides flush-to-zero and approximate division/
  square-root flags; other CUDA source files retain their existing flags.
- Snapshot slots and padded fused-cache slot strides are preserved.

The dispatcher only selects this path for scalar-gated GDN, head sizes 64/128,
1–8 tokens, a 32-lane NVIDIA warp, and suitably aligned pointers/strides. Vector
KDA gates, other dimensions, offset/unaligned views, HIP/MUSA, and longer batches
use the upstream path. The environment variable is read once per process; use
separate processes for A/B testing. Long-prefill acceleration is not claimed.

Reduction order differs from upstream, so bitwise identity is not promised.
Fewer Q/K load instructions do not imply a proportional end-to-end speedup:
register pressure, cache behavior, launch overhead, decay cost, and matrix
multiplication can dominate. Benchmark rather than assume.

## Model and graph correctness

Pure-recurrent Qwen3.5 graphs keep the hybrid input wrapper but omit its attention
input. This avoids unused KV/position tensors while retaining the correct hybrid
context in graph-reuse checks. A recurrent input object must not reinterpret a
hybrid memory context as a standalone recurrent context.

Legacy files declaring 65 blocks and one absent MTP block retain 64 trunk layers.
Do not zero `nextn_predict_layers`: that would change the trunk layer count.
Loading the trunk without MTP continues to work; explicitly requesting an absent
MTP block now fails with a clear error.

## Build and validation

On a machine with a supported CUDA toolkit and NVIDIA GPU:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release --parallel 2 \
  --target llama-bench llama-completion test-backend-ops test-gdn64-model

# CPU reference, including 1,024 recurrent updates and cache snapshot cases:
./build/bin/test-backend-ops test -b CPU -o 'GATED_DELTA_NET.*'

# CUDA: inspect the backend name and number of executed tests, not just exit code.
GGML_CUDA_GDN_TILED=0 ./build/bin/test-backend-ops test -b CUDA0 -o 'GATED_DELTA_NET.*'
GGML_CUDA_GDN_TILED=1 ./build/bin/test-backend-ops test -b CUDA0 -o 'GATED_DELTA_NET.*'

# Synthetic 64-layer models: ordinary metadata and stale 65/1 MTP metadata.
# The small dependency set can be installed in a virtual environment:
# python -m pip install numpy pyyaml tqdm
python scripts/gdn64-model-smoke.py --build-dir build
GGML_CUDA_GDN_TILED=1 python scripts/gdn64-model-smoke.py --build-dir build --gpu-layers 99

# Correctness-gated A/B tests, microbenchmarks, and real-model throughput JSON:
python scripts/bench-gdn64.py --build-dir build --model /path/to/model.gguf \
  --out gdn64-bench-results

# Opt-in generation; omit the variable or set it to 0 to use upstream dispatch.
GGML_CUDA_GDN_TILED=1 ./build/bin/llama-completion -m /path/to/model.gguf -ngl 99 -p 'Hello'
```

Windows uses `build/bin/Release/*.exe` with a multi-config generator; set
`$env:GGML_CUDA_GDN_TILED="1"` in PowerShell. The Python helpers search both
single-config and Release binary paths. For a non-default NVIDIA device, select
it with `CUDA_VISIBLE_DEVICES` before running and use the resulting `CUDA0` name.

The regression suite adds an independent FP64 recurrence with per-element
`2e-4 + 2e-4*abs(reference)` tolerance for outputs **and** final state, long chains,
near-unit/strong decay, multi-sequence/grouped heads, permuted layouts, token-count
fallback, and rollback snapshots with padded cache strides. This tolerance is a
unit-test criterion, not a real-model quality guarantee. Synthetic model tests
check finite logits, repeated graph reuse, and deterministic state reset; they
are deliberately untrained and do not measure linguistic quality.

### Recorded checks (2026-09-26)

Local Release CPU compilation and all 57 GDN operation tests passed. Both synthetic
64-layer fixtures passed model loading, graph reuse, finite-logit and state-reset
checks. GitHub Actions run `36228882247`, on commit `076f2ac7988e61162b8fc19fb1116e1f6f3bb553`,
independently passed the same 57 tests and both fixtures, and compiled both changed
CUDA translation units with CUDA 12.8 for SM86. Its compiler log confirms the tiled
source's `-ftz=false -prec-div=true -prec-sqrt=true` overrides. The later upstream
Jinja-only change does not alter these CUDA sources and was also rebuilt locally.

The hosted CUDA validation job has no GPU: compilation is not a runtime or
throughput test. Before enabling the path by default, run the complete CUDA tests,
compute-sanitizer, long-context real-model logit/perplexity comparisons, and repeated
pp512/tg128 benchmarks with identical weights, offload, context, and batch settings.
Retain baseline and candidate logs; report prompt processing and token generation
separately. No measured speedup factor or real-model quality guarantee is available.
