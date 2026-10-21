# GDN64 upstream integration — 2026-10-21

## Provenance

This integration targets ggml-org/llama.cpp master at `0a1ef309cdbb9f2024f58a38fb4c42352c5db569` (2026-10-21).
It preserves the non-destructive fork master and original draft PR #6.
The integration is structured with upstream as the source tree, and GDN64-related
changes cherry-picked as individual files, rather than re-introducing the old
upstream snapshot.

## Included

- Keep upstream recent CUDA GDN MMA path, grouped warp implementation, and
  other upstream changes.
- Opt-in FP32 tiled GDN kernel `GGML_CUDA_GDN_TILED=1` with fallback for
  unsupported shapes, KDA modes, and non-NVIDIA backends.
- Qwen3.5 physical SSM layer detection in addition to interval metadata.
- Pure recurrent hybrid graph input avoids allocating attention/KV inputs.
- Preserve tests, synthetic model smoke generator, benchmark helper, patch
  history and focused fork CI.

## Compatibility notes

Upstream already uses `nextn_flags(ml)` to handle physically absent NextN/MTP
weights. That is retained instead of overriding the newer helper.
Upstream's hybrid input implementation must check for a null attention input
on both `set_input` and `can_reuse`. The graph interface now takes a
`has_attn` argument with a default for other model callers.

## Validation status

Source integration prepared; **NOT YET compiled or runtime-tested** on the
2026-10-21 upstream version. Earlier 2026-09-26 CPU and CUDA compile
verification belongs to PR #6, and does not validate this integration.

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build --parallel 2 --target test-backend-ops test-gdn64-model
./build/bin/test-backend-ops test -b CPU -o 'GATED_DELTA_NET.*'
python3 scripts/gdn64-model-smoke.py --build-dir build
```

For GPU validation use the linked GDN64 workflow; GPU hardware is needed for
runtime correctness, sanitizer and actual end-to-end pp512/tg128 throughput.
The experimental tiled kernel remains disabled by default.
