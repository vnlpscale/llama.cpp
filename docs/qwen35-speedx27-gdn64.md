# Qwen35 SpeedX27 GDN64 compatibility

This note documents the compatibility fixes required for Qwen35-derived models whose main decoder stack is entirely Gated DeltaNet (GDN), while GGUF metadata still contains a NextN/MTP layer count and a full-attention interval.

## Reproduced model layout

The affected model uses:

- `general.architecture = qwen35`
- `qwen35.block_count = 65`
- `qwen35.nextn_predict_layers = 1`
- 64 physical main decoder blocks: `blk.0` through `blk.63`
- all 64 main blocks contain GDN/SSM tensors
- no physical `blk.64` MTP tensors

For this layout, `n_layer()` must remain `65 - 1 = 64`. The NextN count must not be reset to zero.

## Required loader fix

When `qwen35.attention.recurrent_layers` is absent, the current Qwen35 fallback uses `full_attention_interval` alone. This can incorrectly classify GDN blocks as full-attention blocks. The robust fallback should prefer the actual tensor layout: if `blk.N.ssm_conv1d.weight` exists, classify that main block as recurrent. Use `full_attention_interval` only when tensor evidence is unavailable.

MTP tensor loading should also tolerate metadata that declares one NextN layer when the corresponding physical `blk.64` tensor set is absent. The main decoder pass already excludes NextN layers via `n_layer()`.

## Required all-recurrent hybrid-memory fix

An all-recurrent hybrid model legitimately has zero standard attention KV-cache layers. The following KV input setters must treat an empty attention-layer set as a no-op instead of dereferencing an unallocated backend buffer:

- `llama_kv_cache::set_input_k_idxs`
- `llama_kv_cache::set_input_v_idxs`
- `llama_kv_cache::set_input_kq_mask`

The Qwen35 graph should avoid creating an RoPE position input when no full-attention layer exists. As an additional defensive check, `llm_graph_input_pos::set_input()` must not call `ggml_backend_tensor_set()` when `pos->buffer` is null, because dead graph inputs can remain unallocated.

## Observed failures before the fixes

1. Qwen35 interval fallback misclassified `blk.3` as full attention and expected a 14336-wide QKV tensor instead of the model's 10240-wide GDN projection.
2. Resetting `n_layer_nextn` to zero made recurrent memory allocate 65 layers and include nonexistent layer 64.
3. With all 64 main layers correctly recurrent, the empty attention KV cache caused null-buffer assertions in `set_input_k_idxs`, `set_input_v_idxs`, and `set_input_kq_mask`.
4. After those KV inputs were skipped, an unallocated dead position input caused `ggml_backend_tensor_set()` to assert with `tensor buffer not set`.

## Validation target

A successful run should report an effective main-layer count of 64, allocate recurrent state for 64 layers, create no standard attention KV layers, and decode a text prompt without a backend buffer assertion.
