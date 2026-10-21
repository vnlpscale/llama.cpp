#pragma once

#include "common.cuh"

struct ggml_cuda_gdn_tiled_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state_in;
    float * dst;
    float * state_out;
    int64_t H, n_tokens, n_seqs;
    int64_t sq1, sq2, sq3;
    int64_t sv1, sv2, sv3;
    int64_t sb1, sb2, sb3;
    uint3 neqk1_magic, rq3_magic;
    float scale;
    int64_t state_slot_stride;
    int K;
};

bool ggml_cuda_try_gdn_tiled(const ggml_cuda_gdn_tiled_args & args, int64_t head_size, cudaStream_t stream);
