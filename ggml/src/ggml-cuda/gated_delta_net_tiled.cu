#include "gated_delta_net_tiled.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

template <int R>
static __device__ __forceinline__ void gdn_tiled_load(const float * p, int lane, float (&x)[R]) {
    if constexpr (R == 4) {
        const float4 v = reinterpret_cast<const float4 *>(p)[lane];
        x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
    } else {
        const float2 v = reinterpret_cast<const float2 *>(p)[lane];
        x[0] = v.x; x[1] = v.y;
    }
}

template <int R>
static __device__ __forceinline__ void gdn_tiled_store(float * p, int lane, const float (&x)[R]) {
    if constexpr (R == 4) {
        reinterpret_cast<float4 *>(p)[lane] = make_float4(x[0], x[1], x[2], x[3]);
    } else {
        reinterpret_cast<float2 *>(p)[lane] = make_float2(x[0], x[1]);
    }
}

static __device__ __forceinline__ float gdn_tiled_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, offset));
    }
    return x;
}

template <int S, bool keep_rs>
__global__ void __launch_bounds__(128, 2) gated_delta_net_tiled(ggml_cuda_gdn_tiled_args a) {
    static_assert(S == 64 || S == 128, "unsupported GDN tile");
    constexpr int R = S / 32;
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.z * 8 + (threadIdx.x / 32) * 2;
    const int64_t h = blockIdx.x;
    const int64_t seq = blockIdx.y;
    const uint32_t iq1 = fastmodulo((uint32_t) h, a.neqk1_magic);
    const uint32_t iq3 = fastdiv((uint32_t) seq, a.rq3_magic);
    const int64_t state_offset = (seq * a.H + h) * S * S;
    float * state_out = a.state_out + state_offset;
    float * out = a.dst + (seq * a.n_tokens * a.H + h) * S;
    float s[2][R];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        gdn_tiled_load(a.state_in + state_offset + (col + c) * S, lane, s[c]);
    }

    for (int64_t t = 0; t < a.n_tokens; ++t) {
        float q[R], k[R];
        gdn_tiled_load(a.q + iq3 * a.sq3 + t * a.sq2 + iq1 * a.sq1, lane, q);
        gdn_tiled_load(a.k + iq3 * a.sq3 + t * a.sq2 + iq1 * a.sq1, lane, k);
        const int64_t gb = seq * a.sb3 + t * a.sb2 + h * a.sb1;
        const float * v = a.v + seq * a.sv3 + t * a.sv2 + h * a.sv1;

        // Do not let -use_fast_math turn the decay into __expf. Only lane 0 evaluates exp.
        float decay = lane == 0 ? (float) exp((double) a.g[gb]) : 0.0f;
        decay = __shfl_sync(0xffffffffu, decay, 0);
        float beta = lane == 0 ? a.beta[gb] : 0.0f;
        beta = __shfl_sync(0xffffffffu, beta, 0);

#pragma unroll
        for (int c = 0; c < 2; ++c) {
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                kv = __fmaf_rn(s[c][r], k[r], kv);
            }
            kv = gdn_tiled_sum(kv);
            float delta = lane == 0 ? __fmul_rn(__fmaf_rn(-decay, kv, v[col + c]), beta) : 0.0f;
            delta = __shfl_sync(0xffffffffu, delta, 0);

            float y = 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                s[c][r] = __fmaf_rn(k[r], delta, __fmul_rn(decay, s[c][r]));
                y = __fmaf_rn(s[c][r], q[r], y);
            }
            y = gdn_tiled_sum(y);
            if (lane == 0) {
                out[col + c] = __fmul_rn(y, a.scale);
            }
        }
        out += S * a.H;

        if constexpr (keep_rs) {
            const int64_t slot = a.n_tokens - 1 - t;
            if (slot < a.K) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    gdn_tiled_store(state_out + slot * a.state_slot_stride + (col + c) * S, lane, s[c]);
                }
            }
        }
    }
    if constexpr (!keep_rs) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            gdn_tiled_store(state_out + (col + c) * S, lane, s[c]);
        }
    }
}

template <int S>
static void launch_gdn_tiled(const ggml_cuda_gdn_tiled_args & a, cudaStream_t stream) {
    const ggml_cuda_kernel_launch_params params(dim3(a.H, a.n_seqs, S / 8), dim3(128), 0, stream);
    if (a.K > 1) {
        ggml_cuda_kernel_launch(gated_delta_net_tiled<S, true>, params, a);
    } else {
        ggml_cuda_kernel_launch(gated_delta_net_tiled<S, false>, params, a);
    }
}

#endif

bool ggml_cuda_try_gdn_tiled(const ggml_cuda_gdn_tiled_args & a, int64_t head_size, cudaStream_t stream) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Opt-in until device-specific correctness and throughput have been measured.
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_GDN_TILED");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (!enabled || a.n_tokens < 1 || a.n_tokens > 8 || (head_size != 64 && head_size != 128)) {
        return false;
    }
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 32) {
        return false;
    }
    // Contiguous rows need not be vector-aligned (e.g. offset views).
    const uintptr_t ptr_bits = reinterpret_cast<uintptr_t>(a.q) | reinterpret_cast<uintptr_t>(a.k) |
                               reinterpret_cast<uintptr_t>(a.state_in) | reinterpret_cast<uintptr_t>(a.state_out);
    if ((ptr_bits & 15) || ((a.sq1 | a.sq2 | a.sq3 | a.state_slot_stride) & 3)) {
        return false;
    }
    if (head_size == 128) {
        launch_gdn_tiled<128>(a, stream);
    } else {
        launch_gdn_tiled<64>(a, stream);
    }
    return true;
#else
    GGML_UNUSED(a);
    GGML_UNUSED(head_size);
    GGML_UNUSED(stream);
    return false;
#endif
}
