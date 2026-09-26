#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s synthetic-model.gguf [gpu-layers]\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = argc == 3 ? std::atoi(argv[2]) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { llama_backend_free(); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 8;
    cp.n_ubatch = 8;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); llama_backend_free(); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> first_pass;
    bool ok = true;
    for (int pass = 0; pass < 2 && ok; ++pass) {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int step = 0; step < 16 && ok; ++step) {
            // Repeated one-token calls exercise graph reuse and recurrent-state updates.
            llama_token token = (37 + step) % n_vocab;
            if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) { ok = false; break; }
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) { ok = false; break; }
            for (int i = 0; i < n_vocab; ++i) {
                if (!std::isfinite(logits[i])) { ok = false; break; }
                if (pass == 1) {
                    const float reference = first_pass[size_t(step) * n_vocab + i];
                    if (std::abs(logits[i] - reference) > 1e-5f + 1e-5f * std::abs(reference)) {
                        ok = false;
                        break;
                    }
                }
            }
            if (pass == 0) { first_pass.insert(first_pass.end(), logits, logits + n_vocab); }
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::printf("GDN64 model load, graph reuse, finite logits and state reset: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
