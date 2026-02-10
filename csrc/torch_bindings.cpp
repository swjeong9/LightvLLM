/**
 * PyTorch C++ bindings for LightvLLM CUDA kernels.
 *
 * This file registers all custom CUDA operations with PyTorch
 * using the torch::library mechanism.
 */

#include <torch/extension.h>

// Position encoding kernels
void rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Activation kernels
    // m.def("silu", &silu, "SiLU activation");
    // m.def("silu_and_mul", &silu_and_mul, "SiLU activation with multiplication");

    // LayerNorm kernels
    // m.def("rms_norm", &rms_norm, "RMS Normalization");
    // m.def("fused_add_rms_norm", &fused_add_rms_norm, "Fused Add + RMS Norm");

    // Position encoding kernels
    m.def("rotary_embedding", &rotary_embedding, "Rotary Position Embedding");
}

