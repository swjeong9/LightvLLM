---
name: new-kernel
description: 새 CUDA 커널 구현 워크플로우. 6개 파일 생성 순서와 각 파일의 패턴을 안내한다.
---

# 새 CUDA 커널 구현 워크플로우

## 구현 순서

새 커널 `<name>` 구현 시 아래 순서를 따른다:

### Step 1: 순수 CUDA 헤더 — `csrc/<name>_kernels.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cuda_compat.h"  // VLLM_LDG 매크로

namespace lightvllm {

// __device__ 헬퍼 함수 (fp32 중간 연산)
template <typename T>
__device__ __forceinline__ T helper_fn(T x) {
    return (T)(/* fp32 계산 */);
}

// 메인 커널
template <typename scalar_t>
__global__ void kernel_name(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    int d) {
    // Grid: (num_tokens,), Block: (min(d, 1024),)
    const int token_idx = blockIdx.x;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        // 연산
    }
}

}  // namespace lightvllm
```

**한국어 교육 주석 필수**: 수학적 원리, 설계 이유, GPU 최적화 전략 설명.

### Step 2: PyTorch wrapper — `csrc/<name>_kernels.cu`

```cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "dispatch_utils.h"
#include "<name>_kernels.cuh"

void kernel_name(torch::Tensor& out, torch::Tensor& input) {
    int d = input.size(-1);
    int num_tokens = input.numel() / d;
    if (num_tokens == 0) return;

    dim3 grid(num_tokens);
    dim3 block(std::min(d, 1024));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "kernel_name", [&] {
        lightvllm::kernel_name<scalar_t>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                d);
    });
}
```

**한국어 교육 주석 필수**: 해당 연산의 배경, LLM에서의 역할, 메모리 대역폭 분석.

### Step 3: 바인딩 등록

**`csrc/torch_bindings.cpp`**에 추가:
```cpp
// 전방 선언
void kernel_name(torch::Tensor& out, torch::Tensor& input);

// PYBIND11_MODULE 내부
m.def("kernel_name", &kernel_name, "Description");
```

**`setup.py`**에 추가:
```python
sources=[
    ...,
    "csrc/<name>_kernels.cu",  # ← 추가
],
```

### Step 4: Python 래퍼 — `lightvllm/kernels/<name>.py`

```python
import torch
import lightvllm._C as _C

def kernel_name(input: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(input)
    _C.kernel_name(out, input)
    return out
```

### Step 5: nn.Module — `lightvllm/layers/<name>.py`

```python
import torch
import torch.nn as nn
from lightvllm.kernels.<name> import kernel_name

class ModuleName(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # nn.Parameter, register_buffer 등

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernel_name(x, ...)
```

### Step 6: 테스트

**C++ 테스트** — `tests/test_<name>_kernel.cu`:
- CPU 참조 구현 → GPU 커널 결과 비교
- 수학적 성질 검증, 수치 안정성, 성능 벤치마크

**Python 테스트** — `tests/kernels/test_<name>.py`:
```python
class TestKernelName:
    """_C 직접 호출 테스트"""
    def test_basic_correctness(self): ...
    def test_edge_cases(self): ...
    def test_half_dtypes(self): ...      # @pytest.mark.parametrize
    def test_various_sizes(self): ...    # @pytest.mark.parametrize
    def test_performance_vs_pytorch(self): ...

class TestKernelNameWrappers:
    """kernels/ 래퍼 + layers/ Module 테스트"""
    def test_wrapper_function(self): ...
    def test_nn_module(self): ...
```

### Step 7: 빌드 & 검증

```bash
uv run python setup.py build_ext --inplace
nvcc -O2 -std=c++17 -I csrc tests/test_<name>_kernel.cu -o tests/test_<name>_kernel && ./tests/test_<name>_kernel
uv run pytest tests/kernels/test_<name>.py -v
uv run pytest tests/ -v  # 회귀 테스트
```
