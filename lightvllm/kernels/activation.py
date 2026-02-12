"""
Activation 커널: SiLU, SiLU+Mul (Fused Gate)

CUDA 커널을 래핑하는 Python 함수를 제공합니다.
lightvllm._C 모듈을 통해 커널을 호출합니다.
"""

import torch
import lightvllm._C as _C


def silu(input: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit) 활성화 함수.

    output = input * sigmoid(input)

    Args:
        input: 입력 텐서 [..., d]

    Returns:
        활성화된 텐서 (입력과 동일한 shape, dtype)
    """
    out = torch.empty_like(input)
    _C.silu(out, input)
    return out


def silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """Fused SiLU + Mul (LLaMA MLP 게이트 연산).

    입력의 마지막 차원을 반으로 나누어:
    output = silu(input[..., :d]) * input[..., d:]

    LLaMA MLP에서 gate_proj와 up_proj의 결합된 출력에 적용합니다.

    Args:
        input: 입력 텐서 [..., 2*d]

    Returns:
        게이트된 텐서 [..., d] (마지막 차원이 절반)
    """
    d = input.shape[-1] // 2
    output_shape = input.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    _C.silu_and_mul(out, input)
    return out
