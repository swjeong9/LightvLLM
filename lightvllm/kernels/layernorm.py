"""
LayerNorm 커널: RMSNorm, Fused Add+RMSNorm

CUDA 커널을 래핑하는 Python 함수를 제공합니다.
lightvllm._C 모듈을 통해 커널을 호출합니다.
"""

import torch
import lightvllm._C as _C


def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """RMS Normalization 적용.

    output = input / sqrt(mean(input²) + epsilon) * weight

    Args:
        input: 입력 텐서 [num_tokens, hidden_size]
        weight: 학습 가능한 가중치 [hidden_size]
        epsilon: 수치 안정성 상수 (기본값: 1e-6)

    Returns:
        정규화된 텐서 (입력과 동일한 shape, dtype)
    """
    out = torch.empty_like(input)
    _C.rms_norm(out, input, weight, epsilon)
    return out


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused 잔차 덧셈 + RMS Normalization.

    residual = residual + input
    output = rms_norm(residual) * weight

    두 텐서 모두 in-place로 수정됩니다:
      - input 버퍼: 정규화 결과로 덮어씌워짐
      - residual 버퍼: (원래 input + 원래 residual)로 업데이트됨

    Args:
        input: 입력 텐서 (정규화 결과로 덮어씌워짐) [num_tokens, hidden_size]
        residual: 잔차 텐서 (in-place 업데이트) [num_tokens, hidden_size]
        weight: 학습 가능한 가중치 [hidden_size]
        epsilon: 수치 안정성 상수 (기본값: 1e-6)

    Returns:
        (정규화된 출력, 업데이트된 잔차) 튜플.
        입력과 동일한 텐서 객체 (in-place 수정됨).
    """
    _C.fused_add_rms_norm(input, residual, weight, epsilon)
    return input, residual
