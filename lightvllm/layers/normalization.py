"""
정규화 레이어: RMSNorm

LLaMA, Mistral 등 최신 LLM에서 사용하는 RMSNorm 레이어를 제공합니다.
CUDA 커널을 사용하여 GPU에서 효율적으로 실행됩니다.
"""

import torch
import torch.nn as nn

from lightvllm.kernels.layernorm import rms_norm, fused_add_rms_norm


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    output = weight * input / sqrt(mean(input²) + epsilon)

    LLaMA, Mistral 등 최신 LLM에서 표준 LayerNorm 대신 사용합니다.
    RMSNorm은 mean centering을 생략하여 리덕션 1회를 절약하면서도
    동등한 학습 품질을 달성합니다.

    참고: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467

    Args:
        hidden_size: 입력 텐서의 마지막 차원 크기
        eps: 수치 안정성 상수 (기본값: 1e-6)
        dtype: 가중치 파라미터의 데이터 타입 (기본값: 현재 기본 dtype)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        weight_dtype = dtype or torch.get_default_dtype()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=weight_dtype)
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm 적용. residual이 주어지면 Fused Add+RMSNorm 사용.

        Args:
            x: 입력 텐서 [num_tokens, hidden_size]
            residual: (선택) 잔차 텐서 [num_tokens, hidden_size].
                      주어지면 fused add+rms_norm 커널 사용.

        Returns:
            residual이 None인 경우: 정규화된 텐서
            residual이 주어진 경우: (정규화된 출력, 업데이트된 잔차) 튜플
        """
        if residual is not None:
            return fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}"
