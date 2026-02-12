"""
활성화 레이어: SiluAndMul

LLaMA MLP에서 사용하는 SiLU + Gated Linear Unit 레이어를 제공합니다.
CUDA 커널을 사용하여 GPU에서 효율적으로 실행됩니다.
"""

import torch
import torch.nn as nn

from lightvllm.kernels.activation import silu_and_mul


class SiluAndMul(nn.Module):
    """SiLU activation with gating (SwiGLU).

    입력의 마지막 차원을 반으로 나누어 게이트 연산을 수행합니다:
    output = silu(x[..., :d]) * x[..., d:]

    LLaMA의 MLP 블록에서 gate_proj와 up_proj의 결합된 출력에 적용합니다.
    SwiGLU (Shazeer, 2020) 논문에서 제안된 구조입니다.

    참고: "GLU Variants Improve Transformer" (Shazeer, 2020)
    https://arxiv.org/abs/2002.05202

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)
