"""
LLaMA 모델 구현.

LLaMA 아키텍처의 모델별 클래스들을 정의합니다.
공유 빌딩블록(Linear, MergedLinear, SiluAndMul 등)은 lightvllm/layers/에서 가져오고,
여기서는 LLaMA 고유의 아키텍처(MLP 구성, Attention head 배치 등)를 결정합니다.

현재 구현:
    - LlamaMLP: SwiGLU Feed-Forward Network

향후 추가 예정:
    - LlamaAttention: Multi-Head Attention + RoPE
    - LlamaDecoderLayer: Attention + MLP + RMSNorm
    - LlamaModel: Embedding + DecoderLayer 스택
    - LlamaForCausalLM: LM Head + 가중치 로딩

vLLM 참조: vLLM/vllm/model_executor/models/llama.py
(단순화: TP, 양자화, hidden_act 파라미터 없음)
"""

import torch
import torch.nn as nn

from lightvllm.layers.activation import SiluAndMul
from lightvllm.layers.linear import Linear, MergedLinear


class LlamaMLP(nn.Module):
    """LLaMA의 SwiGLU Feed-Forward Network.

    gate_proj와 up_proj를 MergedLinear로 융합하여 1회 GEMM으로 처리하고,
    SiluAndMul CUDA 커널로 활성화 후, down_proj로 원래 차원을 복원합니다.

    데이터 흐름::

        x: [num_tokens, hidden_size]
            ↓
        gate_up_proj (MergedLinear)  →  [num_tokens, 2 * intermediate_size]
            ↓
        SiluAndMul (CUDA 커널)      →  [num_tokens, intermediate_size]
            ↓
        down_proj (Linear)           →  [num_tokens, hidden_size]

    SwiGLU란?
        기존 FFN:  ReLU(xW₁)W₂
        SwiGLU:    (SiLU(xW_gate) ⊙ xW_up)W_down

        SiLU(x) = x * sigmoid(x) 를 게이트 함수로 사용하고,
        별도의 up_proj 출력과 element-wise 곱셈하는 구조입니다.
        ReLU 대비 학습 품질이 우수하여 LLaMA, Mistral 등에 채택되었습니다.

        참고: "GLU Variants Improve Transformer" (Shazeer, 2020)
        https://arxiv.org/abs/2002.05202

    Args:
        hidden_size: 모델의 hidden 차원 (입력/출력 크기).
            LLaMA-3 8B: 4096, LLaMA-3 70B: 8192, LLaMA-3 405B: 16384.
        intermediate_size: FFN 내부 확장 차원.
            LLaMA-3 8B: 14336 (≈ 3.5 × hidden_size, 8의 배수로 정렬).
        bias: bias 항 포함 여부 (기본값: False, LLaMA는 bias 미사용).
        dtype: 파라미터 dtype (기본값: 현재 default dtype).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # gate_proj + up_proj 를 하나의 weight로 융합
        # 출력: [num_tokens, 2 * intermediate_size]
        # 앞쪽 절반 = gate (SiLU 적용), 뒤쪽 절반 = up (게이트 곱셈 대상)
        self.gate_up_proj = MergedLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=bias,
            dtype=dtype,
        )
        # intermediate → hidden 차원 복원
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
        )
        # SiLU 활성화 + 게이트 곱셈 (CUDA 커널로 융합)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU MLP forward.

        Args:
            x: 입력 텐서 [num_tokens, hidden_size].

        Returns:
            출력 텐서 [num_tokens, hidden_size].
        """
        # 1) gate + up 을 한 번의 GEMM으로 계산
        gate_up = self.gate_up_proj(x)      # [num_tokens, 2 * intermediate_size]
        # 2) SiLU(gate) ⊙ up  (CUDA 커널에서 split + silu + mul 융합)
        activated = self.act_fn(gate_up)     # [num_tokens, intermediate_size]
        # 3) 원래 hidden 차원으로 프로젝션
        output = self.down_proj(activated)   # [num_tokens, hidden_size]
        return output
