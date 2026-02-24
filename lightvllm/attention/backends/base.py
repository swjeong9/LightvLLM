"""
Attention 백엔드 추상 클래스.

모든 attention 구현은 이 ABC를 상속하여 forward()를 구현합니다.
vLLM의 AttentionBackend + AttentionImpl을 교육용으로 단순화한 버전입니다.

vLLM에서는 AttentionBackend(정적 메타 클래스)와 AttentionImpl(연산 클래스)이
분리되어 있지만, Phase 1에서는 하나의 ABC로 통합합니다.

향후 Phase 2에서 FlashAttention, Paged Attention을 추가할 때
이 인터페이스를 확장할 수 있습니다.

vLLM 참조: vLLM/vllm/attention/backends/abstract.py
(단순화: AttentionMetadata, AttentionType, 양자화, 분산처리 없음)
"""

from abc import ABC, abstractmethod

import torch


class AttentionBackend(ABC):
    """Attention 백엔드 추상 클래스.

    모든 attention 구현(Naive, SDPA, FlashAttention 등)이 공유하는 인터페이스.
    Backend는 다음을 계산합니다::

        output = softmax(Q @ K^T / sqrt(head_dim)) @ V

    GQA (Grouped Query Attention) 지원:
        num_kv_heads < num_heads일 때, K/V를 내부적으로 확장합니다.
        예) LLaMA-3.2-3B: num_heads=24, num_kv_heads=8, ratio=3.

    입력 텐서는 호출자(LlamaAttention)가 이미 3D로 reshape한 상태입니다.
    """

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Scaled dot-product attention 계산.

        Args:
            query: Query 텐서 [num_tokens, num_heads, head_dim].
                Prefill: num_tokens == seq_len.
                Decode:  num_tokens == 1.
            key: Key 텐서 [seq_len, num_kv_heads, head_dim].
                KV cache 사용 시 전체 캐시 포함.
            value: Value 텐서 [seq_len, num_kv_heads, head_dim].
            is_causal: Causal mask 적용 여부.
                Prefill: True (미래 토큰 참조 방지).
                Decode with KV cache: False (새 토큰이 전체 캐시에 attend).

        Returns:
            Attention 출력 [num_tokens, num_heads, head_dim].
        """
        ...
