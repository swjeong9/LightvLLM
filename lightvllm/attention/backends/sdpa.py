"""
SDPA (Scaled Dot-Product Attention) 백엔드.

PyTorch 2.0+의 F.scaled_dot_product_attention을 래핑합니다.
내부적으로 FlashAttention 또는 Memory-Efficient Attention을 자동 선택합니다.

학습 포인트:
    - Backend 추상화 패턴: 같은 인터페이스(AttentionBackend), 다른 구현.
      NaiveAttention과 SDPAAttention이 동일한 forward() 시그니처를 구현하므로,
      사용하는 쪽(LlamaAttention)은 backend를 교체해도 코드 변경이 없음.

    - SDPA 내부 커널 선택 기준:
      * FlashAttention (sm80+ GPU): tiling 기반, O(N) 메모리
      * Memory-Efficient Attention (xformers): 유사한 최적화
      * Math (naive): 최후 수단, O(N^2) 메모리
      PyTorch가 GPU capability와 텐서 속성에 따라 자동 선택.

    - Naive O(N^2) 메모리 vs FlashAttention O(N) 메모리:
      Naive는 attention score 행렬 [seq_len, seq_len]을 전체 생성.
      FlashAttention은 tiling으로 블록 단위 처리하여 전체 행렬을 메모리에 올리지 않음.

vLLM 참조: vLLM은 SDPA를 직접 사용하지 않음 (자체 FlashAttention 래퍼 사용).
"""

import torch
import torch.nn.functional as F

from lightvllm.attention.backends.base import AttentionBackend


class SDPAAttention(AttentionBackend):
    """PyTorch SDPA 기반 attention.

    F.scaled_dot_product_attention에 위임합니다.
    GQA expand가 필요한 점만 NaiveAttention과 동일하고,
    나머지는 PyTorch 내부 최적화 커널이 처리합니다.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        num_tokens, num_heads, head_dim = query.shape
        seq_len, num_kv_heads, _ = key.shape

        # --- GQA expand ---
        # SDPA는 GQA를 네이티브 지원하지 않으므로 K/V를 확장해야 함.
        # FlashAttention은 네이티브 GQA 지원 → Phase 2에서 확장 불필요해짐.
        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            k = key.unsqueeze(2).expand(-1, -1, num_groups, -1)
            k = k.reshape(seq_len, num_heads, head_dim)
            v = value.unsqueeze(2).expand(-1, -1, num_groups, -1)
            v = v.reshape(seq_len, num_heads, head_dim)
        else:
            k, v = key, value

        # --- Reshape for SDPA: [batch, heads, seq_len, head_dim] ---
        q = query.transpose(0, 1).unsqueeze(0)  # [1, num_heads, num_tokens, head_dim]
        k = k.transpose(0, 1).unsqueeze(0)      # [1, num_heads, seq_len, head_dim]
        v = v.transpose(0, 1).unsqueeze(0)       # [1, num_heads, seq_len, head_dim]

        # --- SDPA ---
        # is_causal=True: 내부적으로 causal mask를 자동 생성.
        #   주의: Q_len == K_len일 때만 올바름 (prefill).
        #   decode (Q_len=1, K_len>1)에서는 반드시 is_causal=False.
        #   → LlamaAttention에서 is_causal=(kv_cache is None)으로 결정.
        #
        # scale: 1/sqrt(head_dim)을 자동 계산.
        output = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal,
        )
        # output: [1, num_heads, num_tokens, head_dim]

        # --- Reshape back ---
        output = output.squeeze(0).transpose(0, 1)
        # output: [num_tokens, num_heads, head_dim]

        return output
