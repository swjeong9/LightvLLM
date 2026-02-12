"""
위치 인코딩 레이어: Rotary Position Embedding (RoPE)

LLaMA, Mistral 등 최신 LLM에서 사용하는 RoPE 레이어를 제공합니다.
CUDA 커널을 사용하여 GPU에서 효율적으로 실행됩니다.
"""

import torch
import torch.nn as nn

from lightvllm.kernels.pos_encoding import rotary_embedding


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    cos/sin 캐시를 미리 생성하고, forward에서 query/key에 회전 변환을 적용합니다.

    참고: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021) https://arxiv.org/abs/2104.09864

    Args:
        head_size: 각 attention head의 차원 크기
        rotary_dim: 회전을 적용할 차원 수 (기본값: head_size)
        max_position_embeddings: 최대 위치 인덱스 (기본값: 8192)
        base: 주파수 기저값 (기본값: 10000.0)
        is_neox_style: GPT-NeoX 스타일 회전 적용 (기본값: True)
        dtype: 캐시 데이터 타입 (기본값: bfloat16)
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int | None = None,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim or head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style

        # cos/sin 캐시 생성
        cache = self._build_cache(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _build_cache(self, dtype: torch.dtype) -> torch.Tensor:
        """cos/sin 캐시 생성.

        캐시 레이아웃: [max_position, rotary_dim]
          각 행: [cos(θ_0), ..., cos(θ_{d/2-1}), sin(θ_0), ..., sin(θ_{d/2-1})]
        """
        embed_dim = self.rotary_dim // 2
        positions = torch.arange(
            self.max_position_embeddings, dtype=torch.float32
        )
        dims = torch.arange(embed_dim, dtype=torch.float32)
        freqs = positions.unsqueeze(1) * (
            self.base ** (-2.0 * dims / self.rotary_dim)
        ).unsqueeze(0)

        cache = torch.empty(
            self.max_position_embeddings, self.rotary_dim, dtype=dtype
        )
        cache[:, :embed_dim] = freqs.cos().to(dtype)
        cache[:, embed_dim:] = freqs.sin().to(dtype)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> None:
        """RoPE를 query와 key에 적용 (in-place).

        Args:
            positions: 각 토큰의 위치 인덱스 [num_tokens] (int64)
            query: 쿼리 텐서 [num_tokens, num_heads * head_size] (in-place 수정)
            key: 키 텐서 [num_tokens, num_kv_heads * head_size] (in-place 수정)
        """
        rotary_embedding(
            positions, query, key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
