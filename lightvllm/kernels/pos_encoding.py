"""
Position Encoding 커널: Rotary Position Embedding (RoPE)

CUDA 커널을 래핑하는 Python 함수를 제공합니다.
lightvllm._C 모듈을 통해 커널을 호출합니다.
"""

import torch
import lightvllm._C as _C


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """Rotary Position Embedding (RoPE) 적용 (in-place).

    query와 key 텐서에 위치 정보를 인코딩합니다.
    cos/sin 캐시에서 position에 해당하는 값을 읽어 회전 변환을 적용합니다.

    Args:
        positions: 각 토큰의 위치 인덱스 [num_tokens] (int64)
        query: 쿼리 텐서 [num_tokens, num_heads * head_size] (in-place 수정)
        key: 키 텐서 [num_tokens, num_kv_heads * head_size] (in-place 수정)
        head_size: 각 attention head의 차원 크기
        cos_sin_cache: cos/sin 캐시 [max_position, rot_dim]
        is_neox: GPT-NeoX 스타일 회전 적용 (기본값: True)
    """
    _C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
