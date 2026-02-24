"""
KV Cache: Contiguous buffer for autoregressive generation.

Prefill 단계에서 전체 시퀀스의 K/V를 캐시에 저장하고,
Decode 단계에서 새 토큰의 K/V를 추가하며 전체 히스토리를 반환합니다.

왜 Q는 캐시하지 않는가?
    Q는 항상 "현재 토큰"만 필요합니다.
    K/V는 attention 계산에서 "이전 전체 토큰"이 필요하므로 캐시가 필수입니다.
    이것이 KV Cache이지 QKV Cache가 아닌 이유입니다.

Prefill vs Decode::

    Prefill: "오늘 날씨가 어때?"  (5토큰 한 번에 처리)
        → K/V [5, num_kv_heads, head_dim]을 캐시에 저장
        → Q [5, num_heads, head_dim]은 캐시하지 않음

    Decode: "맑" (1토큰 생성)
        → 새 K/V [1, ...]을 캐시에 추가
        → 전체 K/V [6, ...]을 반환 (기존 5 + 새 1)
        → Q [1, ...]로 전체 K/V에 attend

Contiguous vs Paged:
    이 구현은 연속 메모리(contiguous) 방식입니다.
    max_seq_len만큼 미리 할당하므로 메모리 낭비가 발생할 수 있습니다.
    Phase 2에서 vLLM의 핵심 기술인 Paged Attention으로 발전합니다.
    Paged: 고정 크기 블록 단위로 동적 할당 → 메모리 단편화 해결.

vLLM 참조: vLLM/csrc/cache_kernels.cu, vLLM/vllm/v1/core/block_pool.py
(단순화: paging, block table, 배치 없음)
"""

import torch


class KVCache:
    """모든 layer의 KV를 관리하는 contiguous 캐시 버퍼.

    Buffer shape: [num_layers, 2, max_seq_len, num_kv_heads, head_dim]
                   2 = (key: index 0, value: index 1)

    사용 예시::

        cache = KVCache(num_layers=28, max_seq_len=2048,
                        num_kv_heads=8, head_dim=128)

        # Prefill: 프롬프트 전체의 K/V 저장
        k_all, v_all = cache.update(layer_idx=0, key=k, value=v)
        # ... 모든 layer에 대해 update() 호출 ...
        cache.advance(prompt_len)  # step당 1회만!

        # Decode: 1토큰의 K/V 추가
        k_all, v_all = cache.update(layer_idx=0, key=k_new, value=v_new)
        # ... 모든 layer에 대해 update() 호출 ...
        cache.advance(1)

    Args:
        num_layers: Transformer layer 수.
        max_seq_len: 캐시가 보유할 수 있는 최대 시퀀스 길이.
        num_kv_heads: KV attention head 수.
        head_dim: Head당 차원.
        dtype: 캐시 데이터 타입 (기본값: bfloat16).
        device: 캐시 디바이스 (기본값: cuda).
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 전체 캐시를 미리 할당 (contiguous)
        # [num_layers, 2, max_seq_len, num_kv_heads, head_dim]
        # dim 1의 index 0 = key, index 1 = value
        self._cache = torch.zeros(
            num_layers, 2, max_seq_len, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )

        # Write pointer: 현재까지 저장된 토큰 수
        self._seq_len = 0

    @property
    def current_seq_len(self) -> int:
        """현재 캐시에 저장된 토큰 수."""
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """새 K/V를 저장하고 전체 캐시 히스토리를 반환.

        Prefill에서는 num_new_tokens이 프롬프트 길이,
        Decode에서는 1입니다.

        주의: advance()는 이 메서드와 별도로 호출해야 합니다.
        모든 layer의 update()가 끝난 후 advance()를 1회만 호출하세요.

        Args:
            layer_idx: Transformer layer 인덱스 (0-indexed).
            key: 새 key 텐서 [num_new_tokens, num_kv_heads, head_dim].
            value: 새 value 텐서 [num_new_tokens, num_kv_heads, head_dim].

        Returns:
            (all_keys, all_values): 새 토큰을 포함한 전체 캐시.
            all_keys:   [total_seq_len, num_kv_heads, head_dim]
            all_values: [total_seq_len, num_kv_heads, head_dim]
            (버퍼의 view이며, 메모리 복사가 아님)
        """
        num_new_tokens = key.shape[0]
        start = self._seq_len
        end = self._seq_len + num_new_tokens

        assert end <= self.max_seq_len, (
            f"KV cache overflow: {end} > {self.max_seq_len}"
        )

        # 새 K/V를 버퍼에 기록
        self._cache[layer_idx, 0, start:end] = key
        self._cache[layer_idx, 1, start:end] = value

        # 새 토큰을 포함한 전체 히스토리 반환 (view)
        all_keys = self._cache[layer_idx, 0, :end]
        all_values = self._cache[layer_idx, 1, :end]

        return all_keys, all_values

    def advance(self, num_tokens: int) -> None:
        """Write pointer를 전진시킴.

        Forward step당 1회만 호출합니다 (layer당 아님).
        모든 layer가 같은 수의 토큰을 처리하므로,
        pointer는 step 수준에서 관리합니다.

        Args:
            num_tokens: 이번 step에서 처리한 토큰 수.
                Prefill: 프롬프트 길이.
                Decode: 1.
        """
        self._seq_len += num_tokens
        assert self._seq_len <= self.max_seq_len, (
            f"KV cache overflow: {self._seq_len} > {self.max_seq_len}"
        )

    def reset(self) -> None:
        """새 시퀀스를 위해 캐시 초기화.

        버퍼를 0으로 채우고 write pointer를 리셋합니다.
        """
        self._cache.zero_()
        self._seq_len = 0
