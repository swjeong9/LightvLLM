"""Attention Backend + KV Cache 테스트.

NaiveAttention, SDPAAttention, KVCache의 정확성을 검증합니다.

Llama-3.2-3B 스펙:
    num_attention_heads=24, num_kv_heads=8, head_dim=128
    GQA ratio: 24/8 = 3

실행:
    uv run pytest tests/attention/test_attention.py -v

실행 (특정 클래스만):
    uv run pytest tests/attention/test_attention.py::TestNaiveAttention -v

벤치마크만:
    uv run pytest tests/attention/test_attention.py::TestAttentionBenchmark -v -s
"""

import pytest
import torch

from lightvllm.attention.backends.naive import NaiveAttention
from lightvllm.attention.backends.sdpa import SDPAAttention
from lightvllm.attention.kv_cache import KVCache

DEVICE = "cuda"
DEFAULT_DTYPE = torch.bfloat16

# Attention은 softmax + matmul 누적으로 오차가 element-wise보다 큼
TOLERANCES = {
    torch.float32: {"atol": 1e-4, "rtol": 1e-4},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}


# =============================================================================
# 참조 구현
# =============================================================================


def attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """fp32 참조 attention 구현.

    GPU에서 fp32로 계산하여 NaiveAttention/SDPAAttention의 결과를 검증합니다.

    Args:
        query: [num_tokens, num_heads, head_dim]
        key:   [seq_len, num_kv_heads, head_dim]
        value: [seq_len, num_kv_heads, head_dim]
        is_causal: causal mask 적용 여부.

    Returns:
        [num_tokens, num_heads, head_dim] (fp32)
    """
    num_tokens, num_heads, head_dim = query.shape
    seq_len, num_kv_heads, _ = key.shape
    scale = head_dim ** -0.5

    # fp32로 변환
    q = query.float().transpose(0, 1).unsqueeze(0)
    k = key.float().transpose(0, 1).unsqueeze(0)
    v = value.float().transpose(0, 1).unsqueeze(0)

    # GQA expand
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        k = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        k = k.reshape(1, num_heads, seq_len, head_dim)
        v = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        v = v.reshape(1, num_heads, seq_len, head_dim)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones(num_tokens, seq_len, device=query.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    scores = torch.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)

    return output.squeeze(0).transpose(0, 1)


# =============================================================================
# NaiveAttention 테스트
# =============================================================================


class TestNaiveAttention:
    """NaiveAttention 기능 테스트."""

    def test_basic_mha(self):
        """기본 MHA (GQA 없음, num_heads == num_kv_heads)."""
        num_tokens, num_heads, head_dim = 16, 8, 64
        backend = NaiveAttention()

        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        k = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)

        out = backend.forward(q, k, v, is_causal=True)

        assert out.shape == (num_tokens, num_heads, head_dim)
        assert out.dtype == DEFAULT_DTYPE

        ref = attention_reference(q, k, v, is_causal=True).to(DEFAULT_DTYPE)
        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(out, ref, **tol)

    def test_gqa(self):
        """GQA (num_kv_heads < num_heads) 정확성 검증.

        Llama-3.2-3B: num_heads=24, num_kv_heads=8, ratio=3.
        """
        num_tokens = 16
        num_heads, num_kv_heads, head_dim = 24, 8, 128
        backend = NaiveAttention()

        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        k = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)

        out = backend.forward(q, k, v, is_causal=True)

        assert out.shape == (num_tokens, num_heads, head_dim)

        ref = attention_reference(q, k, v, is_causal=True).to(DEFAULT_DTYPE)
        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(out, ref, **tol)

    def test_causal_mask(self):
        """Causal mask가 미래 토큰 attention을 차단하는지 검증.

        각 토큰은 자기 자신과 이전 토큰에만 attend해야 함.
        """
        num_tokens, num_heads, head_dim = 8, 4, 32
        backend = NaiveAttention()

        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=torch.float32, device=DEVICE)
        k = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=torch.float32, device=DEVICE)
        # value를 위치별로 구분 가능하게 설정
        v = torch.zeros(num_tokens, num_heads, head_dim,
                         dtype=torch.float32, device=DEVICE)
        # 마지막 토큰의 value만 1로 설정
        v[-1] = 1.0

        out = backend.forward(q, k, v, is_causal=True)

        # 마지막 토큰 이전의 토큰들은 마지막 토큰에 attend할 수 없음
        # → value가 0인 토큰들에만 attend → 마지막을 제외한 출력에 v[-1]의 영향 없음
        # 마지막 토큰만 자신의 value(1.0)에 접근 가능
        for i in range(num_tokens - 1):
            # 토큰 i는 토큰 0~i에만 attend. v[0:i]은 모두 0이므로 출력도 0.
            torch.testing.assert_close(
                out[i],
                torch.zeros(num_heads, head_dim, device=DEVICE),
                atol=0, rtol=0,
            )

    def test_single_token_decode(self):
        """Decode 모드: 1개 query 토큰이 전체 seq에 attend.

        is_causal=False로 모든 캐시에 attend.
        """
        num_heads, num_kv_heads, head_dim = 8, 4, 64
        seq_len = 32
        backend = NaiveAttention()

        q = torch.randn(1, num_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        k = torch.randn(seq_len, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(seq_len, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)

        out = backend.forward(q, k, v, is_causal=False)

        assert out.shape == (1, num_heads, head_dim)

        ref = attention_reference(q, k, v, is_causal=False).to(DEFAULT_DTYPE)
        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(out, ref, **tol)

    def test_softmax_fp32(self):
        """fp16 입력에서 softmax overflow 없음 확인.

        큰 attention score에서도 NaN/Inf가 발생하지 않아야 함
        (내부적으로 fp32 softmax 사용).
        """
        num_tokens, num_heads, head_dim = 8, 4, 64
        backend = NaiveAttention()

        # 큰 값의 query로 attention score를 크게 만듦
        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=torch.float16, device=DEVICE) * 10.0
        k = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=torch.float16, device=DEVICE) * 10.0
        v = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=torch.float16, device=DEVICE)

        out = backend.forward(q, k, v, is_causal=True)

        assert not torch.isnan(out).any(), "출력에 NaN이 있습니다"
        assert not torch.isinf(out).any(), "출력에 Inf가 있습니다"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtypes(self, dtype):
        """fp16, bf16 dtype별 정확성 검증."""
        num_tokens, num_heads, head_dim = 16, 8, 64
        backend = NaiveAttention()

        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=dtype, device=DEVICE)
        k = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=dtype, device=DEVICE)
        v = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=dtype, device=DEVICE)

        out = backend.forward(q, k, v, is_causal=True)

        assert out.dtype == dtype

        ref = attention_reference(q, k, v, is_causal=True).to(dtype)
        tol = TOLERANCES[dtype]
        torch.testing.assert_close(out, ref, **tol)

    def test_llama3_dimensions(self):
        """Llama-3.2-3B 실제 차원에서 정상 동작 확인."""
        num_tokens = 32
        num_heads, num_kv_heads, head_dim = 24, 8, 128
        backend = NaiveAttention()

        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        k = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)

        out = backend.forward(q, k, v, is_causal=True)

        assert out.shape == (num_tokens, num_heads, head_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# =============================================================================
# SDPAAttention 테스트
# =============================================================================


class TestSDPAAttention:
    """SDPAAttention 기능 테스트."""

    def test_matches_naive(self):
        """SDPA와 Naive가 동일 결과를 내는지 교차 검증."""
        num_tokens = 16
        num_heads, num_kv_heads, head_dim = 24, 8, 128
        naive = NaiveAttention()
        sdpa = SDPAAttention()

        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        k = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)

        out_naive = naive.forward(q, k, v, is_causal=True)
        out_sdpa = sdpa.forward(q, k, v, is_causal=True)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(out_sdpa, out_naive, **tol)

    def test_basic_gqa(self):
        """GQA 지원 확인."""
        num_tokens = 16
        num_heads, num_kv_heads, head_dim = 24, 8, 128
        backend = SDPAAttention()

        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        k = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)

        out = backend.forward(q, k, v, is_causal=True)

        assert out.shape == (num_tokens, num_heads, head_dim)

        ref = attention_reference(q, k, v, is_causal=True).to(DEFAULT_DTYPE)
        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(out, ref, **tol)

    def test_single_token_decode(self):
        """Decode 모드에서 Naive와 동일 결과."""
        num_heads, num_kv_heads, head_dim = 8, 4, 64
        seq_len = 32
        naive = NaiveAttention()
        sdpa = SDPAAttention()

        q = torch.randn(1, num_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        k = torch.randn(seq_len, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(seq_len, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)

        out_naive = naive.forward(q, k, v, is_causal=False)
        out_sdpa = sdpa.forward(q, k, v, is_causal=False)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(out_sdpa, out_naive, **tol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtypes(self, dtype):
        """dtype별 정확성 검증."""
        num_tokens, num_heads, head_dim = 16, 8, 64
        backend = SDPAAttention()

        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=dtype, device=DEVICE)
        k = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=dtype, device=DEVICE)
        v = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=dtype, device=DEVICE)

        out = backend.forward(q, k, v, is_causal=True)
        assert out.dtype == dtype


# =============================================================================
# KVCache 테스트
# =============================================================================


class TestKVCache:
    """KVCache 기능 테스트."""

    def test_basic_prefill(self):
        """프롬프트 저장 후 반환값 검증."""
        num_layers, max_seq_len = 2, 64
        num_kv_heads, head_dim = 8, 128
        prompt_len = 16

        cache = KVCache(num_layers, max_seq_len, num_kv_heads, head_dim,
                        dtype=DEFAULT_DTYPE, device=DEVICE)

        k = torch.randn(prompt_len, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(prompt_len, num_kv_heads, head_dim,
                         dtype=DEFAULT_DTYPE, device=DEVICE)

        k_all, v_all = cache.update(0, k, v)

        # 반환된 K/V가 입력과 일치
        assert k_all.shape == (prompt_len, num_kv_heads, head_dim)
        assert v_all.shape == (prompt_len, num_kv_heads, head_dim)
        torch.testing.assert_close(k_all, k, atol=0, rtol=0)
        torch.testing.assert_close(v_all, v, atol=0, rtol=0)

        # advance 전에는 seq_len이 0
        assert cache.current_seq_len == 0

        cache.advance(prompt_len)
        assert cache.current_seq_len == prompt_len

    def test_incremental_decode(self):
        """Prefill 후 1토큰씩 decode."""
        num_layers, max_seq_len = 1, 64
        num_kv_heads, head_dim = 4, 64
        prompt_len = 8
        num_decode_steps = 4

        cache = KVCache(num_layers, max_seq_len, num_kv_heads, head_dim,
                        dtype=DEFAULT_DTYPE, device=DEVICE)

        # Prefill
        k_prompt = torch.randn(prompt_len, num_kv_heads, head_dim,
                                dtype=DEFAULT_DTYPE, device=DEVICE)
        v_prompt = torch.randn(prompt_len, num_kv_heads, head_dim,
                                dtype=DEFAULT_DTYPE, device=DEVICE)
        cache.update(0, k_prompt, v_prompt)
        cache.advance(prompt_len)

        # Decode: 1토큰씩
        for i in range(num_decode_steps):
            k_new = torch.randn(1, num_kv_heads, head_dim,
                                dtype=DEFAULT_DTYPE, device=DEVICE)
            v_new = torch.randn(1, num_kv_heads, head_dim,
                                dtype=DEFAULT_DTYPE, device=DEVICE)

            k_all, v_all = cache.update(0, k_new, v_new)
            expected_len = prompt_len + i + 1

            assert k_all.shape[0] == expected_len
            assert v_all.shape[0] == expected_len

            # 새 토큰이 올바른 위치에 저장됨
            torch.testing.assert_close(
                k_all[prompt_len + i : prompt_len + i + 1], k_new, atol=0, rtol=0
            )

            cache.advance(1)
            assert cache.current_seq_len == expected_len

    def test_multi_layer(self):
        """Layer별 독립 저장 확인."""
        num_layers, max_seq_len = 4, 32
        num_kv_heads, head_dim = 4, 64
        seq_len = 8

        cache = KVCache(num_layers, max_seq_len, num_kv_heads, head_dim,
                        dtype=DEFAULT_DTYPE, device=DEVICE)

        # 각 layer에 서로 다른 데이터 저장
        keys = []
        for layer_idx in range(num_layers):
            k = torch.randn(seq_len, num_kv_heads, head_dim,
                             dtype=DEFAULT_DTYPE, device=DEVICE)
            v = torch.randn(seq_len, num_kv_heads, head_dim,
                             dtype=DEFAULT_DTYPE, device=DEVICE)
            k_all, _ = cache.update(layer_idx, k, v)
            keys.append(k)

        # 각 layer의 데이터가 독립적으로 유지됨
        for layer_idx in range(num_layers):
            k_stored = cache._cache[layer_idx, 0, :seq_len]
            torch.testing.assert_close(k_stored, keys[layer_idx], atol=0, rtol=0)

    def test_reset(self):
        """Reset 후 상태 검증."""
        cache = KVCache(2, 32, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)

        k = torch.randn(8, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(8, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)
        cache.update(0, k, v)
        cache.advance(8)

        assert cache.current_seq_len == 8

        cache.reset()

        assert cache.current_seq_len == 0
        # 버퍼가 0으로 초기화됨
        assert (cache._cache == 0).all()

    def test_overflow_error(self):
        """max_seq_len 초과 시 AssertionError."""
        cache = KVCache(1, 8, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)

        k = torch.randn(8, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(8, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)
        cache.update(0, k, v)
        cache.advance(8)

        # 1토큰 추가하면 overflow
        k_extra = torch.randn(1, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)
        v_extra = torch.randn(1, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)
        with pytest.raises(AssertionError):
            cache.update(0, k_extra, v_extra)

    def test_advance_separate_from_update(self):
        """update() 만으로는 포인터가 이동하지 않음."""
        cache = KVCache(1, 32, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)

        k = torch.randn(8, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)
        v = torch.randn(8, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)

        cache.update(0, k, v)

        # advance() 전에는 seq_len이 0
        assert cache.current_seq_len == 0

        # 같은 위치에 다시 쓰기 가능 (포인터 안 움직였으므로)
        k2 = torch.randn(8, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)
        v2 = torch.randn(8, 4, 64, dtype=DEFAULT_DTYPE, device=DEVICE)
        k_all, _ = cache.update(0, k2, v2)

        # 덮어쓴 데이터가 반환됨
        torch.testing.assert_close(k_all, k2, atol=0, rtol=0)


# =============================================================================
# 통합 테스트: Attention + KV Cache
# =============================================================================


class TestAttentionWithKVCache:
    """Attention backend + KV Cache 통합 테스트."""

    def test_prefill_then_decode(self):
        """Prefill → Decode 전체 흐름.

        KV cache를 사용한 incremental decode가
        from-scratch 재계산과 동일한 결과를 내는지 검증.
        """
        num_layers = 1
        num_heads, num_kv_heads, head_dim = 8, 4, 64
        prompt_len = 16
        num_decode_steps = 4

        cache = KVCache(num_layers, max_seq_len=64,
                        num_kv_heads=num_kv_heads, head_dim=head_dim,
                        dtype=DEFAULT_DTYPE, device=DEVICE)
        backend = NaiveAttention()

        # 전체 시퀀스의 Q/K/V 미리 생성
        total_len = prompt_len + num_decode_steps
        all_q = torch.randn(total_len, num_heads, head_dim,
                            dtype=DEFAULT_DTYPE, device=DEVICE)
        all_k = torch.randn(total_len, num_kv_heads, head_dim,
                            dtype=DEFAULT_DTYPE, device=DEVICE)
        all_v = torch.randn(total_len, num_kv_heads, head_dim,
                            dtype=DEFAULT_DTYPE, device=DEVICE)

        # --- Prefill ---
        q_prefill = all_q[:prompt_len]
        k_prefill = all_k[:prompt_len]
        v_prefill = all_v[:prompt_len]

        k_cached, v_cached = cache.update(0, k_prefill, v_prefill)
        out_prefill = backend.forward(q_prefill, k_cached, v_cached, is_causal=True)
        cache.advance(prompt_len)

        # 참조: cache 없이 동일 계산
        out_ref = backend.forward(q_prefill, k_prefill, v_prefill, is_causal=True)
        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(out_prefill, out_ref, **tol)

        # --- Decode ---
        for i in range(num_decode_steps):
            pos = prompt_len + i
            q_new = all_q[pos : pos + 1]
            k_new = all_k[pos : pos + 1]
            v_new = all_v[pos : pos + 1]

            k_cached, v_cached = cache.update(0, k_new, v_new)
            out_decode = backend.forward(q_new, k_cached, v_cached, is_causal=False)
            cache.advance(1)

            # 참조: 처음부터 전체 시퀀스로 재계산
            out_ref = backend.forward(
                q_new,
                all_k[: pos + 1],
                all_v[: pos + 1],
                is_causal=False,
            )
            torch.testing.assert_close(out_decode, out_ref, **tol)

    def test_naive_vs_sdpa_with_cache(self):
        """두 backend + KV cache로 동일 결과 확인."""
        num_heads, num_kv_heads, head_dim = 8, 4, 64
        prompt_len = 16
        decode_steps = 2

        naive = NaiveAttention()
        sdpa = SDPAAttention()
        tol = TOLERANCES[DEFAULT_DTYPE]

        # 동일한 K/V를 두 캐시에 저장
        cache_naive = KVCache(1, 64, num_kv_heads, head_dim,
                              dtype=DEFAULT_DTYPE, device=DEVICE)
        cache_sdpa = KVCache(1, 64, num_kv_heads, head_dim,
                             dtype=DEFAULT_DTYPE, device=DEVICE)

        total_len = prompt_len + decode_steps
        all_q = torch.randn(total_len, num_heads, head_dim,
                            dtype=DEFAULT_DTYPE, device=DEVICE)
        all_k = torch.randn(total_len, num_kv_heads, head_dim,
                            dtype=DEFAULT_DTYPE, device=DEVICE)
        all_v = torch.randn(total_len, num_kv_heads, head_dim,
                            dtype=DEFAULT_DTYPE, device=DEVICE)

        # Prefill
        k_n, v_n = cache_naive.update(0, all_k[:prompt_len], all_v[:prompt_len])
        k_s, v_s = cache_sdpa.update(0, all_k[:prompt_len], all_v[:prompt_len])

        out_naive = naive.forward(all_q[:prompt_len], k_n, v_n, is_causal=True)
        out_sdpa = sdpa.forward(all_q[:prompt_len], k_s, v_s, is_causal=True)
        torch.testing.assert_close(out_naive, out_sdpa, **tol)

        cache_naive.advance(prompt_len)
        cache_sdpa.advance(prompt_len)

        # Decode
        for i in range(decode_steps):
            pos = prompt_len + i
            q = all_q[pos : pos + 1]
            k = all_k[pos : pos + 1]
            v = all_v[pos : pos + 1]

            k_n, v_n = cache_naive.update(0, k, v)
            k_s, v_s = cache_sdpa.update(0, k, v)

            out_naive = naive.forward(q, k_n, v_n, is_causal=False)
            out_sdpa = sdpa.forward(q, k_s, v_s, is_causal=False)
            torch.testing.assert_close(out_naive, out_sdpa, **tol)

            cache_naive.advance(1)
            cache_sdpa.advance(1)


# =============================================================================
# 성능 벤치마크: Naive vs SDPA
# =============================================================================


class TestAttentionBenchmark:
    """NaiveAttention vs SDPAAttention 성능 비교.

    Prefill (긴 시퀀스, compute-bound)과 Decode (1토큰, memory-bound)를
    각각 측정하여 두 backend의 특성 차이를 확인합니다.

    SDPA는 내부적으로 FlashAttention/Memory-Efficient 커널을 자동 선택하므로,
    Naive(수동 matmul+softmax)보다 빠를 것으로 예상됩니다.
    """

    def test_prefill_performance(self):
        """Prefill 성능: 긴 시퀀스에서 Naive vs SDPA.

        Llama-3.2-3B 스펙으로 측정합니다.
        Prefill은 compute-bound이므로 FlashAttention의 tiling이 효과적입니다.
        """
        num_tokens = 512
        num_heads, num_kv_heads, head_dim = 24, 8, 128
        dtype = DEFAULT_DTYPE
        warmup = 10
        repeat = 100

        naive = NaiveAttention()
        sdpa = SDPAAttention()

        q = torch.randn(num_tokens, num_heads, head_dim,
                         dtype=dtype, device=DEVICE)
        k = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=dtype, device=DEVICE)
        v = torch.randn(num_tokens, num_kv_heads, head_dim,
                         dtype=dtype, device=DEVICE)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # --- Naive ---
        for _ in range(warmup):
            naive.forward(q, k, v, is_causal=True)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            naive.forward(q, k, v, is_causal=True)
        end.record()
        torch.cuda.synchronize()
        naive_us = start.elapsed_time(end) / repeat * 1000

        # --- SDPA ---
        for _ in range(warmup):
            sdpa.forward(q, k, v, is_causal=True)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            sdpa.forward(q, k, v, is_causal=True)
        end.record()
        torch.cuda.synchronize()
        sdpa_us = start.elapsed_time(end) / repeat * 1000

        print(f"\n  [Prefill] num_tokens={num_tokens}, "
              f"heads={num_heads}, kv_heads={num_kv_heads}, "
              f"head_dim={head_dim}, dtype={dtype}")
        print(f"  [1] Naive (matmul + softmax + matmul):  {naive_us:.1f} us")
        print(f"  [2] SDPA (FlashAttention/MemEfficient): {sdpa_us:.1f} us")
        print(f"  Naive vs SDPA: {naive_us / sdpa_us:.2f}x")

    def test_decode_performance(self):
        """Decode 성능: 1토큰 query가 긴 KV cache에 attend.

        Decode는 Q=[1, heads, dim]이므로 GEMV에 가깝고 memory-bound입니다.
        KV cache 전체를 읽어야 하므로 seq_len이 길수록 메모리 대역폭이 병목입니다.
        """
        seq_len = 2048
        num_heads, num_kv_heads, head_dim = 24, 8, 128
        dtype = DEFAULT_DTYPE
        warmup = 10
        repeat = 100

        naive = NaiveAttention()
        sdpa = SDPAAttention()

        q = torch.randn(1, num_heads, head_dim,
                         dtype=dtype, device=DEVICE)
        k = torch.randn(seq_len, num_kv_heads, head_dim,
                         dtype=dtype, device=DEVICE)
        v = torch.randn(seq_len, num_kv_heads, head_dim,
                         dtype=dtype, device=DEVICE)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # --- Naive ---
        for _ in range(warmup):
            naive.forward(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            naive.forward(q, k, v, is_causal=False)
        end.record()
        torch.cuda.synchronize()
        naive_us = start.elapsed_time(end) / repeat * 1000

        # --- SDPA ---
        for _ in range(warmup):
            sdpa.forward(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            sdpa.forward(q, k, v, is_causal=False)
        end.record()
        torch.cuda.synchronize()
        sdpa_us = start.elapsed_time(end) / repeat * 1000

        print(f"\n  [Decode] seq_len={seq_len}, "
              f"heads={num_heads}, kv_heads={num_kv_heads}, "
              f"head_dim={head_dim}, dtype={dtype}")
        print(f"  [1] Naive (matmul + softmax + matmul):  {naive_us:.1f} us")
        print(f"  [2] SDPA (FlashAttention/MemEfficient): {sdpa_us:.1f} us")
        print(f"  Naive vs SDPA: {naive_us / sdpa_us:.2f}x")
