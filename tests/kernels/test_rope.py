"""
RoPE (Rotary Position Embedding) Python 테스트

CUDA 커널의 결과를 PyTorch 순수 구현과 비교하여 정확성을 검증합니다.
참조 구현도 GPU에서 동일한 dtype으로 실행하여 연산 순서까지 일치시킵니다.

빌드:
    uv pip install -e .

실행 (전체):
    uv run pytest tests/kernels/test_rope.py -v

실행 (특정 테스트만):
    uv run pytest tests/kernels/test_rope.py::TestRoPE::test_basic_correctness -v

실행 (키워드 매칭):
    uv run pytest tests/kernels/test_rope.py -k "norm" -v
"""

import pytest
import torch
import lightvllm._C as _C

# 기본 dtype: 요즘 추론은 bf16이 표준
DEFAULT_DTYPE = torch.bfloat16


# =============================================================================
# PyTorch 참조 구현
# =============================================================================

def rope_reference(
    query: torch.Tensor,          # [num_tokens, num_heads * head_size]
    key: torch.Tensor,            # [num_tokens, num_kv_heads * head_size]
    positions: torch.Tensor,      # [num_tokens] (GPU)
    cos_sin_cache: torch.Tensor,  # [max_position, rot_dim] (GPU)
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE의 PyTorch 참조 구현 (GPT-NeoX 스타일)

    커널과 동일한 로직을 동일한 디바이스/dtype에서 실행합니다.
    CPU bf16은 내부적으로 float32 변환 후 계산하므로 GPU와 결과가 달라질 수 있어,
    반드시 GPU에서 실행해야 합니다.
    """
    rot_dim = cos_sin_cache.shape[1]
    embed_dim = rot_dim // 2
    num_tokens = query.shape[0]

    query_out = query.clone()
    key_out = key.clone()

    for t in range(num_tokens):
        pos = positions[t].item()
        cos = cos_sin_cache[pos, :embed_dim]
        sin = cos_sin_cache[pos, embed_dim:]

        for h in range(num_heads):
            offset = h * head_size
            x = query_out[t, offset:offset + embed_dim].clone()
            y = query_out[t, offset + embed_dim:offset + rot_dim].clone()
            query_out[t, offset:offset + embed_dim] = x * cos - y * sin
            query_out[t, offset + embed_dim:offset + rot_dim] = x * sin + y * cos

        for h in range(num_kv_heads):
            offset = h * head_size
            x = key_out[t, offset:offset + embed_dim].clone()
            y = key_out[t, offset + embed_dim:offset + rot_dim].clone()
            key_out[t, offset:offset + embed_dim] = x * cos - y * sin
            key_out[t, offset + embed_dim:offset + rot_dim] = x * sin + y * cos

    return query_out, key_out


def generate_cos_sin_cache(
    max_position: int,
    rot_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = "cuda",
) -> torch.Tensor:
    """
    cos/sin 캐시 생성

    θ_i = position × base^(-2i/rot_dim)
    캐시 레이아웃: [max_position, rot_dim]
      각 행: [cos(θ_0), ..., cos(θ_{d/2-1}), sin(θ_0), ..., sin(θ_{d/2-1})]
    """
    embed_dim = rot_dim // 2
    positions = torch.arange(max_position, dtype=torch.float32)
    dims = torch.arange(embed_dim, dtype=torch.float32)
    freqs = positions.unsqueeze(1) * (base ** (-2.0 * dims / rot_dim)).unsqueeze(0)

    cache = torch.empty(max_position, rot_dim, dtype=dtype, device=device)
    cache[:, :embed_dim] = freqs.cos().to(dtype).to(device)
    cache[:, embed_dim:] = freqs.sin().to(dtype).to(device)
    return cache


# =============================================================================
# 테스트 케이스
# =============================================================================

class TestRoPE:
    """CUDA RoPE 커널을 PyTorch 참조 구현과 비교하는 테스트"""

    def test_basic_correctness(self):
        """
        Test 1: 기본 정확성 (bf16)

        CUDA 커널과 PyTorch 참조 구현의 결과가 일치하는지 확인
        """
        num_tokens = 32
        num_heads = 8
        num_kv_heads = 2
        head_size = 64
        rot_dim = head_size
        max_position = 128

        query = torch.randn(num_tokens, num_heads * head_size,
                            dtype=DEFAULT_DTYPE, device="cuda")
        key = torch.randn(num_tokens, num_kv_heads * head_size,
                          dtype=DEFAULT_DTYPE, device="cuda")
        positions = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        cos_sin_cache = generate_cos_sin_cache(max_position, rot_dim)

        # 참조 구현 (GPU에서 같은 dtype)
        query_ref, key_ref = rope_reference(
            query.clone(), key.clone(), positions, cos_sin_cache,
            head_size, num_heads, num_kv_heads,
        )

        # CUDA 커널 (in-place)
        _C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, True)

        torch.testing.assert_close(query, query_ref, atol=0, rtol=0)
        torch.testing.assert_close(key, key_ref, atol=0, rtol=0)

    def test_norm_preservation(self):
        """
        Test 2: Norm 보존

        회전 행렬의 직교성: ||R(θ)x|| = ||x||
        RoPE 적용 전후 각 head 벡터의 L2 norm이 보존되어야 함
        """
        num_tokens = 16
        num_heads = 4
        num_kv_heads = 4
        head_size = 128
        rot_dim = head_size
        max_position = 64

        query = torch.randn(num_tokens, num_heads * head_size,
                            dtype=DEFAULT_DTYPE, device="cuda")
        positions = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        cos_sin_cache = generate_cos_sin_cache(max_position, rot_dim)

        # 적용 전 각 head의 norm 저장 (float32로 계산하여 정밀도 확보)
        norms_before = []
        for t in range(num_tokens):
            for h in range(num_heads):
                offset = h * head_size
                vec = query[t, offset:offset + head_size].float()
                norms_before.append(vec.norm().item())

        key = torch.randn(num_tokens, num_kv_heads * head_size,
                          dtype=DEFAULT_DTYPE, device="cuda")

        _C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, True)

        # 적용 후 norm 비교 (bf16 양자화 오차 허용)
        idx = 0
        for t in range(num_tokens):
            for h in range(num_heads):
                offset = h * head_size
                vec = query[t, offset:offset + head_size].float()
                norm_after = vec.norm().item()
                assert abs(norms_before[idx] - norm_after) < 1e-1, \
                    f"token {t}, head {h}: before={norms_before[idx]}, after={norm_after}"
                idx += 1

    def test_position_zero_identity(self):
        """
        Test 3: Position 0 항등 변환

        position=0 → θ=0 → cos(0)=1, sin(0)=0 → 회전 없음
        입력과 출력이 동일해야 함
        """
        num_tokens = 8
        num_heads = 4
        num_kv_heads = 2
        head_size = 64
        rot_dim = head_size
        max_position = 16

        query = torch.randn(num_tokens, num_heads * head_size,
                            dtype=DEFAULT_DTYPE, device="cuda")
        key = torch.randn(num_tokens, num_kv_heads * head_size,
                          dtype=DEFAULT_DTYPE, device="cuda")
        positions = torch.zeros(num_tokens, dtype=torch.int64, device="cuda")
        cos_sin_cache = generate_cos_sin_cache(max_position, rot_dim)

        query_before = query.clone()
        key_before = key.clone()

        _C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, True)

        torch.testing.assert_close(query, query_before, atol=0, rtol=0)
        torch.testing.assert_close(key, key_before, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_dtypes(self, dtype):
        """
        Test 4: half dtype별 동작 검증

        float16, bfloat16 모두 정상 동작해야 함
        """
        num_tokens = 16
        num_heads = 4
        num_kv_heads = 2
        head_size = 64
        rot_dim = head_size
        max_position = 64

        query = torch.randn(num_tokens, num_heads * head_size,
                            dtype=dtype, device="cuda")
        key = torch.randn(num_tokens, num_kv_heads * head_size,
                          dtype=dtype, device="cuda")
        positions = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        cos_sin_cache = generate_cos_sin_cache(max_position, rot_dim, dtype=dtype)

        query_ref, key_ref = rope_reference(
            query.clone(), key.clone(), positions, cos_sin_cache,
            head_size, num_heads, num_kv_heads,
        )

        _C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, True)

        torch.testing.assert_close(query, query_ref, atol=0, rtol=0)
        torch.testing.assert_close(key, key_ref, atol=0, rtol=0)

    def test_large_scale(self):
        """
        Test 5: 대규모 텐서

        LLaMA-like 설정 (num_heads=32, num_kv_heads=8, head_size=128, bf16)
        """
        num_tokens = 1024
        num_heads = 32
        num_kv_heads = 8
        head_size = 128
        rot_dim = head_size
        max_position = 4096

        query = torch.randn(num_tokens, num_heads * head_size,
                            dtype=DEFAULT_DTYPE, device="cuda")
        key = torch.randn(num_tokens, num_kv_heads * head_size,
                          dtype=DEFAULT_DTYPE, device="cuda")
        positions = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        cos_sin_cache = generate_cos_sin_cache(max_position, rot_dim)

        query_ref, key_ref = rope_reference(
            query.clone(), key.clone(), positions, cos_sin_cache,
            head_size, num_heads, num_kv_heads,
        )

        _C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, True)

        torch.testing.assert_close(query, query_ref, atol=0, rtol=0)
        torch.testing.assert_close(key, key_ref, atol=0, rtol=0)
