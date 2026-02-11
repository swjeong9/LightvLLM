"""
RMSNorm (Root Mean Square Layer Normalization) Python 테스트

CUDA 커널의 결과를 PyTorch 순수 구현과 비교하여 정확성을 검증합니다.
참조 구현도 GPU에서 동일한 dtype으로 실행합니다.

빌드:
    uv pip install -e .

실행 (전체):
    uv run pytest tests/kernels/test_layernorm.py -v

실행 (특정 테스트만):
    uv run pytest tests/kernels/test_layernorm.py::TestRMSNorm::test_rms_norm_basic -v

실행 (키워드 매칭):
    uv run pytest tests/kernels/test_layernorm.py -k "fused" -v
"""

import pytest
import torch
import lightvllm._C as _C

# 기본 dtype: 요즘 추론은 bf16이 표준
DEFAULT_DTYPE = torch.bfloat16


# =============================================================================
# PyTorch 참조 구현
# =============================================================================

def rms_norm_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    RMSNorm의 PyTorch 참조 구현

    output = input / sqrt(mean(input²) + epsilon) * weight

    fp32로 변환하여 중간 계산을 수행한 후, 원래 dtype으로 변환합니다.
    이것은 CUDA 커널의 동작과 동일합니다 (커널도 fp32로 누적).
    """
    input_float = input.float()
    variance = input_float.pow(2).mean(dim=-1, keepdim=True)
    input_normalized = input_float * torch.rsqrt(variance + epsilon)
    return (input_normalized * weight.float()).to(input.dtype)


def fused_add_rms_norm_reference(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Add+RMSNorm의 PyTorch 참조 구현

    Step 1: residual = residual + input
    Step 2: output = rms_norm(residual) * weight
    """
    residual_updated = residual + input
    residual_float = residual_updated.float()
    variance = residual_float.pow(2).mean(dim=-1, keepdim=True)
    output = residual_float * torch.rsqrt(variance + epsilon)
    output = (output * weight.float()).to(input.dtype)
    return output, residual_updated


# =============================================================================
# 허용 오차 설정
# =============================================================================
# RMSNorm은 병렬 리덕션(CUB BlockReduce)을 사용하므로,
# PyTorch의 순차적 리덕션과 부동소수점 누적 순서가 다릅니다.
# 따라서 bit-exact 비교(atol=0, rtol=0)는 불가능하며,
# dtype에 따른 적절한 허용 오차를 사용합니다.

TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}


# =============================================================================
# 테스트 케이스
# =============================================================================

class TestRMSNorm:
    """CUDA RMSNorm 커널을 PyTorch 참조 구현과 비교하는 테스트"""

    def test_rms_norm_basic(self):
        """
        Test 1: 기본 정확성 (bf16)

        CUDA 커널과 PyTorch 참조 구현의 결과가 일치하는지 확인
        """
        num_tokens = 32
        hidden_size = 256
        epsilon = 1e-6

        input = torch.randn(num_tokens, hidden_size,
                            dtype=DEFAULT_DTYPE, device="cuda")
        weight = torch.randn(hidden_size, dtype=DEFAULT_DTYPE, device="cuda")

        # 참조 구현
        output_ref = rms_norm_reference(input, weight, epsilon)

        # CUDA 커널
        output = torch.empty_like(input)
        _C.rms_norm(output, input, weight, epsilon)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_fused_add_rms_norm_basic(self):
        """
        Test 2: Fused Add+RMSNorm 기본 정확성

        정규화 결과와 업데이트된 residual 모두 참조 구현과 일치해야 함
        """
        num_tokens = 32
        hidden_size = 256
        epsilon = 1e-6

        input = torch.randn(num_tokens, hidden_size,
                            dtype=DEFAULT_DTYPE, device="cuda")
        residual = torch.randn(num_tokens, hidden_size,
                               dtype=DEFAULT_DTYPE, device="cuda")
        weight = torch.randn(hidden_size, dtype=DEFAULT_DTYPE, device="cuda")

        # 참조 구현
        output_ref, residual_ref = fused_add_rms_norm_reference(
            input.clone(), residual.clone(), weight, epsilon)

        # CUDA 커널 (in-place)
        _C.fused_add_rms_norm(input, residual, weight, epsilon)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(input, output_ref, **tol)
        torch.testing.assert_close(residual, residual_ref, **tol)

    def test_fused_residual_update(self):
        """
        Test 3: Fused 커널의 residual 업데이트 정확성

        fused 커널 실행 후 residual == (원래 input + 원래 residual)인지 확인
        """
        num_tokens = 16
        hidden_size = 128
        epsilon = 1e-6

        input_orig = torch.randn(num_tokens, hidden_size,
                                 dtype=DEFAULT_DTYPE, device="cuda")
        residual_orig = torch.randn(num_tokens, hidden_size,
                                    dtype=DEFAULT_DTYPE, device="cuda")
        weight = torch.ones(hidden_size, dtype=DEFAULT_DTYPE, device="cuda")

        # 기대값: 단순 덧셈
        expected_residual = input_orig + residual_orig

        # CUDA 커널 실행
        input_buf = input_orig.clone()
        residual_buf = residual_orig.clone()
        _C.fused_add_rms_norm(input_buf, residual_buf, weight, epsilon)

        # residual은 정확히 덧셈 결과여야 함 (리덕션 무관)
        torch.testing.assert_close(residual_buf, expected_residual, atol=0, rtol=0)

    def test_unit_weight_property(self):
        """
        Test 4: 단위 가중치 수학적 성질

        weight = 1일 때, RMSNorm의 출력은 다음 성질을 만족:
        mean(output²) ≈ 1 / (1 + epsilon/mean(input²))

        직관적으로: 입력을 RMS로 나누면 결과의 RMS ≈ 1
        """
        num_tokens = 64
        hidden_size = 512
        epsilon = 1e-6

        input = torch.randn(num_tokens, hidden_size,
                            dtype=torch.float32, device="cuda")
        weight = torch.ones(hidden_size, dtype=torch.float32, device="cuda")

        output = torch.empty_like(input)
        _C.rms_norm(output, input, weight, epsilon)

        # 각 행의 mean(output²)이 ≈ 1이어야 함
        output_mean_sq = output.float().pow(2).mean(dim=-1)
        expected = torch.ones(num_tokens, device="cuda")
        torch.testing.assert_close(output_mean_sq, expected, atol=1e-4, rtol=1e-4)

    def test_zero_input(self):
        """
        Test 5: 영벡터 입력 안정성

        입력이 모두 0일 때 NaN이나 Inf가 발생하지 않아야 함
        (epsilon이 0으로 나누는 것을 방지)
        """
        num_tokens = 8
        hidden_size = 128
        epsilon = 1e-6

        input = torch.zeros(num_tokens, hidden_size,
                            dtype=DEFAULT_DTYPE, device="cuda")
        weight = torch.ones(hidden_size, dtype=DEFAULT_DTYPE, device="cuda")

        output = torch.empty_like(input)
        _C.rms_norm(output, input, weight, epsilon)

        # 출력이 0이어야 하고, NaN/Inf가 없어야 함
        assert not output.isnan().any(), "NaN detected in output"
        assert not output.isinf().any(), "Inf detected in output"
        torch.testing.assert_close(
            output,
            torch.zeros_like(output),
            atol=0, rtol=0,
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_dtypes(self, dtype):
        """
        Test 6: half dtype별 동작 검증

        float16, bfloat16 모두 정상 동작해야 함
        """
        num_tokens = 32
        hidden_size = 256
        epsilon = 1e-6

        input = torch.randn(num_tokens, hidden_size,
                            dtype=dtype, device="cuda")
        weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

        output_ref = rms_norm_reference(input, weight, epsilon)

        output = torch.empty_like(input)
        _C.rms_norm(output, input, weight, epsilon)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_large_scale(self):
        """
        Test 7: 대규모 텐서

        LLaMA-like 설정 (hidden_size=4096, num_tokens=1024, bf16)
        """
        num_tokens = 1024
        hidden_size = 4096
        epsilon = 1e-5

        input = torch.randn(num_tokens, hidden_size,
                            dtype=DEFAULT_DTYPE, device="cuda")
        weight = torch.randn(hidden_size, dtype=DEFAULT_DTYPE, device="cuda")

        output_ref = rms_norm_reference(input, weight, epsilon)

        output = torch.empty_like(input)
        _C.rms_norm(output, input, weight, epsilon)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)

    @pytest.mark.parametrize("hidden_size", [128, 256, 1024, 2048, 4096, 8192])
    def test_various_hidden_sizes(self, hidden_size):
        """
        Test 8: 다양한 hidden_size에서의 정확성

        LLM에서 사용되는 일반적인 hidden_size 값들에 대해 검증
        """
        num_tokens = 32
        epsilon = 1e-6

        input = torch.randn(num_tokens, hidden_size,
                            dtype=DEFAULT_DTYPE, device="cuda")
        weight = torch.randn(hidden_size, dtype=DEFAULT_DTYPE, device="cuda")

        output_ref = rms_norm_reference(input, weight, epsilon)

        output = torch.empty_like(input)
        _C.rms_norm(output, input, weight, epsilon)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)
