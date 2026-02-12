"""
Activation 커널 Python 테스트

CUDA 커널의 결과를 PyTorch 순수 구현과 비교하여 정확성을 검증합니다.
참조 구현도 GPU에서 동일한 dtype으로 실행합니다.

빌드:
    uv pip install -e .

실행 (전체):
    uv run pytest tests/kernels/test_activation.py -v

실행 (특정 테스트만):
    uv run pytest tests/kernels/test_activation.py::TestSiLU::test_silu_basic -v

실행 (키워드 매칭):
    uv run pytest tests/kernels/test_activation.py -k "silu_and_mul" -v
"""

import pytest
import torch
import torch.nn.functional as F
import lightvllm._C as _C
from lightvllm.kernels.activation import silu, silu_and_mul
from lightvllm.layers.activation import SiluAndMul

# 기본 dtype: 요즘 추론은 bf16이 표준
DEFAULT_DTYPE = torch.bfloat16


# =============================================================================
# PyTorch 참조 구현
# =============================================================================

def silu_reference(input: torch.Tensor) -> torch.Tensor:
    """SiLU의 PyTorch 참조 구현: F.silu() 사용

    GPU에서 동일 dtype으로 실행합니다.
    """
    return F.silu(input)


def silu_and_mul_reference(input: torch.Tensor) -> torch.Tensor:
    """silu_and_mul의 PyTorch 참조 구현

    input[..., :d]에 SiLU를 적용하고 input[..., d:]와 곱합니다.
    """
    d = input.shape[-1] // 2
    return F.silu(input[..., :d]) * input[..., d:]


# =============================================================================
# 허용 오차 설정
# =============================================================================
# Activation 커널은 element-wise 연산 (리덕션 없음)이므로
# RMSNorm보다 오차가 작습니다. 주된 오차 원인은 fp32 중간 계산과
# PyTorch 네이티브 연산 간의 미세한 expf() 구현 차이입니다.

TOLERANCES = {
    torch.float32: {"atol": 1e-6, "rtol": 1e-5},
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}


# =============================================================================
# SiLU 테스트
# =============================================================================

class TestSiLU:
    """CUDA SiLU 커널을 PyTorch F.silu()와 비교하는 테스트"""

    def test_silu_basic(self):
        """
        Test 1: 기본 정확성 (bf16)

        CUDA 커널과 PyTorch F.silu()의 결과가 일치하는지 확인
        """
        num_tokens = 32
        d = 256

        input = torch.randn(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")

        # 참조 구현
        output_ref = silu_reference(input)

        # CUDA 커널
        output = torch.empty_like(input)
        _C.silu(output, input)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_silu_zero_input(self):
        """
        Test 2: silu(0) == 0 (정확히)

        SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        """
        input = torch.zeros(8, 64, dtype=DEFAULT_DTYPE, device="cuda")

        output = torch.empty_like(input)
        _C.silu(output, input)

        torch.testing.assert_close(
            output,
            torch.zeros_like(output),
            atol=0, rtol=0,
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_silu_half_dtypes(self, dtype):
        """
        Test 3: half dtype별 동작 검증

        float16, bfloat16 모두 정상 동작해야 함
        """
        num_tokens = 32
        d = 256

        input = torch.randn(num_tokens, d, dtype=dtype, device="cuda")

        output_ref = silu_reference(input)

        output = torch.empty_like(input)
        _C.silu(output, input)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_silu_large_values(self):
        """
        Test 4: 큰 양수/음수에서의 동작

        - 큰 양수: silu(x) ≈ x (sigmoid → 1)
        - 큰 음수: silu(x) ≈ 0 (sigmoid → 0)
        """
        input = torch.tensor(
            [100.0, -100.0, 50.0, -50.0],
            dtype=torch.float32, device="cuda",
        )

        output = torch.empty_like(input)
        _C.silu(output, input)

        # 큰 양수: silu(100) ≈ 100
        assert abs(output[0].item() - 100.0) < 1e-3
        # 큰 음수: silu(-100) ≈ 0
        assert abs(output[1].item()) < 1e-6
        # 양수: silu(50) ≈ 50
        assert abs(output[2].item() - 50.0) < 1e-3
        # 음수: silu(-50) ≈ 0
        assert abs(output[3].item()) < 1e-6

    @pytest.mark.parametrize("d", [128, 256, 1024, 4096, 11008])
    def test_silu_various_sizes(self, d):
        """
        Test 5: 다양한 차원 크기

        LLaMA에서 사용되는 일반적인 크기들에 대해 검증
        (11008은 LLaMA-7B의 intermediate_size)
        """
        num_tokens = 32

        input = torch.randn(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")

        output_ref = silu_reference(input)

        output = torch.empty_like(input)
        _C.silu(output, input)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_silu_vs_pytorch_module(self):
        """
        Test 6: torch.nn.SiLU() Module과 비교

        PyTorch의 공식 SiLU Module과 동일한 결과를 생성하는지 확인
        """
        num_tokens = 64
        d = 512

        input = torch.randn(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")

        # PyTorch Module
        pytorch_silu = torch.nn.SiLU()
        output_pytorch = pytorch_silu(input)

        # CUDA 커널
        output = torch.empty_like(input)
        _C.silu(output, input)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_pytorch, **tol)

    def test_performance_vs_pytorch(self):
        """
        Test 7: SiLU 단독 성능 벤치마크

        3가지 비교:
        1. PyTorch F.silu() — 새 텐서 할당 + SiLU 커널
        2. PyTorch F.silu(inplace=True) — 할당 없이 in-place SiLU
        3. CUDA _C.silu() — 미리 할당된 출력 텐서에 기록

        SiLU는 element-wise 연산이므로 순수 메모리 대역폭 바운드입니다.
        PyTorch의 F.silu()도 내부적으로 CUDA 커널을 호출하기 때문에,
        연산 자체의 성능 차이는 거의 없습니다.
        차이가 있다면 텐서 할당(allocation) 오버헤드 때문입니다.
        """
        num_tokens = 4096
        d = 11008
        warmup = 10
        repeat = 100

        input = torch.randn(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        output = torch.empty_like(input)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # ----- [1] PyTorch F.silu() — 새 텐서 할당 -----
        for _ in range(warmup):
            output = F.silu(input)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            output = F.silu(input)
        end.record()
        torch.cuda.synchronize()
        pytorch_us = start.elapsed_time(end) / repeat * 1000

        # ----- [2] PyTorch F.silu(inplace=True) — in-place -----
        # 실제 추론에서는 입력을 다시 쓸 일이 없으므로 clone 없이 측정.
        # inplace는 출력 텐서 할당을 피하는 것이 목적.
        input_inplace = input.clone()
        for _ in range(warmup):
            F.silu(input_inplace, inplace=True)
        torch.cuda.synchronize()

        input_inplace.copy_(input)
        start.record()
        for _ in range(repeat):
            F.silu(input_inplace, inplace=True)
        end.record()
        torch.cuda.synchronize()
        pytorch_inplace_us = start.elapsed_time(end) / repeat * 1000

        # ----- [3] CUDA _C.silu() — 미리 할당된 출력 텐서 -----
        for _ in range(warmup):
            _C.silu(output, input)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            _C.silu(output, input)
        end.record()
        torch.cuda.synchronize()
        cuda_us = start.elapsed_time(end) / repeat * 1000

        print(f"\n  설정: num_tokens={num_tokens}, d={d}, dtype={DEFAULT_DTYPE}")
        print(f"  [1] PyTorch F.silu():              {pytorch_us:.1f} us")
        print(f"  [2] PyTorch F.silu(inplace=True):   {pytorch_inplace_us:.1f} us")
        print(f"  [3] CUDA _C.silu():                {cuda_us:.1f} us")
        print(f"  F.silu vs inplace 속도 비율:       {pytorch_us / pytorch_inplace_us:.2f}x")
        print(f"  F.silu vs CUDA 속도 비율:          {pytorch_us / cuda_us:.2f}x")
        print(f"  inplace vs CUDA 속도 비율:         {pytorch_inplace_us / cuda_us:.2f}x")


# =============================================================================
# SiLU + Mul 테스트
# =============================================================================

class TestSiLUAndMul:
    """CUDA silu_and_mul 커널을 PyTorch 참조 구현과 비교하는 테스트"""

    def test_silu_and_mul_basic(self):
        """
        Test 1: 기본 정확성 (bf16)

        silu(input[..., :d]) * input[..., d:]
        """
        num_tokens = 32
        d = 256

        input = torch.randn(num_tokens, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")

        # 참조 구현
        output_ref = silu_and_mul_reference(input)

        # CUDA 커널
        output = torch.empty(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        _C.silu_and_mul(output, input)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_silu_and_mul_output_shape(self):
        """
        Test 2: 출력 shape 검증

        입력 [..., 2*d] → 출력 [..., d]
        """
        num_tokens = 16
        d = 128

        input = torch.randn(num_tokens, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")

        output = torch.empty(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        _C.silu_and_mul(output, input)

        assert output.shape == (num_tokens, d)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_silu_and_mul_half_dtypes(self, dtype):
        """
        Test 3: half dtype별 동작 검증
        """
        num_tokens = 32
        d = 256

        input = torch.randn(num_tokens, 2 * d, dtype=dtype, device="cuda")

        output_ref = silu_and_mul_reference(input)

        output = torch.empty(num_tokens, d, dtype=dtype, device="cuda")
        _C.silu_and_mul(output, input)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_silu_and_mul_large_scale(self):
        """
        Test 4: LLaMA-like 대규모 텐서

        LLaMA-7B: intermediate_size=11008, num_tokens=1024
        """
        num_tokens = 1024
        d = 11008

        input = torch.randn(num_tokens, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")

        output_ref = silu_and_mul_reference(input)

        output = torch.empty(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        _C.silu_and_mul(output, input)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)

    @pytest.mark.parametrize("d", [128, 256, 1024, 4096, 11008])
    def test_silu_and_mul_various_sizes(self, d):
        """
        Test 5: 다양한 intermediate size
        """
        num_tokens = 32

        input = torch.randn(num_tokens, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")

        output_ref = silu_and_mul_reference(input)

        output = torch.empty(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        _C.silu_and_mul(output, input)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_silu_and_mul_3d_input(self):
        """
        Test 6: 3D 배치 입력 [batch, seq_len, 2*d]

        PyTorch 래퍼가 임의 차원의 텐서를 올바르게 처리하는지 확인
        """
        batch = 4
        seq_len = 32
        d = 256

        input = torch.randn(batch, seq_len, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")

        output_ref = silu_and_mul_reference(input)

        output = torch.empty(batch, seq_len, d, dtype=DEFAULT_DTYPE, device="cuda")
        _C.silu_and_mul(output, input)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_silu_and_mul_zero_gate(self):
        """
        Test 7: gate=0일 때 출력이 0이어야 함

        silu(0) * up = 0 * up = 0
        """
        num_tokens = 16
        d = 128

        input = torch.randn(num_tokens, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")
        # gate 부분을 0으로 설정
        input[:, :d] = 0.0

        output = torch.empty(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        _C.silu_and_mul(output, input)

        torch.testing.assert_close(
            output,
            torch.zeros_like(output),
            atol=0, rtol=0,
        )

    def test_silu_and_mul_unit_up(self):
        """
        Test 8: up=1일 때 출력이 silu(gate)이어야 함

        silu(gate) * 1 = silu(gate)
        """
        num_tokens = 16
        d = 128

        gate = torch.randn(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        up = torch.ones(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        input = torch.cat([gate, up], dim=-1)

        # 기대값: silu(gate)
        output_ref = F.silu(gate)

        output = torch.empty(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        _C.silu_and_mul(output, input)

        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)

    def test_performance_vs_pytorch(self):
        """
        Test 9: 성능 벤치마크

        4가지 비교:
        1. PyTorch Non-Fused: F.silu(gate) * up (새 텐서 할당 + 슬라이싱)
        2. PyTorch Non-Fused (inplace): F.silu(gate, inplace=True) → mul_(up)
        3. CUDA Non-Fused: _C.silu(gate) → torch.mul (커널 2회, 할당 없음)
        4. CUDA Fused: _C.silu_and_mul (커널 1회)

        [1]~[3]은 모두 Non-Fused이므로 비슷한 성능을 보입니다.
        PyTorch의 F.silu()도 내부적으로 CUDA 커널이기 때문입니다.
        핵심 차이는 [4] Fused가 중간 버퍼 없이 커널 1회로 처리하는 것입니다.
        """
        num_tokens = 4096
        d = 11008
        warmup = 10
        repeat = 100

        input = torch.randn(num_tokens, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")
        output = torch.empty(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # ----- [1] PyTorch Non-Fused: F.silu + slice + mul (새 텐서 할당) -----
        for _ in range(warmup):
            silu_and_mul_reference(input)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            silu_and_mul_reference(input)
        end.record()
        torch.cuda.synchronize()
        pytorch_us = start.elapsed_time(end) / repeat * 1000

        # ----- [2] PyTorch Non-Fused (inplace): silu(inplace) + mul_ -----
        # 실제 추론에서는 gate를 다시 쓸 일이 없으므로 clone 없이 측정.
        gate = input[:, :d].contiguous()
        up = input[:, d:].contiguous()
        gate_inplace = gate.clone()

        for _ in range(warmup):
            F.silu(gate_inplace, inplace=True)
            gate_inplace.mul_(up)
        torch.cuda.synchronize()

        gate_inplace.copy_(gate)
        start.record()
        for _ in range(repeat):
            F.silu(gate_inplace, inplace=True)
            gate_inplace.mul_(up)
        end.record()
        torch.cuda.synchronize()
        pytorch_inplace_us = start.elapsed_time(end) / repeat * 1000

        # ----- [3] CUDA Non-Fused: silu 커널 + torch.mul (커널 2회) -----
        gate_activated = torch.empty_like(gate)

        for _ in range(warmup):
            _C.silu(gate_activated, gate)
            torch.mul(gate_activated, up, out=output)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            _C.silu(gate_activated, gate)
            torch.mul(gate_activated, up, out=output)
        end.record()
        torch.cuda.synchronize()
        nonfused_us = start.elapsed_time(end) / repeat * 1000

        # ----- [4] CUDA Fused: silu_and_mul (커널 1회) -----
        for _ in range(warmup):
            _C.silu_and_mul(output, input)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            _C.silu_and_mul(output, input)
        end.record()
        torch.cuda.synchronize()
        fused_us = start.elapsed_time(end) / repeat * 1000

        print(f"\n  설정: num_tokens={num_tokens}, d={d}, dtype={DEFAULT_DTYPE}")
        print(f"  [1] PyTorch (F.silu + mul, 할당 O):         {pytorch_us:.1f} us")
        print(f"  [2] PyTorch (inplace silu + mul_):          {pytorch_inplace_us:.1f} us")
        print(f"  [3] CUDA Non-Fused (silu + mul, 커널 2회):  {nonfused_us:.1f} us")
        print(f"  [4] CUDA Fused (silu_and_mul, 커널 1회):    {fused_us:.1f} us")
        print(f"  [1] PyTorch vs [4] Fused:   {pytorch_us / fused_us:.2f}x")
        print(f"  [2] inplace vs [4] Fused:   {pytorch_inplace_us / fused_us:.2f}x")
        print(f"  [3] Non-Fused vs [4] Fused: {nonfused_us / fused_us:.2f}x")


# =============================================================================
# Python 래퍼 테스트 (kernels/ + layers/)
# =============================================================================

class TestActivationWrappers:
    """kernels.activation 래퍼와 layers.activation nn.Module 검증

    _C 직접 호출 결과와 비교하여, Python 래퍼 계층이
    출력 텐서 생성, shape 계산, 인자 전달을 올바르게 수행하는지 확인합니다.
    """

    def test_silu_wrapper(self):
        """
        kernels.activation.silu() 래퍼 검증

        래퍼가 _C.silu()와 동일한 결과를 반환하는지 확인
        """
        num_tokens = 32
        d = 256

        input = torch.randn(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")

        # _C 직접 호출
        output_direct = torch.empty_like(input)
        _C.silu(output_direct, input)

        # 래퍼 호출
        output_wrapper = silu(input)

        assert output_wrapper.shape == input.shape
        assert output_wrapper.dtype == input.dtype
        torch.testing.assert_close(output_wrapper, output_direct, atol=0, rtol=0)

    def test_silu_and_mul_wrapper(self):
        """
        kernels.activation.silu_and_mul() 래퍼 검증

        래퍼가 출력 shape [num_tokens, d]를 올바르게 생성하고
        _C.silu_and_mul()과 동일한 결과를 반환하는지 확인
        """
        num_tokens = 32
        d = 256

        input = torch.randn(num_tokens, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")

        # _C 직접 호출
        output_direct = torch.empty(num_tokens, d, dtype=DEFAULT_DTYPE, device="cuda")
        _C.silu_and_mul(output_direct, input)

        # 래퍼 호출
        output_wrapper = silu_and_mul(input)

        assert output_wrapper.shape == (num_tokens, d)
        assert output_wrapper.dtype == input.dtype
        torch.testing.assert_close(output_wrapper, output_direct, atol=0, rtol=0)

    def test_silu_and_mul_wrapper_3d(self):
        """
        kernels.activation.silu_and_mul() 3D 입력 래퍼 검증

        [batch, seq_len, 2*d] → [batch, seq_len, d] shape 변환이 올바른지 확인
        """
        batch = 4
        seq_len = 32
        d = 256

        input = torch.randn(batch, seq_len, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")

        output_wrapper = silu_and_mul(input)

        assert output_wrapper.shape == (batch, seq_len, d)

        # 참조 결과와도 비교
        output_ref = silu_and_mul_reference(input)
        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output_wrapper, output_ref, **tol)

    def test_silu_and_mul_module(self):
        """
        layers.activation.SiluAndMul nn.Module 검증

        Module.forward()가 kernels.activation.silu_and_mul()과 동일한 결과를 반환하는지 확인
        """
        num_tokens = 32
        d = 256

        input = torch.randn(num_tokens, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")

        # 래퍼 함수 호출
        output_fn = silu_and_mul(input)

        # nn.Module 호출
        module = SiluAndMul()
        output_module = module(input)

        assert output_module.shape == (num_tokens, d)
        torch.testing.assert_close(output_module, output_fn, atol=0, rtol=0)

    def test_silu_and_mul_module_3d(self):
        """
        layers.activation.SiluAndMul nn.Module 3D 입력 검증

        배치 입력 [batch, seq_len, 2*d]에서도 올바르게 동작하는지 확인
        """
        batch = 4
        seq_len = 32
        d = 256

        input = torch.randn(batch, seq_len, 2 * d, dtype=DEFAULT_DTYPE, device="cuda")

        module = SiluAndMul()
        output = module(input)

        assert output.shape == (batch, seq_len, d)

        output_ref = silu_and_mul_reference(input)
        tol = TOLERANCES[DEFAULT_DTYPE]
        torch.testing.assert_close(output, output_ref, **tol)
