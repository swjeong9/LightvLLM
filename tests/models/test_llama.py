"""LLaMA 모델 테스트.

LLaMA 모델의 각 구성 요소(MLP, Attention 등)를 개별 검증하고,
최종적으로 HuggingFace 구현과 비교하여 전체 모델의 정확성을 확인합니다.
"""

import torch
import torch.nn.functional as F
import pytest

from lightvllm.models.llama import LlamaMLP
from lightvllm.layers.linear import SHARD_GATE, SHARD_UP

DEVICE = "cuda"


class TestLlamaMLP:
    """LlamaMLP 기능 테스트."""

    def test_output_shape(self):
        """출력 shape이 [num_tokens, hidden_size]인지 검증."""
        hidden, inter = 256, 512
        mlp = LlamaMLP(hidden, inter, dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(32, hidden, dtype=torch.bfloat16, device=DEVICE)

        out = mlp(x)

        assert out.shape == (32, hidden)
        assert out.dtype == torch.bfloat16

    def test_fused_vs_separate(self):
        """융합 경로(MergedLinear)와 별도 경로(Linear×2)의 수치 일치 검증.

        동일 weight를 공유하되, 한쪽은 LlamaMLP(MergedLinear 경로),
        다른 쪽은 별도 gate_proj/up_proj Linear로 수동 계산하여 비교합니다.
        """
        hidden, inter = 128, 256
        dtype = torch.bfloat16

        # --- 별도 경로용 weight 생성 ---
        gate_w = torch.randn(inter, hidden, dtype=dtype, device=DEVICE)
        up_w = torch.randn(inter, hidden, dtype=dtype, device=DEVICE)
        down_w = torch.randn(hidden, inter, dtype=dtype, device=DEVICE)

        # --- 융합 경로: LlamaMLP ---
        mlp = LlamaMLP(hidden, inter, dtype=dtype).to(DEVICE)
        # weight_loader로 별도 weight를 융합 weight에 로딩
        mlp.gate_up_proj.weight_loader(
            mlp.gate_up_proj.weight, gate_w, shard_id=SHARD_GATE
        )
        mlp.gate_up_proj.weight_loader(
            mlp.gate_up_proj.weight, up_w, shard_id=SHARD_UP
        )
        mlp.down_proj.weight.data.copy_(down_w)

        # --- 별도 경로: 수동 계산 ---
        x = torch.randn(16, hidden, dtype=dtype, device=DEVICE)

        # 융합 경로
        fused_out = mlp(x)

        # 별도 경로: silu(x @ gate_w.T) * (x @ up_w.T) → down_proj
        gate_out = F.linear(x, gate_w)     # [16, inter]
        up_out = F.linear(x, up_w)         # [16, inter]
        activated = F.silu(gate_out) * up_out  # [16, inter]
        separate_out = F.linear(activated, down_w)  # [16, hidden]

        torch.testing.assert_close(fused_out, separate_out, atol=1e-2, rtol=1e-2)

    def test_dtype_fp16(self):
        """float16 dtype에서 정상 동작 확인."""
        hidden, inter = 128, 256
        mlp = LlamaMLP(hidden, inter, dtype=torch.float16).to(DEVICE)
        x = torch.randn(8, hidden, dtype=torch.float16, device=DEVICE)

        out = mlp(x)

        assert out.dtype == torch.float16
        assert out.shape == (8, hidden)

    def test_dtype_bf16(self):
        """bfloat16 dtype에서 정상 동작 확인."""
        hidden, inter = 128, 256
        mlp = LlamaMLP(hidden, inter, dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(8, hidden, dtype=torch.bfloat16, device=DEVICE)

        out = mlp(x)

        assert out.dtype == torch.bfloat16
        assert out.shape == (8, hidden)

    def test_llama_8b_dims(self):
        """LLaMA-3 8B 실제 크기 (hidden=4096, intermediate=14336)에서 shape 검증."""
        hidden, inter = 4096, 14336
        mlp = LlamaMLP(hidden, inter, dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(4, hidden, dtype=torch.bfloat16, device=DEVICE)

        out = mlp(x)

        assert out.shape == (4, hidden)

    def test_zero_input(self):
        """영벡터 입력 → 영벡터 출력 (SiLU(0)=0이므로 gate 출력=0)."""
        hidden, inter = 128, 256
        mlp = LlamaMLP(hidden, inter, dtype=torch.bfloat16).to(DEVICE)
        # torch.empty weight에 NaN이 있을 수 있으므로 초기화
        # (0 * NaN = NaN 방지)
        torch.nn.init.normal_(mlp.gate_up_proj.weight)
        torch.nn.init.normal_(mlp.down_proj.weight)
        x = torch.zeros(8, hidden, dtype=torch.bfloat16, device=DEVICE)

        out = mlp(x)

        # SiLU(0) = 0 * sigmoid(0) = 0, 따라서 gate 경로가 0 → 최종 출력도 0
        torch.testing.assert_close(
            out,
            torch.zeros_like(out),
            atol=0, rtol=0,
        )

    def test_weight_loading(self):
        """weight_loader로 HF 스타일 별도 가중치 로딩 후 forward 검증.

        실제 추론 시나리오: 체크포인트에서 gate_proj, up_proj, down_proj를
        각각 로딩하고, forward 결과가 올바른지 확인합니다.
        """
        hidden, inter = 64, 128
        dtype = torch.bfloat16

        # 체크포인트에서 로드한 것처럼 별도 weight 준비
        gate_w = torch.randn(inter, hidden, dtype=dtype, device=DEVICE)
        up_w = torch.randn(inter, hidden, dtype=dtype, device=DEVICE)
        down_w = torch.randn(hidden, inter, dtype=dtype, device=DEVICE)

        mlp = LlamaMLP(hidden, inter, dtype=dtype).to(DEVICE)

        # HF 체크포인트 로딩 시뮬레이션
        mlp.gate_up_proj.weight_loader(
            mlp.gate_up_proj.weight, gate_w, shard_id=SHARD_GATE
        )
        mlp.gate_up_proj.weight_loader(
            mlp.gate_up_proj.weight, up_w, shard_id=SHARD_UP
        )
        mlp.down_proj.weight.data.copy_(down_w)

        # gate_up_proj weight의 각 shard가 올바르게 로딩되었는지 직접 검증
        loaded_gate = mlp.gate_up_proj.weight.data[:inter]
        loaded_up = mlp.gate_up_proj.weight.data[inter:]
        torch.testing.assert_close(loaded_gate, gate_w, atol=0, rtol=0)
        torch.testing.assert_close(loaded_up, up_w, atol=0, rtol=0)

        # forward가 에러 없이 실행되고 올바른 shape 반환
        x = torch.randn(4, hidden, dtype=dtype, device=DEVICE)
        out = mlp(x)
        assert out.shape == (4, hidden)


class TestLLaMAModel:
    """LLaMA 전체 모델 테스트."""

    def test_llama_forward(self):
        """LLaMA forward pass가 HuggingFace와 일치하는지 검증."""
        # TODO: 모델 완성 후 구현
        pass

    def test_llama_generate(self):
        """LLaMA 토큰 생성 테스트."""
        # TODO: 모델 완성 후 구현
        pass
