"""
Linear 레이어 테스트

Linear와 MergedLinear 클래스의 정확성을 검증합니다.
MergedLinear의 weight_loader를 통한 HuggingFace 가중치 로딩도 테스트합니다.

실행:
    uv run pytest tests/layers/test_linear.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from lightvllm.layers.linear import Linear, MergedLinear, SHARD_GATE, SHARD_UP

DEFAULT_DTYPE = torch.bfloat16


# =============================================================================
# TestLinear: 기본 Linear 레이어 테스트
# =============================================================================


class TestLinear:
    """기본 Linear 레이어 테스트."""

    def test_linear_basic(self):
        """forward 결과가 F.linear과 bit-exact 일치하는지 확인."""
        torch.manual_seed(42)
        layer = Linear(128, 256, dtype=DEFAULT_DTYPE).cuda()
        torch.nn.init.normal_(layer.weight)
        x = torch.randn(32, 128, dtype=DEFAULT_DTYPE, device="cuda")

        output = layer(x)
        expected = F.linear(x, layer.weight, layer.bias)

        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "input_size,output_size",
        [(64, 128), (256, 512), (4096, 11008), (128, 64)],
    )
    def test_linear_shapes(self, input_size, output_size):
        """다양한 input/output 크기 조합에서 shape과 dtype이 올바른지 확인."""
        layer = Linear(input_size, output_size, dtype=DEFAULT_DTYPE).cuda()
        num_tokens = 16
        x = torch.randn(num_tokens, input_size, dtype=DEFAULT_DTYPE, device="cuda")

        output = layer(x)

        assert output.shape == (num_tokens, output_size)
        assert output.dtype == DEFAULT_DTYPE

    def test_linear_no_bias(self):
        """기본값 bias=False일 때 bias 파라미터가 없는지 확인."""
        layer = Linear(128, 256)
        assert layer.bias is None
        assert "bias" not in dict(layer.named_parameters())

    def test_linear_with_bias(self):
        """bias=True일 때 bias가 실제로 적용되는지 확인."""
        layer = Linear(128, 256, bias=True, dtype=DEFAULT_DTYPE).cuda()
        torch.nn.init.normal_(layer.weight)
        torch.nn.init.normal_(layer.bias)
        assert layer.bias is not None

        x = torch.randn(16, 128, dtype=DEFAULT_DTYPE, device="cuda")
        output = layer(x)

        # bias가 실제로 결과에 영향을 주는지 확인
        output_no_bias = F.linear(x, layer.weight, None)
        assert not torch.equal(output, output_no_bias)

        # F.linear(x, weight, bias)와 bit-exact 일치
        expected = F.linear(x, layer.weight, layer.bias)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_linear_dtypes(self, dtype):
        """half precision dtype (fp16, bf16)에서 정상 동작하는지 확인."""
        layer = Linear(128, 256, dtype=dtype).cuda()
        x = torch.randn(16, 128, dtype=dtype, device="cuda")

        output = layer(x)

        assert output.dtype == dtype
        assert output.shape == (16, 256)

    def test_linear_weight_shape(self):
        """weight shape이 [output_size, input_size] (F.linear 규약)인지 확인."""
        layer = Linear(128, 256)
        assert layer.weight.shape == (256, 128)

    def test_linear_3d_input(self):
        """3D 입력 [batch, seq_len, hidden_size]에서도 동작하는지 확인."""
        layer = Linear(128, 256, dtype=DEFAULT_DTYPE).cuda()
        x = torch.randn(2, 16, 128, dtype=DEFAULT_DTYPE, device="cuda")

        output = layer(x)

        assert output.shape == (2, 16, 256)


# =============================================================================
# TestMergedLinear: 융합 프로젝션 레이어 테스트
# =============================================================================


class TestMergedLinear:
    """MergedLinear (융합 프로젝션) 테스트."""

    def test_merged_linear_gate_up(self):
        """gate+up 융합 결과가 별도 Linear 2개의 concat과 일치하는지 확인."""
        torch.manual_seed(42)
        input_size = 128
        intermediate_size = 256
        num_tokens = 32

        # 별도 프로젝션
        gate_proj = torch.nn.Linear(input_size, intermediate_size, bias=False).cuda()
        up_proj = torch.nn.Linear(input_size, intermediate_size, bias=False).cuda()

        # 융합 프로젝션: gate와 up weight를 쌓음
        merged = MergedLinear(
            input_size=input_size,
            output_sizes=[intermediate_size, intermediate_size],
        ).cuda()
        merged.weight.data.copy_(
            torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0)
        )

        x = torch.randn(num_tokens, input_size, device="cuda")

        # 별도 실행 후 concat
        expected = torch.cat([gate_proj(x), up_proj(x)], dim=-1)
        output = merged(x)

        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_merged_linear_qkv(self):
        """Q+K+V 융합 결과가 별도 Linear 3개의 concat과 일치하는지 확인."""
        torch.manual_seed(42)
        hidden_size = 256
        num_heads = 4
        num_kv_heads = 2
        head_dim = 64
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        num_tokens = 16

        # 별도 프로젝션
        q_proj = torch.nn.Linear(hidden_size, q_size, bias=False).cuda()
        k_proj = torch.nn.Linear(hidden_size, kv_size, bias=False).cuda()
        v_proj = torch.nn.Linear(hidden_size, kv_size, bias=False).cuda()

        # 융합 프로젝션
        merged = MergedLinear(
            input_size=hidden_size,
            output_sizes=[q_size, kv_size, kv_size],
        ).cuda()
        merged.weight.data.copy_(
            torch.cat([q_proj.weight.data, k_proj.weight.data, v_proj.weight.data], dim=0)
        )

        x = torch.randn(num_tokens, hidden_size, device="cuda")

        expected = torch.cat([q_proj(x), k_proj(x), v_proj(x)], dim=-1)
        output = merged(x)

        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_merged_linear_output_shape(self):
        """출력 shape이 [num_tokens, sum(output_sizes)]인지 확인."""
        merged = MergedLinear(
            input_size=128,
            output_sizes=[256, 64, 64],
            dtype=DEFAULT_DTYPE,
        ).cuda()
        x = torch.randn(32, 128, dtype=DEFAULT_DTYPE, device="cuda")

        output = merged(x)

        assert output.shape == (32, 256 + 64 + 64)

    def test_weight_loader_gate_up(self):
        """weight_loader로 gate, up shard를 별도 로딩한 결과가 올바른지 확인."""
        torch.manual_seed(42)
        input_size = 128
        intermediate_size = 256

        # HuggingFace 체크포인트를 시뮬레이션하는 별도 weight
        gate_weight = torch.randn(intermediate_size, input_size)
        up_weight = torch.randn(intermediate_size, input_size)

        # weight_loader로 로딩
        merged = MergedLinear(
            input_size=input_size,
            output_sizes=[intermediate_size, intermediate_size],
        ).cuda()
        merged.weight.data.zero_()  # 초기화
        merged.weight_loader(merged.weight, gate_weight.cuda(), shard_id=SHARD_GATE)
        merged.weight_loader(merged.weight, up_weight.cuda(), shard_id=SHARD_UP)

        # weight가 올바르게 로딩되었는지 확인
        expected_weight = torch.cat([gate_weight, up_weight], dim=0).cuda()
        torch.testing.assert_close(merged.weight.data, expected_weight, atol=0, rtol=0)

        # forward 결과도 일치하는지 확인
        x = torch.randn(16, input_size, device="cuda")
        output = merged(x)
        expected_output = F.linear(x, expected_weight)
        torch.testing.assert_close(output, expected_output, atol=1e-5, rtol=1e-5)

    def test_weight_loader_qkv(self):
        """weight_loader로 Q, K, V를 문자열 shard_id로 별도 로딩한 결과가 올바른지 확인."""
        torch.manual_seed(42)
        hidden_size = 256
        q_size = 256   # 4 heads * 64 dim
        kv_size = 128  # 2 heads * 64 dim

        # HuggingFace q/k/v weight 시뮬레이션
        q_weight = torch.randn(q_size, hidden_size)
        k_weight = torch.randn(kv_size, hidden_size)
        v_weight = torch.randn(kv_size, hidden_size)

        merged = MergedLinear(
            input_size=hidden_size,
            output_sizes=[q_size, kv_size, kv_size],
        ).cuda()
        merged.weight.data.zero_()
        merged.weight_loader(merged.weight, q_weight.cuda(), shard_id="q")
        merged.weight_loader(merged.weight, k_weight.cuda(), shard_id="k")
        merged.weight_loader(merged.weight, v_weight.cuda(), shard_id="v")

        # 검증
        expected_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).cuda()
        torch.testing.assert_close(merged.weight.data, expected_weight, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_merged_linear_dtypes(self, dtype):
        """half precision dtype (fp16, bf16)에서 정상 동작하는지 확인."""
        merged = MergedLinear(
            input_size=128,
            output_sizes=[256, 256],
            dtype=dtype,
        ).cuda()
        x = torch.randn(16, 128, dtype=dtype, device="cuda")

        output = merged(x)

        assert output.dtype == dtype
        assert output.shape == (16, 512)

    def test_merged_linear_offsets(self):
        """내부 오프셋이 올바르게 계산되는지 확인."""
        merged = MergedLinear(
            input_size=128,
            output_sizes=[256, 64, 64],
        )
        assert merged._offsets == [0, 256, 320, 384]

    def test_merged_linear_with_bias(self):
        """bias=True에서 bias shape과 forward가 올바른지 확인."""
        merged = MergedLinear(
            input_size=128,
            output_sizes=[256, 256],
            bias=True,
            dtype=DEFAULT_DTYPE,
        ).cuda()
        assert merged.bias is not None
        assert merged.bias.shape == (512,)

        x = torch.randn(16, 128, dtype=DEFAULT_DTYPE, device="cuda")
        output = merged(x)
        assert output.shape == (16, 512)
