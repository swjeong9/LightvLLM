"""LLaMA 모델 테스트.

LLaMA 모델의 각 구성 요소(MLP, Attention 등)를 개별 검증하고,
최종적으로 HuggingFace 구현과 비교하여 전체 모델의 정확성을 확인합니다.

실행:
    uv run pytest tests/models/test_llama.py -v

실행 (MLP만):
    uv run pytest tests/models/test_llama.py::TestLlamaMLP -v
"""

import torch
import torch.nn.functional as F

from lightvllm.models.llama import (
    LlamaMLP, LlamaAttention, LlamaDecoderLayer,
    LlamaModel, LlamaForCausalLM,
)
from lightvllm.layers.linear import Linear, MergedLinear, SHARD_GATE, SHARD_UP
from lightvllm.layers.activation import SiluAndMul
from lightvllm.attention.kv_cache import KVCache

DEVICE = "cuda"

# LlamaModel / LlamaForCausalLM 테스트용 소형 config (GPU 메모리 절약)
SMALL_CONFIG = dict(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=32,
    dtype=torch.bfloat16,
)


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


class TestLlamaMLPBenchmark:
    """LlamaMLP Fused vs Non-Fused 성능 벤치마크.

    비교 대상:
    1. Non-Fused: 별도 gate_proj + up_proj (GEMM 2회) + F.silu + mul + down_proj
    2. Fused: MergedLinear (GEMM 1회) + SiluAndMul (CUDA 커널 1회) + down_proj

    MergedLinear의 GEMM 융합과 SiluAndMul의 커널 퓨전이 합쳐진 효과를 측정합니다.
    """

    def test_fused_vs_nonfused_performance(self):
        """Fused MLP vs Non-Fused MLP 성능 비교.

        LLaMA-3 8B 크기 (hidden=4096, intermediate=14336)에서 측정합니다.
        """
        hidden, inter = 4096, 14336
        num_tokens = 128
        dtype = torch.bfloat16
        warmup = 10
        repeat = 100

        # --- Fused 경로: LlamaMLP (MergedLinear + SiluAndMul) ---
        mlp_fused = LlamaMLP(hidden, inter, dtype=dtype).to(DEVICE)
        x = torch.randn(num_tokens, hidden, dtype=dtype, device=DEVICE)

        for _ in range(warmup):
            mlp_fused(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(repeat):
            mlp_fused(x)
        end.record()
        torch.cuda.synchronize()
        fused_us = start.elapsed_time(end) / repeat * 1000

        # --- Non-Fused 경로: 별도 Linear×2 + F.silu + mul ---
        gate_proj = Linear(hidden, inter, dtype=dtype).to(DEVICE)
        up_proj = Linear(hidden, inter, dtype=dtype).to(DEVICE)
        down_proj = Linear(inter, hidden, dtype=dtype).to(DEVICE)

        for _ in range(warmup):
            gate_out = gate_proj(x)
            up_out = up_proj(x)
            activated = F.silu(gate_out) * up_out
            down_proj(activated)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeat):
            gate_out = gate_proj(x)
            up_out = up_proj(x)
            activated = F.silu(gate_out) * up_out
            down_proj(activated)
        end.record()
        torch.cuda.synchronize()
        nonfused_us = start.elapsed_time(end) / repeat * 1000

        print(f"\n  설정: num_tokens={num_tokens}, hidden={hidden}, "
              f"intermediate={inter}, dtype={dtype}")
        print(f"  [1] Non-Fused (Linear×2 + F.silu + mul + Linear): {nonfused_us:.1f} us")
        print(f"  [2] Fused (MergedLinear + SiluAndMul + Linear):   {fused_us:.1f} us")
        print(f"  Non-Fused vs Fused: {nonfused_us / fused_us:.2f}x")


class TestLlamaAttention:
    """LlamaAttention 기능 테스트.

    QKV 프로젝션(MergedLinear), RoPE, GQA, KV Cache 통합을 검증합니다.
    """

    def test_output_shape(self):
        """기본 출력 shape [num_tokens, hidden_size] 및 dtype 검증."""
        hidden, num_heads, num_kv_heads, head_dim = 256, 8, 4, 32
        attn = LlamaAttention(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        ).to(DEVICE)

        positions = torch.arange(16, device=DEVICE)
        x = torch.randn(16, hidden, dtype=torch.bfloat16, device=DEVICE)

        out = attn(positions, x)

        assert out.shape == (16, hidden)
        assert out.dtype == torch.bfloat16

    def test_gqa(self):
        """GQA 동작 검증 (Llama-3.2-3B 스펙: num_heads=24, num_kv_heads=8).

        GQA ratio 3:1에서 shape이 올바르고 NaN/Inf가 없는지 확인합니다.
        """
        num_heads, num_kv_heads, head_dim = 24, 8, 128
        hidden = num_heads * head_dim  # 3072

        attn = LlamaAttention(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        ).to(DEVICE)

        positions = torch.arange(4, device=DEVICE)
        x = torch.randn(4, hidden, dtype=torch.bfloat16, device=DEVICE)

        out = attn(positions, x)

        assert out.shape == (4, hidden)
        assert not torch.isnan(out).any(), "출력에 NaN이 있습니다"
        assert not torch.isinf(out).any(), "출력에 Inf가 있습니다"

    def test_rope_applied(self):
        """RoPE가 적용되어 위치에 따라 출력이 달라지는지 검증.

        동일 hidden_states, 동일 weight, 다른 positions → 출력이 달라야 합니다.
        """
        hidden, num_heads, num_kv_heads, head_dim = 256, 8, 4, 32

        attn = LlamaAttention(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        ).to(DEVICE)
        # torch.empty weight → 초기화 필요 (0이나 NaN 방지)
        for p in attn.parameters():
            torch.nn.init.normal_(p, std=0.02)

        x = torch.randn(4, hidden, dtype=torch.bfloat16, device=DEVICE)

        pos1 = torch.arange(0, 4, device=DEVICE)
        pos2 = torch.arange(10, 14, device=DEVICE)

        out1 = attn(pos1, x)
        out2 = attn(pos2, x)

        # bf16 정밀도에서 RoPE 차이가 작을 수 있으므로 엄격한 tolerance 사용
        assert not torch.allclose(out1, out2, atol=1e-5, rtol=1e-5), \
            "다른 positions에서 동일한 출력 — RoPE가 적용되지 않았을 수 있습니다"

    def test_prefill_with_kv_cache(self):
        """KV Cache와 함께 prefill이 에러 없이 실행되는지 확인.

        LlamaAttention.forward에 kv_cache를 전달하고,
        shape이 올바른지 검증합니다.
        NOTE: advance()는 LlamaAttention이 아닌 LlamaModel이 호출합니다.
        """
        hidden, num_heads, num_kv_heads, head_dim = 256, 8, 4, 32
        num_tokens = 8

        attn = LlamaAttention(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        ).to(DEVICE)

        cache = KVCache(
            num_layers=1,
            max_seq_len=64,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device=DEVICE,
        )

        positions = torch.arange(num_tokens, device=DEVICE)
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=DEVICE)

        out = attn(positions, x, kv_cache=cache, layer_idx=0)

        assert out.shape == (num_tokens, hidden)
        assert out.dtype == torch.bfloat16

    def test_decode_with_kv_cache(self):
        """Prefill 후 1토큰 decode가 정상 동작하는지 확인.

        수동으로 cache.advance()를 호출하여 prefill → decode 흐름을 시뮬레이션합니다.
        """
        hidden, num_heads, num_kv_heads, head_dim = 256, 8, 4, 32
        prompt_len = 8

        attn = LlamaAttention(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        ).to(DEVICE)

        cache = KVCache(
            num_layers=1,
            max_seq_len=64,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device=DEVICE,
        )

        # Prefill
        positions_prefill = torch.arange(prompt_len, device=DEVICE)
        x_prefill = torch.randn(prompt_len, hidden, dtype=torch.bfloat16, device=DEVICE)
        attn(positions_prefill, x_prefill, kv_cache=cache, layer_idx=0)
        cache.advance(prompt_len)

        # Decode: 1토큰
        positions_decode = torch.tensor([prompt_len], device=DEVICE)
        x_decode = torch.randn(1, hidden, dtype=torch.bfloat16, device=DEVICE)
        out = attn(positions_decode, x_decode, kv_cache=cache, layer_idx=0)

        assert out.shape == (1, hidden)
        assert out.dtype == torch.bfloat16

    def test_prefill_decode_consistency(self):
        """Cache decode가 에러 없이 동작하고 올바른 shape을 반환하는지 검증.

        4 tokens prefill → 1 token decode 흐름이 정상 동작하는지 확인합니다.
        """
        hidden, num_heads, num_kv_heads, head_dim = 256, 8, 4, 32
        prompt_len = 4

        attn = LlamaAttention(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        ).to(DEVICE)
        for p in attn.parameters():
            torch.nn.init.normal_(p, std=0.02)

        cache = KVCache(
            num_layers=1,
            max_seq_len=64,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device=DEVICE,
        )

        # Prefill 4 tokens
        positions_prefill = torch.arange(prompt_len, device=DEVICE)
        x_prefill = torch.randn(prompt_len, hidden, dtype=torch.bfloat16, device=DEVICE)
        out_prefill = attn(positions_prefill, x_prefill, kv_cache=cache, layer_idx=0)
        cache.advance(prompt_len)

        assert out_prefill.shape == (prompt_len, hidden)

        # Decode 1 token
        positions_decode = torch.tensor([prompt_len], device=DEVICE)
        x_decode = torch.randn(1, hidden, dtype=torch.bfloat16, device=DEVICE)
        out_decode = attn(positions_decode, x_decode, kv_cache=cache, layer_idx=0)

        assert out_decode.shape == (1, hidden)
        assert not torch.isnan(out_decode).any(), "Decode 출력에 NaN이 있습니다"
        assert not torch.isinf(out_decode).any(), "Decode 출력에 Inf가 있습니다"

    def test_weight_loading_qkv(self):
        """MergedLinear의 weight_loader로 Q/K/V shard 로딩 검증.

        별도 q_w, k_w, v_w를 weight_loader로 로딩한 후,
        qkv_proj.weight의 올바른 위치에 저장되었는지 확인합니다.
        """
        hidden, num_heads, num_kv_heads, head_dim = 256, 8, 4, 32
        q_size = num_heads * head_dim      # 256
        kv_size = num_kv_heads * head_dim  # 128
        dtype = torch.bfloat16

        attn = LlamaAttention(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ).to(DEVICE)

        # 별도 Q/K/V weight 생성
        q_w = torch.randn(q_size, hidden, dtype=dtype, device=DEVICE)
        k_w = torch.randn(kv_size, hidden, dtype=dtype, device=DEVICE)
        v_w = torch.randn(kv_size, hidden, dtype=dtype, device=DEVICE)

        # weight_loader로 각 shard 로딩
        attn.qkv_proj.weight_loader(attn.qkv_proj.weight, q_w, shard_id="q")
        attn.qkv_proj.weight_loader(attn.qkv_proj.weight, k_w, shard_id="k")
        attn.qkv_proj.weight_loader(attn.qkv_proj.weight, v_w, shard_id="v")

        # 올바른 위치에 저장되었는지 검증
        loaded_q = attn.qkv_proj.weight.data[:q_size]
        loaded_k = attn.qkv_proj.weight.data[q_size:q_size + kv_size]
        loaded_v = attn.qkv_proj.weight.data[q_size + kv_size:]

        torch.testing.assert_close(loaded_q, q_w, atol=0, rtol=0)
        torch.testing.assert_close(loaded_k, k_w, atol=0, rtol=0)
        torch.testing.assert_close(loaded_v, v_w, atol=0, rtol=0)

    def test_dtype_bf16(self):
        """bfloat16에서 forward 결과의 dtype이 bf16인지 확인."""
        hidden, num_heads, num_kv_heads, head_dim = 256, 8, 4, 32

        attn = LlamaAttention(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        ).to(DEVICE)

        positions = torch.arange(8, device=DEVICE)
        x = torch.randn(8, hidden, dtype=torch.bfloat16, device=DEVICE)

        out = attn(positions, x)

        assert out.dtype == torch.bfloat16
        assert out.shape == (8, hidden)


class TestLlamaDecoderLayer:
    """LlamaDecoderLayer 기능 테스트.

    Attention + MLP + RMSNorm (Pre-Norm Residual) 구조의 단일 Decoder Layer를 검증합니다.
    residual=None (첫 layer) / residual!=None (이후 layer, fused add+norm) 동작을 포함합니다.
    """

    def test_output_shape_and_residual(self):
        """반환값이 (hidden_states, residual) 튜플이고 올바른 shape인지 검증."""
        hidden, inter = 256, 512
        num_heads, num_kv_heads, head_dim = 8, 4, 32
        dtype = torch.bfloat16

        layer = LlamaDecoderLayer(
            hidden_size=hidden,
            intermediate_size=inter,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ).to(DEVICE)

        # torch.empty weight → 초기화 필요
        for p in layer.parameters():
            torch.nn.init.normal_(p, std=0.02)

        positions = torch.arange(8, device=DEVICE)
        x = torch.randn(8, hidden, dtype=dtype, device=DEVICE)

        result = layer(positions, x, residual=None)

        assert isinstance(result, tuple), "반환값이 tuple이어야 합니다"
        assert len(result) == 2, "반환값이 (hidden_states, residual) 2-tuple이어야 합니다"

        hidden_states, residual = result
        assert hidden_states.shape == (8, hidden)
        assert residual.shape == (8, hidden)

    def test_first_layer_residual_none(self):
        """residual=None (첫 layer)에서 에러 없이 동작하고 NaN/Inf가 없는지 검증."""
        hidden, inter = 256, 512
        num_heads, num_kv_heads, head_dim = 8, 4, 32
        dtype = torch.bfloat16

        layer = LlamaDecoderLayer(
            hidden_size=hidden,
            intermediate_size=inter,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ).to(DEVICE)

        for p in layer.parameters():
            torch.nn.init.normal_(p, std=0.02)

        positions = torch.arange(8, device=DEVICE)
        x = torch.randn(8, hidden, dtype=dtype, device=DEVICE)

        hidden_states, residual = layer(positions, x, residual=None)

        assert not torch.isnan(hidden_states).any(), "hidden_states에 NaN이 있습니다"
        assert not torch.isinf(hidden_states).any(), "hidden_states에 Inf가 있습니다"
        assert not torch.isnan(residual).any(), "residual에 NaN이 있습니다"
        assert not torch.isinf(residual).any(), "residual에 Inf가 있습니다"

    def test_subsequent_layer_fused_norm(self):
        """두 layer를 체이닝하여 residual!=None에서 fused norm이 정상 동작하는지 검증.

        첫 번째 layer의 (hidden_states, residual) 출력을
        두 번째 layer의 입력으로 전달하여 fused add+norm 경로를 테스트합니다.
        """
        hidden, inter = 256, 512
        num_heads, num_kv_heads, head_dim = 8, 4, 32
        dtype = torch.bfloat16

        layer1 = LlamaDecoderLayer(
            hidden_size=hidden,
            intermediate_size=inter,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ).to(DEVICE)
        layer2 = LlamaDecoderLayer(
            hidden_size=hidden,
            intermediate_size=inter,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ).to(DEVICE)

        for p in layer1.parameters():
            torch.nn.init.normal_(p, std=0.02)
        for p in layer2.parameters():
            torch.nn.init.normal_(p, std=0.02)

        positions = torch.arange(8, device=DEVICE)
        x = torch.randn(8, hidden, dtype=dtype, device=DEVICE)

        # 첫 번째 layer: residual=None
        hidden_states, residual = layer1(positions, x, residual=None)

        # 두 번째 layer: residual!=None (fused add+norm 경로)
        hidden_states2, residual2 = layer2(positions, hidden_states, residual=residual)

        assert hidden_states2.shape == (8, hidden)
        assert residual2.shape == (8, hidden)
        assert not torch.isnan(hidden_states2).any(), "두 번째 layer hidden_states에 NaN이 있습니다"
        assert not torch.isinf(hidden_states2).any(), "두 번째 layer hidden_states에 Inf가 있습니다"

    def test_with_kv_cache(self):
        """KVCache와 함께 prefill + decode 동작 검증.

        prefill 8 tokens → cache.advance(8) → decode 1 token 흐름을 테스트합니다.
        """
        hidden, inter = 256, 512
        num_heads, num_kv_heads, head_dim = 8, 4, 32
        dtype = torch.bfloat16

        layer = LlamaDecoderLayer(
            hidden_size=hidden,
            intermediate_size=inter,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ).to(DEVICE)

        for p in layer.parameters():
            torch.nn.init.normal_(p, std=0.02)

        cache = KVCache(
            num_layers=1,
            max_seq_len=64,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=DEVICE,
        )

        # Prefill 8 tokens
        positions_prefill = torch.arange(8, device=DEVICE)
        x_prefill = torch.randn(8, hidden, dtype=dtype, device=DEVICE)
        hidden_states, residual = layer(
            positions_prefill, x_prefill, residual=None,
            kv_cache=cache, layer_idx=0,
        )

        assert hidden_states.shape == (8, hidden)
        cache.advance(8)

        # Decode 1 token
        positions_decode = torch.tensor([8], device=DEVICE)
        x_decode = torch.randn(1, hidden, dtype=dtype, device=DEVICE)
        hidden_states_dec, residual_dec = layer(
            positions_decode, x_decode, residual=None,
            kv_cache=cache, layer_idx=0,
        )

        assert hidden_states_dec.shape == (1, hidden)
        assert residual_dec.shape == (1, hidden)

    def test_dtype(self):
        """bf16 dtype이 출력에서 보존되는지 검증."""
        hidden, inter = 256, 512
        num_heads, num_kv_heads, head_dim = 8, 4, 32
        dtype = torch.bfloat16

        layer = LlamaDecoderLayer(
            hidden_size=hidden,
            intermediate_size=inter,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ).to(DEVICE)

        for p in layer.parameters():
            torch.nn.init.normal_(p, std=0.02)

        positions = torch.arange(8, device=DEVICE)
        x = torch.randn(8, hidden, dtype=dtype, device=DEVICE)

        hidden_states, residual = layer(positions, x, residual=None)

        assert hidden_states.dtype == torch.bfloat16, \
            f"hidden_states dtype이 bf16이어야 하지만 {hidden_states.dtype}입니다"
        assert residual.dtype == torch.bfloat16, \
            f"residual dtype이 bf16이어야 하지만 {residual.dtype}입니다"


class TestLlamaModel:
    """LlamaModel (Embedding + DecoderLayer 스택 + 최종 RMSNorm) 테스트."""

    def test_forward_shape(self):
        """출력 shape이 [num_tokens, hidden_size]인지 검증."""
        model = LlamaModel(**SMALL_CONFIG).to(DEVICE)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)

        input_ids = torch.randint(0, 256, (8,), device=DEVICE)
        positions = torch.arange(8, device=DEVICE)

        out = model(input_ids, positions)

        assert out.shape == (8, 128), f"Expected (8, 128), got {out.shape}"
        assert out.dtype == torch.bfloat16

    def test_multi_layer(self):
        """2-layer 모델이 정상 동작하고 NaN/Inf가 없는지 검증."""
        model = LlamaModel(**SMALL_CONFIG).to(DEVICE)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)

        input_ids = torch.randint(0, 256, (8,), device=DEVICE)
        positions = torch.arange(8, device=DEVICE)

        out = model(input_ids, positions)

        assert not torch.isnan(out).any(), "출력에 NaN이 있습니다"
        assert not torch.isinf(out).any(), "출력에 Inf가 있습니다"

    def test_with_kv_cache(self):
        """KV Cache와 함께 prefill 8 tokens → decode 1 token 동작 검증.

        NOTE: model.forward 내부에서 cache.advance()를 호출하므로
        수동 호출이 불필요합니다.
        """
        model = LlamaModel(**SMALL_CONFIG).to(DEVICE)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)

        cache = KVCache(
            num_layers=2,
            max_seq_len=64,
            num_kv_heads=2,
            head_dim=32,
            dtype=torch.bfloat16,
            device=DEVICE,
        )

        # Prefill 8 tokens
        input_ids = torch.randint(0, 256, (8,), device=DEVICE)
        positions = torch.arange(8, device=DEVICE)
        out_prefill = model(input_ids, positions, kv_cache=cache)

        assert out_prefill.shape == (8, 128)

        # Decode 1 token
        input_ids_dec = torch.randint(0, 256, (1,), device=DEVICE)
        positions_dec = torch.tensor([8], device=DEVICE)
        out_decode = model(input_ids_dec, positions_dec, kv_cache=cache)

        assert out_decode.shape == (1, 128)
        assert not torch.isnan(out_decode).any(), "Decode 출력에 NaN이 있습니다"
        assert not torch.isinf(out_decode).any(), "Decode 출력에 Inf가 있습니다"


class TestLlamaForCausalLM:
    """LlamaForCausalLM (LlamaModel + LM Head) 테스트."""

    def test_logits_shape(self):
        """출력 logits shape이 [num_tokens, vocab_size]인지 검증."""
        model = LlamaForCausalLM(**SMALL_CONFIG).to(DEVICE)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)

        input_ids = torch.randint(0, 256, (8,), device=DEVICE)
        positions = torch.arange(8, device=DEVICE)

        logits = model(input_ids, positions)

        assert logits.shape == (8, 256), f"Expected (8, 256), got {logits.shape}"

    def test_logits_fp32(self):
        """bf16 모델이어도 logits dtype이 항상 float32인지 검증."""
        model = LlamaForCausalLM(**SMALL_CONFIG).to(DEVICE)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)

        input_ids = torch.randint(0, 256, (8,), device=DEVICE)
        positions = torch.arange(8, device=DEVICE)

        logits = model(input_ids, positions)

        assert logits.dtype == torch.float32, \
            f"logits dtype이 float32이어야 하지만 {logits.dtype}입니다"

    def test_greedy_generate(self):
        """간단한 greedy decode loop (3토큰 생성)이 에러 없이 동작하는지 검증.

        prefill 4 tokens → argmax → decode 1 token을 3회 반복합니다.
        생성된 토큰이 vocab 범위 안에 있는지 확인합니다.
        """
        model = LlamaForCausalLM(**SMALL_CONFIG).to(DEVICE)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)

        cache = KVCache(
            num_layers=2,
            max_seq_len=64,
            num_kv_heads=2,
            head_dim=32,
            dtype=torch.bfloat16,
            device=DEVICE,
        )

        vocab_size = SMALL_CONFIG["vocab_size"]

        # Prefill 4 tokens
        prompt = torch.randint(0, vocab_size, (4,), device=DEVICE)
        positions = torch.arange(4, device=DEVICE)

        logits = model(prompt, positions, kv_cache=cache)
        assert logits.shape == (4, vocab_size)

        # argmax로 다음 토큰 결정 (마지막 토큰의 logits 사용)
        next_token = logits[-1].argmax(dim=-1, keepdim=True)  # [1]

        generated_tokens = []
        current_pos = 4

        # Decode 3 tokens
        for _ in range(3):
            positions_dec = torch.tensor([current_pos], device=DEVICE)
            logits = model(next_token, positions_dec, kv_cache=cache)

            assert logits.shape == (1, vocab_size)

            next_token = logits[0].argmax(dim=-1, keepdim=True)  # [1]
            generated_tokens.append(next_token.item())
            current_pos += 1

        # 생성된 토큰이 vocab 범위 안에 있는지 검증
        for token_id in generated_tokens:
            assert 0 <= token_id < vocab_size, \
                f"생성된 token {token_id}이 vocab 범위 [0, {vocab_size}) 밖입니다"


class TestLlamaHFValidation:
    """LightvLLM vs HuggingFace 출력 비교 검증.

    LLaMA-3.2-3B-Instruct 체크포인트를 로딩하여
    HuggingFace 구현과 동일한 출력을 내는지 확인합니다.

    실행:
        uv run pytest tests/models/test_llama.py::TestLlamaHFValidation -v -s
    """

    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

    def test_forward_matches_hf(self):
        """from_pretrained 출력이 HuggingFace LlamaForCausalLM과 일치하는지 검증.

        동일 입력에 대해 logits를 비교합니다.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.MODEL_ID
        dtype = torch.bfloat16

        # --- HuggingFace 모델 ---
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype,
        ).to(DEVICE)
        hf_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"].squeeze(0)  # [num_tokens]

        with torch.no_grad():
            hf_output = hf_model(input_ids.unsqueeze(0))
            hf_logits = hf_output.logits.squeeze(0).float()  # [num_tokens, vocab]

        # HF 모델 메모리 해제
        del hf_model
        torch.cuda.empty_cache()

        # --- LightvLLM 모델 ---
        # HF 캐시 경로에서 safetensors를 로딩
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_id)
        light_model = LlamaForCausalLM.from_pretrained(model_dir, dtype=dtype)
        light_model.eval()

        positions = torch.arange(input_ids.shape[0], device=DEVICE)

        with torch.no_grad():
            light_logits = light_model(input_ids, positions)  # [num_tokens, vocab]

        # --- 비교 ---
        # bf16 모델이지만 logits는 fp32.
        # Attention 구현 경로 차이(SDPA vs HF의 eager/SDPA)와 28-layer 누적으로
        # 개별 logit에 최대 ~0.15 차이가 발생할 수 있으나, 전체적으로 일치.
        # argmax 예측 토큰 일치로 functional correctness를 검증.
        max_diff = (light_logits - hf_logits).abs().max().item()
        mean_diff = (light_logits - hf_logits).abs().mean().item()
        assert max_diff < 0.5, f"max abs diff가 너무 큽니다: {max_diff:.4f}"
        assert mean_diff < 0.1, f"mean abs diff가 너무 큽니다: {mean_diff:.4f}"
        print(f"\n  HF 검증 통과! prompt: '{prompt}'")
        print(f"  logits shape: {light_logits.shape}")
        print(f"  max abs diff: {(light_logits - hf_logits).abs().max().item():.6f}")

        # 동일 토큰 예측 확인
        hf_pred = hf_logits[-1].argmax().item()
        light_pred = light_logits[-1].argmax().item()
        pred_token = tokenizer.decode([light_pred])
        print(f"  HF 예측: {hf_pred} ({tokenizer.decode([hf_pred])})")
        print(f"  Light 예측: {light_pred} ({pred_token})")
        assert hf_pred == light_pred, \
            f"예측 토큰 불일치: HF={hf_pred}, Light={light_pred}"

        del light_model
        torch.cuda.empty_cache()
