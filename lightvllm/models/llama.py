"""
LLaMA 모델 구현.

LLaMA 아키텍처의 모델별 클래스들을 정의합니다.
공유 빌딩블록(Linear, MergedLinear, SiluAndMul 등)은 lightvllm/layers/에서 가져오고,
여기서는 LLaMA 고유의 아키텍처(MLP 구성, Attention head 배치 등)를 결정합니다.

현재 구현:
    - LlamaMLP: SwiGLU Feed-Forward Network
    - LlamaAttention: Multi-Head Attention + RoPE + GQA + KV Cache
    - LlamaDecoderLayer: Attention + MLP + RMSNorm (Pre-Norm Residual)
    - LlamaModel: Embedding + DecoderLayer 스택 + 최종 RMSNorm
    - LlamaForCausalLM: LM Head + HF 가중치 로딩 (from_pretrained)

vLLM 참조: vLLM/vllm/model_executor/models/llama.py
(단순화: TP, 양자화, hidden_act 파라미터 없음)
"""

import json
import os

import torch
import torch.nn as nn

from lightvllm.layers.activation import SiluAndMul
from lightvllm.layers.linear import Linear, MergedLinear
from lightvllm.layers.rotary_embedding import RotaryEmbedding
from lightvllm.attention.backends.sdpa import SDPAAttention
from lightvllm.layers.normalization import RMSNorm
from lightvllm.attention.kv_cache import KVCache


class LlamaMLP(nn.Module):
    """LLaMA의 SwiGLU Feed-Forward Network.

    gate_proj와 up_proj를 MergedLinear로 융합하여 1회 GEMM으로 처리하고,
    SiluAndMul CUDA 커널로 활성화 후, down_proj로 원래 차원을 복원합니다.

    데이터 흐름::

        x: [num_tokens, hidden_size]
            ↓
        gate_up_proj (MergedLinear)  →  [num_tokens, 2 * intermediate_size]
            ↓
        SiluAndMul (CUDA 커널)      →  [num_tokens, intermediate_size]
            ↓
        down_proj (Linear)           →  [num_tokens, hidden_size]

    SwiGLU란?
        기존 FFN:  ReLU(xW₁)W₂
        SwiGLU:    (SiLU(xW_gate) ⊙ xW_up)W_down

        SiLU(x) = x * sigmoid(x) 를 게이트 함수로 사용하고,
        별도의 up_proj 출력과 element-wise 곱셈하는 구조입니다.
        ReLU 대비 학습 품질이 우수하여 LLaMA, Mistral 등에 채택되었습니다.

        참고: "GLU Variants Improve Transformer" (Shazeer, 2020)
        https://arxiv.org/abs/2002.05202

    Args:
        hidden_size: 모델의 hidden 차원 (입력/출력 크기).
            LLaMA-3 8B: 4096, LLaMA-3 70B: 8192, LLaMA-3 405B: 16384.
        intermediate_size: FFN 내부 확장 차원.
            LLaMA-3 8B: 14336 (≈ 3.5 × hidden_size, 8의 배수로 정렬).
        bias: bias 항 포함 여부 (기본값: False, LLaMA는 bias 미사용).
        dtype: 파라미터 dtype (기본값: 현재 default dtype).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # gate_proj + up_proj 를 하나의 weight로 융합
        # 출력: [num_tokens, 2 * intermediate_size]
        # 앞쪽 절반 = gate (SiLU 적용), 뒤쪽 절반 = up (게이트 곱셈 대상)
        self.gate_up_proj = MergedLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=bias,
            dtype=dtype,
        )
        # intermediate → hidden 차원 복원
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
        )
        # SiLU 활성화 + 게이트 곱셈 (CUDA 커널로 융합)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU MLP forward.

        Args:
            x: 입력 텐서 [num_tokens, hidden_size].

        Returns:
            출력 텐서 [num_tokens, hidden_size].
        """
        # 1) gate + up 을 한 번의 GEMM으로 계산
        x = self.gate_up_proj(x)    # [num_tokens, 2 * intermediate_size]
        # 2) SiLU(gate) ⊙ up  (CUDA 커널에서 split + silu + mul 융합)
        # 재할당으로 이전 [2*inter] 텐서 참조 해제 → GPU 메모리 즉시 반환 가능
        x = self.act_fn(x)          # [num_tokens, intermediate_size]
        # 3) 원래 hidden 차원으로 프로젝션
        x = self.down_proj(x)       # [num_tokens, hidden_size]
        return x


class LlamaAttention(nn.Module):
    """LLaMA의 Multi-Head Attention (GQA + RoPE + KV Cache).

    Grouped Query Attention (GQA):
        Q head 수와 KV head 수가 다를 수 있습니다.
        예) LLaMA-3 8B: num_heads=32, num_kv_heads=8 → 4개의 Q head가 1개의 KV head를 공유.
        메모리와 계산량을 줄이면서 MHA에 근접한 품질을 유지합니다.

        참고: "GQA: Training Generalized Multi-Query Transformer Models from
        Multi-Head Checkpoints" (Ainslie et al., 2023)
        https://arxiv.org/abs/2305.13245

    데이터 흐름::

        hidden_states: [N, hidden_size]
            ↓
        qkv_proj (MergedLinear)       →  [N, q_size + 2*kv_size]
            ↓
        split → Q [N, q_size], K [N, kv_size], V [N, kv_size]
            ↓
        RoPE (in-place)               →  Q, K에 위치 정보 주입
            ↓
        reshape to 3D                 →  Q [N, num_heads, head_dim]
                                         K [N, num_kv_heads, head_dim]
                                         V [N, num_kv_heads, head_dim]
            ↓
        KV Cache update (선택적)      →  K, V에 이전 토큰 히스토리 포함
            ↓
        Attention backend (SDPA)      →  [N, num_heads, head_dim]
            ↓
        reshape to 2D                 →  [N, q_size]
            ↓
        o_proj (Linear)               →  [N, hidden_size]

    Args:
        hidden_size: 모델의 hidden 차원 (입력/출력 크기).
            LLaMA-3 8B: 4096.
        num_heads: Query attention head 수.
            LLaMA-3 8B: 32.
        num_kv_heads: Key/Value attention head 수 (GQA).
            LLaMA-3 8B: 8 (num_heads의 1/4).
        head_dim: 각 head의 차원 크기 (기본값: 128).
        max_position_embeddings: RoPE 최대 위치 (기본값: 8192).
        rope_base: RoPE 주파수 기저값 (기본값: 500000.0).
            LLaMA-3는 500000.0을 사용 (LLaMA-2는 10000.0).
        bias: Linear에 bias 포함 여부 (기본값: False, LLaMA는 bias 미사용).
        dtype: 파라미터 dtype (기본값: 현재 default dtype).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        max_position_embeddings: int = 8192,
        rope_base: float = 500000.0,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Q의 총 차원 = num_heads * head_dim
        # KV의 총 차원 = num_kv_heads * head_dim (GQA이므로 Q보다 작을 수 있음)
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        # Q, K, V 프로젝션을 하나의 GEMM으로 융합
        # 출력: [N, q_size + kv_size + kv_size]
        # HuggingFace의 별도 q_proj, k_proj, v_proj를 weight_loader로 로딩 가능
        self.qkv_proj = MergedLinear(
            hidden_size,
            [self.q_size, self.kv_size, self.kv_size],
            bias=bias,
            dtype=dtype,
        )

        # Attention 출력을 hidden_size로 프로젝션
        self.o_proj = Linear(
            self.q_size, hidden_size, bias=bias, dtype=dtype,
        )

        # RoPE: cos/sin 캐시를 미리 생성하여 위치 인코딩 준비
        self.rotary_emb = RotaryEmbedding(
            head_size=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
            dtype=dtype or torch.bfloat16,
        )

        # Attention 연산 백엔드 (SDPA: PyTorch F.scaled_dot_product_attention)
        self.attn_backend = SDPAAttention()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """LlamaAttention forward.

        Args:
            positions: 각 토큰의 위치 인덱스 [num_tokens] (int64).
                Prefill: [0, 1, 2, ..., seq_len-1].
                Decode: [current_pos] (길이 1).
            hidden_states: 입력 텐서 [num_tokens, hidden_size].
            kv_cache: KV 캐시 (None이면 캐시 미사용, prefill-only).
            layer_idx: 현재 Transformer layer 인덱스 (KV cache 접근용).

        Returns:
            출력 텐서 [num_tokens, hidden_size].
        """
        num_tokens = hidden_states.shape[0]

        # 1) Q, K, V를 한 번의 GEMM으로 계산
        qkv = self.qkv_proj(hidden_states)  # [N, q_size + 2*kv_size]

        # 2) Q, K, V 분리
        # split은 마지막 차원을 기준으로 지정된 크기만큼 분할
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1,
        )

        # 3) RoPE CUDA 커널은 contiguous 메모리를 요구
        # split 결과가 비연속(non-contiguous)일 수 있으므로 명시적 보장
        q, k = q.contiguous(), k.contiguous()

        # 4) RoPE 적용 (in-place): Q, K에 위치 정보를 주입
        # 2D 텐서 [N, heads*head_dim] 상태에서 적용 (커널이 내부적으로 head 분리 처리)
        self.rotary_emb(positions, q, k)

        # 5) 3D로 reshape: attention backend가 [N, num_heads, head_dim] 형태를 기대
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.reshape(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.reshape(num_tokens, self.num_kv_heads, self.head_dim)

        # 6) KV Cache 업데이트 (있으면)
        # 캐시에 현재 K, V를 저장하고, 이전 토큰을 포함한 전체 히스토리를 반환받음
        # Prefill: 프롬프트 전체 저장, Decode: 새 1토큰 추가
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # 7) Causal mask 결정
        # Prefill (num_tokens > 1): 미래 토큰 참조 방지 → is_causal=True
        # Decode (num_tokens == 1): 새 토큰이 전체 캐시에 attend → is_causal=False
        is_causal = num_tokens > 1

        # 8) Attention 연산 (SDPA backend)
        # GQA expand, scaled dot-product, softmax 등을 내부적으로 처리
        attn_output = self.attn_backend.forward(
            q, k, v, is_causal=is_causal,
        )  # [N, num_heads, head_dim]

        # 9) 2D로 복원: o_proj에 전달하기 위해 head 차원을 합침
        attn_output = attn_output.reshape(num_tokens, self.q_size)

        # 10) Output projection: attention 결과를 hidden_size로 변환
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    """LLaMA Transformer Decoder Layer (Pre-Norm Residual 패턴).

    하나의 Decoder Layer는 Self-Attention과 MLP를 직렬로 연결하며,
    각 서브레이어 앞에 RMSNorm을 적용하는 Pre-Norm 구조입니다.

    Pre-Norm Residual 패턴::

        ┌─────────────────────────────────────────────────────┐
        │  Layer 0 (residual=None):                           │
        │    residual = hidden_states                         │
        │    hidden_states = RMSNorm(hidden_states)           │
        │                                                     │
        │  Layer 1+ (residual is not None):                   │
        │    residual += hidden_states   ← fused add          │
        │    hidden_states = RMSNorm(residual)                │
        └─────────────────────────────────────────────────────┘

    핵심 설계:
        vLLM은 residual을 layer 간에 명시적으로 전달합니다.
        일반적인 구현(residual = x + sublayer(x))과 달리,
        fused_add_rms_norm 커널이 "덧셈 + 정규화"를 한 번의 커널 호출로 처리하여
        메모리 읽기/쓰기를 절반으로 줄입니다.

        - 첫 번째 layer: embedding 출력이 그대로 residual이 되고, norm만 적용
        - 이후 layer: 이전 layer의 MLP 출력(hidden_states)을 residual에 더하고 norm 적용

        이 패턴 덕분에 residual stream은 gradient highway 역할을 하며,
        깊은 네트워크에서도 gradient가 안정적으로 전파됩니다.

    데이터 흐름::

        (hidden_states, residual)
            ↓
        input_layernorm (RMSNorm / Fused Add+RMSNorm)
            ↓
        self_attn (LlamaAttention)
            ↓
        post_attention_layernorm (Fused Add+RMSNorm)
            ↓
        mlp (LlamaMLP)
            ↓
        (hidden_states, residual)  →  다음 layer로 전달

    Args:
        hidden_size: 모델의 hidden 차원. LLaMA-3 8B: 4096.
        intermediate_size: MLP 내부 확장 차원. LLaMA-3 8B: 14336.
        num_heads: Query attention head 수. LLaMA-3 8B: 32.
        num_kv_heads: Key/Value attention head 수. LLaMA-3 8B: 8.
        head_dim: 각 head의 차원 크기 (기본값: 128).
        rms_norm_eps: RMSNorm epsilon (기본값: 1e-5).
        max_position_embeddings: RoPE 최대 위치 (기본값: 8192).
        rope_base: RoPE 주파수 기저값 (기본값: 500000.0).
        bias: Linear에 bias 포함 여부 (기본값: False).
        dtype: 파라미터 dtype (기본값: 현재 default dtype).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 8192,
        rope_base: float = 500000.0,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Self-Attention 서브레이어
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            bias=bias,
            dtype=dtype,
        )

        # Feed-Forward (SwiGLU MLP) 서브레이어
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
            dtype=dtype,
        )

        # Attention 전 RMSNorm
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, dtype=dtype,
        )

        # MLP 전 RMSNorm
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, dtype=dtype,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LlamaDecoderLayer forward.

        Args:
            positions: 각 토큰의 위치 인덱스 [num_tokens] (int64).
            hidden_states: 입력 텐서 [num_tokens, hidden_size].
                첫 번째 layer: embedding 출력.
                이후 layer: 이전 layer의 MLP 출력.
            residual: 잔차 텐서 [num_tokens, hidden_size] 또는 None.
                첫 번째 layer에서는 None, 이후 layer에서는 이전 layer가 전달.
            kv_cache: KV 캐시 (None이면 캐시 미사용).
            layer_idx: 현재 layer 인덱스 (KV cache 접근용).

        Returns:
            (hidden_states, residual) 튜플.
            다음 layer의 forward에 그대로 전달됩니다.
        """
        # === Self Attention ===
        #
        # Pre-Norm: attention 서브레이어 앞에 RMSNorm 적용
        #
        # [첫 번째 layer] residual=None:
        #   embedding 출력이 곧 residual stream의 시작점.
        #   hidden_states를 residual로 저장하고, 단순 RMSNorm만 적용.
        #
        # [이후 layer] residual is not None:
        #   fused_add_rms_norm 커널 사용:
        #     residual = residual + hidden_states  (이전 MLP 출력을 residual에 누적)
        #     hidden_states = RMSNorm(residual)    (정규화된 값을 attention 입력으로)
        #   → 덧셈과 정규화를 1회 커널 호출로 처리하여 메모리 대역폭 절약
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual,
            )

        hidden_states = self.self_attn(
            positions, hidden_states, kv_cache, layer_idx,
        )

        # === Feed-Forward (MLP) ===
        #
        # Post-Attention Norm: MLP 서브레이어 앞에 Fused Add+RMSNorm 적용
        #   residual = residual + hidden_states  (attention 출력을 residual에 누적)
        #   hidden_states = RMSNorm(residual)    (정규화된 값을 MLP 입력으로)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual,
        )

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Module):
    """LLaMA Transformer 본체: Embedding + DecoderLayer 스택 + 최종 RMSNorm.

    전체 구조::

        input_ids: [num_tokens] (int64)
            ↓
        embed_tokens (nn.Embedding)   →  [num_tokens, hidden_size]
            ↓
        DecoderLayer × num_hidden_layers
          (Pre-Norm Residual 패턴으로 hidden_states + residual을 layer 간 전달)
            ↓
        최종 RMSNorm                  →  [num_tokens, hidden_size]

    Pre-Norm Residual 전체 흐름:
        1) embed_tokens 출력이 첫 번째 layer에 (hidden_states=embedding, residual=None)으로 전달.
        2) 각 layer는 fused_add_rms_norm으로 residual 누적 + 정규화를 반복.
        3) 마지막 layer의 MLP 출력(hidden_states)은 아직 residual에 합산되지 않은 상태.
           → 최종 RMSNorm에서 hidden_states + residual 합산 후 정규화하여 완성.

        이 설계 덕분에 모든 서브레이어 출력이 residual stream에 누적되며,
        gradient가 embedding까지 직통으로 흐르는 "gradient highway"를 형성합니다.

    Args:
        vocab_size: 어휘 사전 크기. LLaMA-3 8B: 128256.
        hidden_size: 모델의 hidden 차원. LLaMA-3 8B: 4096.
        intermediate_size: MLP 내부 확장 차원. LLaMA-3 8B: 14336.
        num_hidden_layers: Decoder layer 수. LLaMA-3 8B: 32.
        num_heads: Query attention head 수. LLaMA-3 8B: 32.
        num_kv_heads: Key/Value attention head 수. LLaMA-3 8B: 8.
        head_dim: 각 head의 차원 크기 (기본값: 128).
        rms_norm_eps: RMSNorm epsilon (기본값: 1e-5).
        max_position_embeddings: RoPE 최대 위치 (기본값: 8192).
        rope_base: RoPE 주파수 기저값 (기본값: 500000.0).
        bias: Linear에 bias 포함 여부 (기본값: False).
        dtype: 파라미터 dtype (기본값: 현재 default dtype).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 8192,
        rope_base: float = 500000.0,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # 토큰 ID → hidden_size 차원의 밀집 벡터로 변환
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)

        # Transformer Decoder Layer 스택
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
                rope_base=rope_base,
                bias=bias,
                dtype=dtype,
            )
            for _ in range(num_hidden_layers)
        ])

        # 최종 RMSNorm: 마지막 layer의 잔차를 합산하고 정규화
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """LlamaModel forward.

        Args:
            input_ids: 토큰 ID 시퀀스 [num_tokens] (int64).
            positions: 각 토큰의 위치 인덱스 [num_tokens] (int64).
            kv_cache: KV 캐시 (None이면 캐시 미사용).

        Returns:
            최종 hidden states [num_tokens, hidden_size].
        """
        # 1) Embedding: 토큰 ID → 밀집 벡터
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        # 2) Decoder Layer 스택 순회
        # 각 layer는 (hidden_states, residual) 쌍을 받아 다음 layer로 전달.
        # Pre-Norm Residual 패턴에 의해 residual stream이 layer를 관통합니다.
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions, hidden_states, residual,
                kv_cache=kv_cache, layer_idx=idx,
            )

        # 3) 최종 RMSNorm
        # 마지막 layer의 MLP 출력(hidden_states)이 아직 residual에 합산되지 않았으므로,
        # fused_add_rms_norm으로 합산 + 정규화를 한 번에 처리합니다.
        hidden_states, _ = self.norm(hidden_states, residual)

        # 4) KV cache 포인터 전진
        # 모든 layer가 동일한 num_tokens만큼 KV를 추가했으므로,
        # advance()를 1회 호출하여 cache의 쓰기 위치를 일괄 갱신합니다.
        if kv_cache is not None:
            kv_cache.advance(input_ids.shape[0])

        return hidden_states


class LlamaForCausalLM(nn.Module):
    """LLaMA Causal Language Model: LlamaModel + LM Head.

    LlamaModel의 hidden states를 어휘 확률(logits)로 변환하는 최종 모델입니다.
    ``from_pretrained``로 HuggingFace 체크포인트를 직접 로딩할 수 있습니다.

    구조::

        input_ids → LlamaModel → hidden_states [N, hidden_size]
                                       ↓
                                  lm_head (Linear)
                                       ↓
                                  logits [N, vocab_size] (float32)

    logits을 float32로 변환하는 이유:
        모델 파라미터는 bf16/fp16으로 추론하지만, logits은 softmax/sampling에
        사용되므로 수치 안정성을 위해 fp32로 올립니다. bf16의 유효 자릿수(~3자리)로는
        vocab_size(128K)에 걸친 확률 분포를 정확히 표현할 수 없습니다.
        vLLM, HuggingFace 모두 logits을 fp32로 변환합니다.

    Args:
        vocab_size: 어휘 사전 크기. LLaMA-3 8B: 128256.
        hidden_size: 모델의 hidden 차원. LLaMA-3 8B: 4096.
        intermediate_size: MLP 내부 확장 차원. LLaMA-3 8B: 14336.
        num_hidden_layers: Decoder layer 수. LLaMA-3 8B: 32.
        num_heads: Query attention head 수. LLaMA-3 8B: 32.
        num_kv_heads: Key/Value attention head 수. LLaMA-3 8B: 8.
        head_dim: 각 head의 차원 크기 (기본값: 128).
        rms_norm_eps: RMSNorm epsilon (기본값: 1e-5).
        max_position_embeddings: RoPE 최대 위치 (기본값: 8192).
        rope_base: RoPE 주파수 기저값 (기본값: 500000.0).
        tie_word_embeddings: lm_head와 embed_tokens의 weight 공유 여부 (기본값: False).
        bias: Linear에 bias 포함 여부 (기본값: False).
        dtype: 파라미터 dtype (기본값: 현재 default dtype).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 8192,
        rope_base: float = 500000.0,
        tie_word_embeddings: bool = False,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Transformer 본체 (Embedding + Decoder Layers + 최종 Norm)
        self.model = LlamaModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            bias=bias,
            dtype=dtype,
        )

        # LM Head: hidden_size → vocab_size 프로젝션
        # 이 Linear의 출력이 각 토큰 위치에서의 다음 토큰 확률(logits)이 됩니다.
        self.lm_head = Linear(hidden_size, vocab_size, bias=False, dtype=dtype)

        self.tie_word_embeddings = tie_word_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """LlamaForCausalLM forward.

        Args:
            input_ids: 토큰 ID 시퀀스 [num_tokens] (int64).
            positions: 각 토큰의 위치 인덱스 [num_tokens] (int64).
            kv_cache: KV 캐시 (None이면 캐시 미사용).

        Returns:
            logits [num_tokens, vocab_size] (float32).
        """
        hidden_states = self.model(input_ids, positions, kv_cache)
        logits = self.lm_head(hidden_states)
        # logits은 항상 fp32: softmax/sampling의 수치 안정성을 위해
        return logits.float()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> "LlamaForCausalLM":
        """HuggingFace 체크포인트에서 모델을 로딩합니다.

        1) config.json에서 모델 하이퍼파라미터를 읽어 빈 모델을 생성합니다.
        2) safetensors 가중치를 stacked_params_mapping으로 융합 파라미터에 로딩합니다.
        3) tie_word_embeddings가 True이면 lm_head.weight를 embed_tokens.weight에 연결합니다.

        stacked_params_mapping 설명:
            HuggingFace는 q_proj, k_proj, v_proj (또는 gate_proj, up_proj)를
            별도 텐서로 저장하지만, 우리 모델은 qkv_proj (또는 gate_up_proj)로 융합합니다.
            이 매핑 테이블이 "HF 이름 → 융합 파라미터 이름 + shard 위치" 변환을 정의합니다.

            예) HF "model.layers.0.self_attn.q_proj.weight"
                → 우리 "model.layers.0.self_attn.qkv_proj.weight" (shard_id="q")

        Args:
            model_dir: HuggingFace 체크포인트 디렉토리 (config.json + *.safetensors).
            dtype: 모델 파라미터 dtype (기본값: torch.bfloat16).
            device: 모델을 로딩할 디바이스 (기본값: "cuda").

        Returns:
            가중치가 로딩된 LlamaForCausalLM 인스턴스.
        """
        # 1) config.json 읽기
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # 2) 빈 모델 생성 후 device로 이동
        model = cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config.get(
                "num_key_value_heads", config["num_attention_heads"],
            ),
            head_dim=config.get(
                "head_dim",
                config["hidden_size"] // config["num_attention_heads"],
            ),
            rms_norm_eps=config.get("rms_norm_eps", 1e-5),
            max_position_embeddings=config.get("max_position_embeddings", 8192),
            rope_base=config.get("rope_theta", 500000.0),
            tie_word_embeddings=config.get("tie_word_embeddings", False),
            dtype=dtype,
        ).to(device)

        # 3) 가중치 로딩
        # 여기서 import하여 순환 의존성 방지 (loader가 nn.Module을 사용하므로)
        from lightvllm.layers.linear import SHARD_GATE, SHARD_UP
        from lightvllm.models.loader import load_weights

        # HF 별도 가중치 → 융합 파라미터 매핑 테이블
        # (융합 파라미터 접미사, HF 접미사, shard ID)
        stacked_params_mapping = [
            # Attention QKV 융합: q_proj + k_proj + v_proj → qkv_proj
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # MLP gate+up 융합: gate_proj + up_proj → gate_up_proj
            (".gate_up_proj", ".gate_proj", SHARD_GATE),
            (".gate_up_proj", ".up_proj", SHARD_UP),
        ]

        load_weights(model, model_dir, stacked_params_mapping)

        # 4) tie_word_embeddings: lm_head가 체크포인트에 없을 때 embedding weight 공유
        # 일부 소형 모델은 파라미터 수를 줄이기 위해 embed_tokens와 lm_head의
        # weight를 공유합니다 (vocab_size × hidden_size 만큼 절약).
        if config.get("tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight

        return model
