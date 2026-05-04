"""
Naive PyTorch Attention 구현 (교육용).

수동으로 softmax(QK^T / sqrt(d_k)) @ V를 계산합니다.
GQA (Grouped Query Attention), causal masking, fp32 softmax를 지원합니다.

이 구현은 O(N^2) 메모리를 사용하는 "standard attention"입니다.
Phase 2에서 FlashAttention (O(N) 메모리)으로 최적화합니다.

학습 포인트:
    - Q @ K^T가 "유사도 행렬"인 이유 (내적 = 벡터 간 코사인 유사도 * 크기)
    - Causal masking: 미래 토큰 참조 방지 (autoregressive decoding 전제)
    - GQA: KV heads < Q heads일 때 메모리 절감
      (LLaMA-3.2-3B: 8 KV heads, 24 Q heads → ratio=3)
    - Softmax fp32: half precision에서 overflow 방지
      (exp(x)에서 x가 크면 inf, HuggingFace LLaMA도 fp32 softmax 사용)

vLLM 참조: vLLM에는 naive 구현이 없음 (FlashAttention/SDPA 직접 사용)
"""

import torch

from lightvllm.attention.backends.base import AttentionBackend


class NaiveAttention(AttentionBackend):
    """수동 scaled dot-product attention.

    Step-by-step으로 attention을 계산하여 학습용으로 명확하게 작성:
    1. Reshape: batched matmul을 위해 [1, heads, seq, dim] 형태로
    2. GQA expand: K/V를 num_kv_heads → num_heads로 확장
    3. Attention scores: Q @ K^T / sqrt(d_k)
    4. Causal mask: 미래 위치에 -inf
    5. Softmax in fp32: 수치 안정성
    6. Weighted sum: scores @ V
    7. Reshape back: [num_tokens, num_heads, head_dim]
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        num_tokens, num_heads, head_dim = query.shape
        seq_len, num_kv_heads, _ = key.shape
        scale = head_dim ** -0.5
        original_dtype = query.dtype

        # --- Step 1: Reshape for batched matmul ---
        # PyTorch batched matmul은 [batch, heads, seq_len, head_dim]을 기대.
        # batch=1 (Phase 1은 단일 시퀀스).
        q = query.transpose(0, 1).unsqueeze(0)  # [1, num_heads, num_tokens, head_dim]
        k = key.transpose(0, 1).unsqueeze(0)    # [1, num_kv_heads, seq_len, head_dim]
        v = value.transpose(0, 1).unsqueeze(0)  # [1, num_kv_heads, seq_len, head_dim]

        # --- Step 2: GQA expand ---
        # num_kv_heads < num_heads이면, 각 KV head가 여러 Q head를 담당.
        # 예) LLaMA-3.2-3B: 24 Q heads, 8 KV heads → ratio=3
        #     K/V [1, 8, seq, d] → [1, 24, seq, d]
        #
        # expand()는 메모리를 복사하지 않고 stride=0인 view를 반환.
        # reshape()에서 contiguous 복사가 발생하지만, Phase 1에서는 무관.
        # Phase 2의 FlashAttention은 GQA를 네이티브로 지원하여 확장 불필요.
        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            # [1, kv_heads, seq, d] → [1, kv_heads, 1, seq, d]
            #                        → [1, kv_heads, groups, seq, d]
            #                        → [1, num_heads, seq, d]
            k = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
            k = k.reshape(1, num_heads, seq_len, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
            v = v.reshape(1, num_heads, seq_len, head_dim)

        # --- Step 3: Attention scores ---
        # Q @ K^T: 각 query 토큰이 모든 key 토큰과 얼마나 유사한지 계산.
        # 내적이 크면 두 벡터가 같은 방향 → 높은 attention weight.
        # scale = 1/sqrt(d_k): head_dim이 클수록 내적 값이 커지므로 정규화.
        #
        # [1, heads, num_tokens, d] @ [1, heads, d, seq_len]
        # = [1, heads, num_tokens, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # --- Step 4: Causal mask ---
        # Autoregressive 생성에서 미래 토큰을 참조하면 안 됨.
        # 미래 위치에 -inf를 넣으면 softmax 후 0이 됨 (exp(-inf) = 0).
        #
        # Prefill: num_tokens == seq_len, 정사각 causal mask
        # Decode with KV cache: is_causal=False이므로 이 블록 skip
        #   (새 토큰 1개가 전체 캐시에 attend)
        if is_causal:
            # diagonal=1: 주대각선 위(미래)가 True
            mask = torch.triu(
                torch.ones(
                    num_tokens, seq_len,
                    device=query.device, dtype=torch.bool,
                ),
                diagonal=1,
            )
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # --- Step 5: Softmax in fp32 ---
        # fp16/bf16에서 softmax를 직접 하면 exp(x)가 overflow할 수 있음.
        # HuggingFace LLaMA도 fp32로 softmax 후 원래 dtype으로 복원.
        scores = torch.softmax(scores.float(), dim=-1).to(original_dtype)

        # --- Step 6: Weighted sum ---
        # attention weight와 value의 가중합.
        # weight가 큰 토큰의 value가 출력에 더 많이 반영됨.
        #
        # [1, heads, num_tokens, seq_len] @ [1, heads, seq_len, d]
        # = [1, heads, num_tokens, d]
        output = torch.matmul(scores, v)

        # --- Step 7: Reshape back ---
        # [1, num_heads, num_tokens, head_dim] → [num_tokens, num_heads, head_dim]
        output = output.squeeze(0).transpose(0, 1)

        return output
