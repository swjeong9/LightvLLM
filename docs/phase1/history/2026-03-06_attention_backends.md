# 2026-03-06: Attention Backend + KV Cache 코드 작성

## 작업 개요

Phase 1.3의 Attention 인프라로, **AttentionBackend ABC**, **NaiveAttention**, **SDPAAttention**,
**KVCache**, 그리고 포괄적 테스트 스위트를 작성하였다.
코드 작성은 완료되었으나, 학습 및 테스트 검증이 필요한 상태이다.

**`lightvllm/attention/layer.py` (LlamaAttention)**은 빈 스텁으로, 아직 구현되지 않았다.

---

## 배경

### Scaled Dot-Product Attention

Transformer의 핵심 연산. 각 query 토큰이 모든 key 토큰과의 유사도를 계산하고,
유사도에 비례하여 value를 가중합한다:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

- `QK^T`: 유사도 행렬 (내적 = 코사인 유사도 * 크기)
- `sqrt(d_k)`: head_dim이 클수록 내적이 커지므로 정규화
- `softmax`: 유사도를 확률 분포로 변환

### GQA (Grouped Query Attention)

LLaMA-3.2-3B: Q heads=24, KV heads=8 (ratio=3).
KV heads < Q heads일 때, 각 KV head를 여러 Q head가 공유한다.
메모리 절감 효과: KV cache 크기가 ratio만큼 줄어든다.

### Causal Masking

Autoregressive 생성에서 미래 토큰 참조를 방지한다.
미래 위치에 -inf를 넣으면 softmax 후 0이 된다 (exp(-inf) = 0).

- Prefill: 정사각 causal mask 적용
- Decode with KV cache: is_causal=False (새 토큰 1개가 전체 캐시에 attend)

### KV Cache

Autoregressive 생성에서 이전 토큰의 K/V를 재계산하지 않고 캐시에서 재사용한다.
Q는 현재 토큰만 필요하므로 캐시하지 않는다 (KV Cache, not QKV Cache).

- Prefill: 프롬프트 전체의 K/V를 캐시에 저장
- Decode: 1토큰의 K/V를 추가, 전체 히스토리 반환

이 구현은 contiguous 버퍼 방식 (Phase 2에서 Paged Attention으로 발전).

### 왜 Phase 1에서는 CUDA 커널 없이 Python으로 구현하는가?

Prefill 단계의 attention은 큰 행렬 곱셈(GEMM)이므로 compute-bound이다.
`torch.matmul`이 내부적으로 cuBLAS를 호출하므로 커스텀 커널의 이점이 없다.

하지만 **Decode 단계는 memory-bound**이다.
Q가 1토큰이므로 `[1, d] × [seq_len, d]^T`는 GEMV(벡터-행렬 곱)에 가까우며,
연산량은 적지만 KV cache 전체를 글로벌 메모리에서 읽어야 한다.
이것이 Phase 2에서 **Paged Attention CUDA 커널**(`paged_attention_v1.cu` 등)이 필요한 이유이다.

Phase 1은 정확성 검증이 목표이므로 Python(PyTorch) 조합으로 충분하다.

---

## 구현 상세

### AttentionBackend ABC (`lightvllm/attention/backends/base.py`, 62줄)

```python
class AttentionBackend(ABC):
    @abstractmethod
    def forward(self, query, key, value, is_causal=True) -> torch.Tensor:
        # query:  [num_tokens, num_heads, head_dim]
        # key:    [seq_len, num_kv_heads, head_dim]
        # value:  [seq_len, num_kv_heads, head_dim]
        # Returns: [num_tokens, num_heads, head_dim]
```

### NaiveAttention (`lightvllm/attention/backends/naive.py`, 121줄)

수동 7단계 구현:
1. Reshape: `[1, heads, seq, dim]` for batched matmul
2. GQA expand: `unsqueeze(2).expand().reshape()` 패턴
3. Attention scores: `Q @ K^T * scale`
4. Causal mask: `torch.triu(ones, diagonal=1)` → `masked_fill_(-inf)`
5. Softmax in fp32: 수치 안정성 (HuggingFace 동일 패턴)
6. Weighted sum: `scores @ V`
7. Reshape back: `[num_tokens, num_heads, head_dim]`

### SDPAAttention (`lightvllm/attention/backends/sdpa.py`, 83줄)

PyTorch 2.0+ `F.scaled_dot_product_attention` 래핑.
GQA expand 후 `[1, num_heads, seq_len, head_dim]` shape 변환.
내부적으로 FlashAttention 또는 Memory-Efficient Attention 자동 선택.

### KVCache (`lightvllm/attention/kv_cache.py`, 163줄)

```python
class KVCache:
    # Buffer: [num_layers, 2, max_seq_len, num_kv_heads, head_dim]
    # dim 1: index 0 = key, index 1 = value
    def update(layer_idx, key, value) -> (all_keys, all_values)  # view 반환
    def advance(num_tokens) -> None    # step당 1회만 호출
    def reset() -> None                # 새 시퀀스를 위한 초기화
```

핵심 설계:
- `update()`와 `advance()`를 분리: 모든 layer의 update 완료 후 advance 1회
- 반환값은 버퍼의 view (메모리 복사 없음)

---

## 테스트 스위트 (`tests/attention/test_attention.py`, 611줄)

**작성 완료, 실행/검증 필요.**

| 테스트 클래스 | 테스트 수 | 내용 |
|--------------|-----------|------|
| `TestNaiveAttention` | 7 | 기본 MHA, GQA, causal mask, decode, softmax fp32, dtype별, Llama-3.2-3B 차원 |
| `TestSDPAAttention` | 4 | Naive와 교차 비교, GQA, decode, dtype별 |
| `TestKVCache` | 6 | prefill, incremental decode, multi-layer, reset, overflow, advance 분리 |
| `TestAttentionWithKVCache` | 2 | Prefill→Decode 전체 흐름, Naive vs SDPA + cache 교차 검증 |

fp32 참조 구현(`attention_reference()`)으로 정확성 검증.
허용 오차: float32(1e-4), float16(1e-2), bfloat16(2e-2) — softmax + matmul 누적 고려.

---

## 생성된 파일

| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `lightvllm/attention/backends/base.py` | 62 | AttentionBackend ABC |
| `lightvllm/attention/backends/naive.py` | 121 | 수동 attention (교육용) |
| `lightvllm/attention/backends/sdpa.py` | 83 | PyTorch SDPA 래퍼 |
| `lightvllm/attention/kv_cache.py` | 163 | Contiguous KV cache |
| `tests/attention/test_attention.py` | 611 | 포괄적 테스트 스위트 |
| `lightvllm/attention/layer.py` | 1 | 빈 스텁 (docstring만) |

---

## 학습 상태

- **전체**: 코드 작성 완료, **학습 및 테스트 검증 필요**
- **LlamaAttention** (`attention/layer.py`): **미구현** (QKV 프로젝션 + RoPE + Attention + Output 통합 필요)
- 다음 단계: 코드 리뷰 → 테스트 실행 (`uv run pytest tests/attention/test_attention.py -v`) → LlamaAttention 구현
