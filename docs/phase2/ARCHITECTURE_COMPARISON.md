# LightvLLM vs vLLM: Attention + KV Cache 아키텍처 비교 분석

Phase 1 완성 후, 우리 LlamaAttention 구현과 vLLM의 구현 차이를 분석한다.
**목표**: 단순 학습용이 아닌, 실제 사용 가능한 프로덕션 코드로 발전시키기 위한 기반 마련.
장기적으로 TensorEngine이라는 별도 프로세스에서 weight/KV cache 메모리를 외부 핸들로 관리할 계획.

**핵심 발견**: Paged Attention은 KV cache 아키텍처 변경과 불가분하게 결부되어 있어,
Phase 2의 "Attention 리팩토링"과 "Paged Attention 구현"을 하나의 작업으로 통합해야 한다.

---

## 1. 현재 차이점 요약

### LlamaAttention.forward() 시그니처

| | LightvLLM | vLLM |
|---|---|---|
| **시그니처** | `forward(positions, hidden_states, kv_cache, layer_idx)` | `forward(positions, hidden_states)` |
| **KV cache 접근** | 명시적 인자 | `Attention` 레이어의 인스턴스 속성 (`self.kv_cache`) |
| **메타데이터** | `is_causal = (num_tokens > 1)` 직접 판단 | `ForwardContext`에서 `attn_metadata` 가져옴 |
| **캐시 쓰기** | `kv_cache.update(layer_idx, k, v)` — 모델 코드에서 | `reshape_and_cache` 커널 — backend 내부에서 |
| **포인터 관리** | `kv_cache.advance()` — 모델 코드에서 | `slot_mapping` — scheduler가 계산 |

### KV Cache 구조

| | LightvLLM | vLLM |
|---|---|---|
| **추상화** | `KVCache` 클래스 (write pointer 기반) | raw `torch.Tensor` (metadata 기반) |
| **Shape** | `[num_layers, 2, max_seq_len, heads, dim]` contiguous | `[2, num_blocks, block_size, heads, dim]` paged |
| **위치 추적** | `_seq_len` (단일 write pointer) | `slot_mapping` (토큰별 물리 슬롯 ID) |
| **할당** | 시퀀스 시작 시 max_seq_len 전체 할당 | 블록 단위 동적 할당/해제 |
| **배치** | 단일 시퀀스만 | 수백~수천 시퀀스 동시 |

---

## 2. vLLM의 Paged Attention 전체 흐름

### 2.1 데이터 구조

**Block Table** — 시퀀스 → 물리 블록 매핑:
```
block_table[req_idx, block_num] = physical_block_id
예) Request 0: [0, 5, 10]  (block_size=16 → 48토큰 수용)
    Request 1: [1, 6, 11]
```

**Slot Mapping** — 토큰 → 캐시 슬롯 매핑:
```
slot_mapping[token_idx] = block_id * block_size + (pos % block_size)
예) 토큰이 position 35에 있고, block_table[req, 2]=10이면:
    slot = 10 * 16 + (35 % 16) = 163
```

**AttentionMetadata** — backend에 전달되는 메타데이터:
```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor     # [batch+1] — 배치 내 각 시퀀스의 시작 위치
    seq_lens: torch.Tensor            # [batch] — 각 시퀀스의 전체 길이 (context + new)
    block_table_tensor: torch.Tensor  # [batch, max_blocks] — 물리 블록 ID
    slot_mapping: torch.Tensor        # [num_tokens] — 토큰 → 캐시 슬롯
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
```

### 2.2 캐시 쓰기: `reshape_and_cache` CUDA 커널

**파일**: `vLLM/csrc/cache_kernels.cu`

```cuda
__global__ void reshape_and_cache_flash_kernel(...) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];  // 핵심: 토큰 → 슬롯

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    // K/V를 해당 블록의 해당 오프셋에 복사
    key_cache[block_idx][block_offset] = key[token_idx];
    value_cache[block_idx][block_offset] = value[token_idx];
}
```

vs 우리: `self._cache[layer_idx, 0, start:end] = key` (단순 연속 복사)

### 2.3 캐시 읽기: FlashAttention이 block_table 사용

```python
flash_attn_varlen_func(
    q=query,
    k_cache=key_cache,      # [num_blocks, block_size, heads, dim]
    v_cache=value_cache,
    block_table=block_table, # [batch, max_blocks] — FA가 블록을 직접 참조
    cu_seqlens_q=query_start_loc,
    seqused_k=seq_lens,
    ...
)
```

FlashAttention 커널이 `block_table`을 사용하여 비연속적으로 흩어진 블록들에서 K/V를 직접 읽는다.
우리의 SDPA는 연속 메모리의 K/V를 받으므로 이 기능이 없다.

### 2.4 전체 실행 흐름

```
1. SCHEDULER: 어떤 시퀀스들을 이번 step에서 처리할지 결정
   → SchedulerOutput (num_scheduled_tokens per request)

2. KV CACHE MANAGER: 새 토큰을 위한 물리 블록 할당
   → block_ids → BlockTable에 기록

3. BLOCK TABLE: slot_mapping 계산 (Triton 커널)
   → slot_mapping[token] = block_id * block_size + offset

4. ATTENTION METADATA 생성
   → CommonAttentionMetadata(query_start_loc, seq_lens, block_table, slot_mapping)

5. ForwardContext에 metadata 설정
   → set_forward_context(attn_metadata)

6. MODEL FORWARD (KV cache 인자 없음!)
   → LlamaAttention.forward(positions, hidden_states)
     → self.attn(q, k, v)  # Attention 레이어 호출

7. ATTENTION LAYER 내부
   → kv_cache = self.kv_cache[virtual_engine]      # 인스턴스 속성
   → metadata = get_forward_context().attn_metadata  # 전역 컨텍스트
   → self.impl.forward(q, k, v, kv_cache, metadata)

8. BACKEND IMPL (FlashAttention)
   → reshape_and_cache_flash(k, v, cache, slot_mapping)  # 캐시 쓰기
   → flash_attn_varlen_func(q, cache, block_table, ...)  # attention 연산
```

---

## 3. positions 텐서: 왜 외부에서 전달하는가

### vLLM: ModelRunner가 positions를 계산

**파일**: `vLLM/vllm/v1/worker/gpu_model_runner.py`

```python
# Scheduler가 알고 있는 per-sequence offset + local index
positions[batch_offset:batch_offset+seq_len] = (
    num_computed_tokens[req_idx] + np.arange(seq_len)
)
```

**Continuous batching에서의 예시**:
```
Seq A: 50토큰 이미 처리, 새 3토큰 → positions = [50, 51, 52]
Seq B: 0토큰, 새 3토큰            → positions = [0, 1, 2]
Batch positions: [50, 51, 52, 0, 1, 2]  (flat)
```

**왜 모델이 아닌 외부에서 계산하는가?**
1. **Scheduler가 시퀀스 상태를 소유**: `num_computed_tokens`는 scheduler가 추적
2. **모델은 배치 구조를 몰라야 함**: flat `[N, hidden]` 텐서만 받음
3. **Dynamic batching**: 시퀀스마다 다른 offset (prefill + decode 혼합)

### 우리 구현: 이미 올바른 구조

우리도 positions를 외부에서 받는다:
```python
LlamaForCausalLM.forward(input_ids, positions, kv_cache)
→ self.rotary_emb(positions, q, k)  # positions로 cos/sin 캐시 인덱스
```

RoPE는 positions를 cos/sin 캐시의 **인덱스**로만 사용 (`index_select`).
모델이 positions를 직접 계산할 이유가 없다 — scheduler가 이미 알고 있다.

**변경 불필요**: positions 전달 방식은 이미 vLLM과 동일한 패턴.

---

## 4. Attention 레이어 아키텍처 비교

### vLLM의 Attention (통합 레이어)

**파일**: `vLLM/vllm/attention/layer.py`

```python
class Attention(nn.Module):
    """
    1. Store K/V in KV cache (via reshape_and_cache)
    2. Perform attention (via FlashAttention/SDPA)
    3. Return output
    """
    def __init__(self, num_heads, head_size, scale, ...):
        self.kv_cache = [torch.tensor([])]  # bind_kv_cache()로 교체됨
        self.impl = backend.get_impl_cls()(...)

    def forward(self, query, key, value):  # KV cache 인자 없음
        kv_cache = self.kv_cache[virtual_engine]
        metadata = get_forward_context().attn_metadata
        return self.impl.forward(self, q, k, v, kv_cache, metadata)
```

**핵심**: `Attention`은 nn.Module이면서 **KV cache 저장소 + backend 디스패처**.
모델 코드(LlamaAttention)는 `self.attn = Attention(...)`으로 사용하며, 캐시를 전혀 모른다.

### 우리: AttentionBackend = 상태 없는 순수 연산

```python
class AttentionBackend(ABC):
    def forward(self, query, key, value, is_causal=True):
        ...  # 캐시와 무관한 순수 연산
```

캐시 업데이트는 LlamaAttention이 직접 수행. Backend는 이미 합쳐진 K/V만 받는다.

### 아키텍처 도식

```
vLLM:
  LlamaAttention → Attention(nn.Module) → AttentionImpl
                    ↑ self.kv_cache         ↑ reshape_and_cache + flash_attn
                    ↑ ForwardContext           캐시 R/W + 연산

LightvLLM (현재):
  LlamaAttention → KVCache.update() → AttentionBackend.forward()
  ↑ 캐시 관리 직접 수행                ↑ 순수 연산만
```

---

## 5. Phase 2 통합 구현 방향

Paged Attention + Attention 리팩토링은 하나의 작업 단위로 묶어야 한다.

### 5.1 구현 순서 (의존성 순)

```
Step 1: ForwardContext + AttentionMetadata
Step 2: Block Allocator + Block Table
Step 3: reshape_and_cache CUDA 커널
Step 4: Attention 통합 레이어 (캐시 관리 캡슐화)
Step 5: FlashAttention backend (block_table 지원)
Step 6: LlamaAttention 리팩토링 (kv_cache/layer_idx 제거)
Step 7: ModelRunner (scheduler → metadata → model.forward)
```

### 5.2 TensorEngine과의 연결

vLLM의 `bind_kv_cache()` 패턴은 TensorEngine 구조와 자연스럽게 연결된다:
- 외부 프로세스(TensorEngine)가 GPU 메모리 할당 → KV cache 텐서 생성
- `bind_kv_cache(tensor)`로 모델의 Attention 레이어에 주입
- 모델은 자신이 사용하는 KV cache의 lifecycle을 모름 → 메모리 관리 완전 분리

---

## 6. Phase 1 → Phase 2 전환 요약

| 구성요소 | Phase 1 (현재) | Phase 2 (목표) |
|----------|---------------|----------------|
| KV cache 구조 | Contiguous `[layers, 2, seq, h, d]` | Paged `[2, blocks, block_size, h, d]` |
| 캐시 관리 주체 | 모델 코드 (LlamaAttention) | Attention 레이어 (backend 내부) |
| 위치 추적 | write pointer (`_seq_len`) | slot_mapping (토큰별) |
| 모델 시그니처 | `forward(..., kv_cache, layer_idx)` | `forward(positions, hidden_states)` |
| metadata 전달 | 없음 (is_causal만) | ForwardContext + AttentionMetadata |
| 블록 할당 | N/A (전체 사전 할당) | Block pool + block table |
| 캐시 쓰기 커널 | 텐서 슬라이싱 | `reshape_and_cache` CUDA 커널 |
| Attention backend | SDPA (연속 K/V) | FlashAttention (block_table) |
| 배치 지원 | 단일 시퀀스 | Continuous batching |

---

## 참조 파일

### vLLM 핵심 파일
- `vLLM/vllm/model_executor/models/llama.py` — LlamaAttention (forward에 KV cache 없음)
- `vLLM/vllm/attention/layer.py` — Attention 통합 레이어 (self.kv_cache + ForwardContext)
- `vLLM/vllm/attention/backends/abstract.py` — AttentionBackend + AttentionImpl ABC
- `vLLM/vllm/v1/attention/backends/flash_attn.py` — FlashAttention backend (reshape_and_cache 사용)
- `vLLM/vllm/v1/worker/gpu_model_runner.py` — positions 계산, metadata 생성
- `vLLM/vllm/v1/worker/gpu/block_table.py` — slot_mapping 계산 (Triton 커널)
- `vLLM/vllm/v1/core/kv_cache_manager.py` — 블록 할당/해제
- `vLLM/vllm/forward_context.py` — ForwardContext 정의
- `vLLM/csrc/cache_kernels.cu` — reshape_and_cache CUDA 커널

### LightvLLM 핵심 파일
- `lightvllm/models/llama.py` — LlamaAttention (kv_cache 명시적 전달)
- `lightvllm/attention/kv_cache.py` — KVCache 클래스 (contiguous, write pointer)
- `lightvllm/attention/backends/base.py` — AttentionBackend ABC (상태 없는 순수 연산)
- `lightvllm/attention/backends/sdpa.py` — SDPA backend
- `lightvllm/attention/layer.py` — 빈 스텁 (Phase 2에서 구현)
- `lightvllm/layers/rotary_embedding.py` — RoPE (positions를 외부에서 받음, 올바른 구조)
