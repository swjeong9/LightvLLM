# LightvLLM 학습 및 구현 로드맵

## 프로젝트 목표

vLLM을 기반으로 한 LLM Serving Engine을 단계적으로 학습하고 직접 구현한다.

- **타겟 모델**: LLaMA 계열 (추후 확장 가능한 추상화 유지)
- **분산 추론**: 다중 NVIDIA GPU 지원, Tensor/Pipeline Parallelism
- **오케스트레이션**: Ray 이해 후 Kubernetes 기반으로 전환
- **향후 확장**: Heterogeneous GPU 지원 고려

---

## 전체 로드맵

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LightvLLM 학습 로드맵                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: 기초           Phase 2: 핵심             Phase 3: 모델            │
│  ┌───────────────┐       ┌───────────────┐        ┌───────────────┐        │
│  │ CUDA 기초     │       │ Attention     │        │ Model Loader  │        │
│  │ PyTorch Ext   │  ──▶  │ KV Cache      │  ──▶   │ Layers        │        │
│  │ 기본 커널     │       │ Paged Attn    │        │ LLaMA 구현    │        │
│  └───────────────┘       └───────────────┘        └───────────────┘        │
│         │                       │                        │                 │
│         ▼                       ▼                        ▼                 │
│  Phase 4: 스케줄링       Phase 5: 분산              Phase 6: 서버           │
│  ┌───────────────┐       ┌───────────────┐        ┌───────────────┐        │
│  │ Scheduler     │       │ Ray 분석      │        │ API Server    │        │
│  │ Batching      │  ──▶  │ TP/PP 구현    │  ──▶   │ OpenAI 호환   │        │
│  │ Engine        │       │ K8s 전환      │        │ Streaming     │        │
│  └───────────────┘       └───────────────┘        └───────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: CUDA 및 PyTorch Extension 기초

### 1.1 CUDA 프로그래밍 기초

**목표:** CUDA 커널 작성 및 PyTorch와의 연동 이해

**학습 내용:**
- CUDA 메모리 모델 (Global, Shared, Register)
- Thread/Block/Grid 구조
- Memory Coalescing, Bank Conflict
- PyTorch C++ Extension (`torch.utils.cpp_extension`)

**vLLM 참고 파일:**
- `vLLM/csrc/torch_bindings.cpp` - PyTorch 바인딩
- `vLLM/csrc/core/scalar_type.hpp` - 타입 시스템

**구현 과제:**
1. 간단한 벡터 연산 CUDA 커널 작성
2. PyTorch Extension 빌드 및 Python 호출
3. PyTorch 내장 연산과 성능 비교

### 1.2 LLM 기본 연산 커널

**목표:** Transformer에서 사용되는 기본 연산 구현

**학습 내용:**
- RMSNorm (LLaMA에서 사용)
- SiLU/GELU Activation
- Fused 커널의 이점

**vLLM 참고 파일:**
- `vLLM/csrc/layernorm_kernels.cu`
- `vLLM/csrc/activation_kernels.cu`

**구현 과제:**
1. RMSNorm 커널 구현 (forward)
2. Fused SiLU-Gate 커널 구현
3. 수치 정확성 검증

---

## Phase 2: Attention 및 KV Cache

### 2.1 Self-Attention 이해

**목표:** Transformer Attention의 동작 원리 이해

**학습 내용:**
- Scaled Dot-Product Attention 수식
- Multi-Head Attention 구조
- Causal Masking

**vLLM 참고 파일:**
- `vLLM/vllm/attention/layer.py`
- `vLLM/vllm/attention/backends/`

**구현 과제:**
1. 순수 PyTorch Attention 구현
2. Multi-Head Attention 구현
3. Flash Attention 논문 분석

### 2.2 KV Cache 메커니즘

**목표:** Autoregressive 생성에서 KV Cache의 역할 이해

**학습 내용:**
- Prefill vs Decode 단계
- Cache 메모리 관리
- Cache Miss/Hit 영향

**vLLM 참고 파일:**
- `vLLM/vllm/config/cache.py`
- `vLLM/csrc/cache_kernels.cu`

**구현 과제:**
1. 기본 KV Cache 구조 구현
2. Cache 할당/재사용 로직 구현
3. 메모리 사용량 분석

### 2.3 Flash Attention 심화 학습

**목표:** Flash Attention의 원리 이해, 사용법 습득, 가능하다면 직접 구현

#### 2.3.1 Flash Attention 이론

**학습 내용:**
- Standard Attention의 메모리 복잡도 문제 (O(N²) 메모리)
- Tiling 기반 접근법 (Block-wise 계산)
- Online Softmax 알고리즘 (numerically stable)
- IO-Aware 알고리즘 설계 원칙
- Flash Attention 2 vs 3 차이점

**핵심 논문:**
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023)

**수식 이해:**
```
Standard: O = softmax(QK^T / √d) V
- QK^T 전체를 메모리에 저장 → O(N²) 메모리

Flash:
- Q, K, V를 블록으로 분할
- 각 블록별로 부분 softmax 계산
- Online algorithm으로 점진적 결합
- O(N) 메모리로 exact attention 계산
```

#### 2.3.2 vLLM에서의 Flash Attention 통합 분석

**vLLM 참고 파일:**
- `vLLM/vllm/attention/utils/fa_utils.py` - FA 유틸리티 및 버전 관리
- `vLLM/vllm/v1/attention/backends/flash_attn.py` - FA 백엔드 구현
- `vLLM/vllm/attention/backends/registry.py` - 백엔드 레지스트리
- `vLLM/vllm/platforms/cuda.py` - 백엔드 선택 로직
- `vLLM/csrc/cache_kernels.cu` - `reshape_and_cache_flash` 커널

**핵심 함수 분석:**
```python
# 메인 Attention 호출
flash_attn_varlen_func(
    q=query,                    # [num_tokens, num_heads, head_dim]
    k=key_cache,                # Paged KV cache
    v=value_cache,
    cu_seqlens_q=query_start_loc,  # Cumulative sequence lengths
    max_seqlen_q=max_query_len,
    seqused_k=seq_lens,
    max_seqlen_k=max_seq_len,
    softmax_scale=scale,
    causal=True,
    block_table=block_table,    # Paged attention용
    scheduler_metadata=...,      # FA3 AOT scheduling
)

# KV Cache 저장 (CUDA 커널)
reshape_and_cache_flash(
    key, value,
    key_cache, value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale, v_scale,  # FP8 양자화용
)
```

**백엔드 선택 우선순위 (CUDA):**
```
Non-MLA 모델:
1. FLASH_ATTN (기본)
2. FLASHINFER
3. TRITON_ATTN
4. FLEX_ATTENTION

MLA 모델 (Blackwell):
1. CUTLASS_MLA
2. FLASHINFER_MLA
3. FLASH_ATTN_MLA
```

#### 2.3.3 Flash Attention 사용법 실습

**구현 과제:**
1. flash-attn 패키지 설치 및 기본 사용법 익히기
2. `flash_attn_varlen_func` 직접 호출하여 Attention 계산
3. vLLM 스타일로 백엔드 추상화 레이어 구현
4. Paged KV Cache와 Flash Attention 연동

**실습 코드 예시:**
```python
from flash_attn import flash_attn_varlen_func

# Variable length attention
output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,      # [batch_size + 1]
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=1.0 / math.sqrt(head_dim),
    causal=True,
)
```

#### 2.3.4 Flash Attention 직접 구현 (선택적)

**목표:** Triton으로 Simplified Flash Attention 구현

**단계별 구현:**
1. **Basic Tiled Attention**: 블록 단위 QK^T 계산
2. **Online Softmax**: 점진적 softmax 계산
3. **Full Flash Attention**: 블록별 결과 결합

**Triton 구현 참고:**
```python
@triton.jit
def flash_attn_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Tiled attention computation
    # 1. Load Q block
    # 2. Iterate over K, V blocks
    # 3. Compute partial attention with online softmax
    # 4. Store output
    pass
```

**구현 난이도별 목표:**
- **기본**: flash-attn 패키지 사용법 완전 숙지
- **중급**: vLLM 백엔드 인터페이스 이해 및 커스텀 백엔드 작성
- **고급**: Triton으로 Flash Attention 핵심 로직 직접 구현

### 2.4 Paged Attention (vLLM 핵심)

**목표:** vLLM의 핵심 기술인 Paged Attention 완전 이해

**학습 내용:**
- Virtual Memory 개념의 KV Cache 적용
- Block 기반 메모리 관리
- Block Table 구조
- 메모리 단편화 해결
- Flash Attention + Paged KV Cache 통합

**vLLM 참고 파일:**
- `vLLM/csrc/attention/paged_attention_v1.cu`
- `vLLM/csrc/attention/paged_attention_v2.cu`
- `vLLM/vllm/v1/core/block_pool.py`

**구현 과제:**
1. Block Table 자료구조 구현
2. Block Allocator 구현
3. Flash Attention과 Paged KV Cache 연동 구현

---

## Phase 3: 모델 실행 계층

### 3.1 모델 로딩

**목표:** HuggingFace 모델의 효율적 로딩

**학습 내용:**
- Safetensors 포맷
- 메모리 매핑 (lazy loading)
- 가중치 변환

**vLLM 참고 파일:**
- `vLLM/vllm/model_executor/model_loader/`
- `vLLM/vllm/model_executor/parameter.py`

**구현 과제:**
1. HuggingFace 모델 메타데이터 파싱
2. Safetensors 로더 구현
3. 메모리 효율적 로딩 구현

### 3.2 레이어 구현

**목표:** LLM 구성 레이어 구현

**학습 내용:**
- Linear Layer
- Rotary Position Embedding (RoPE)
- MLP (Gate + Up + Down)

**vLLM 참고 파일:**
- `vLLM/vllm/model_executor/layers/linear.py`
- `vLLM/vllm/model_executor/layers/rotary_embedding.py`

**구현 과제:**
1. Linear Layer (추상화 포함)
2. RoPE 구현
3. LLaMA MLP 구현

### 3.3 LLaMA 모델 구현

**목표:** 완전한 LLaMA 모델 구현

**학습 내용:**
- LLaMA 아키텍처 분석
- Decoder Block 구조
- 추상화 설계 (향후 모델 확장용)

**vLLM 참고 파일:**
- `vLLM/vllm/model_executor/models/` - 200+ 모델 참조

**구현 과제:**
1. LLaMA Decoder Block 구현
2. 전체 모델 조립
3. HuggingFace 모델과 출력 비교 검증

---

## Phase 4: 스케줄링 및 배칭

### 4.1 요청 스케줄링

**목표:** 다중 요청 처리를 위한 스케줄러 구현

**학습 내용:**
- Continuous Batching
- Prefill/Decode 분리 스케줄링
- 메모리 기반 승인 제어
- Preemption 전략

**vLLM 참고 파일:**
- `vLLM/vllm/v1/core/sched/`
- `vLLM/vllm/config/scheduler.py`

**구현 과제:**
1. Request 자료구조 설계
2. FCFS 스케줄러 구현
3. 메모리 기반 승인 제어 구현

### 4.2 배치 처리

**목표:** 효율적인 동적 배칭 구현

**학습 내용:**
- Dynamic Batching
- Sequence Padding vs Packing
- 배치 크기 최적화

**vLLM 참고 파일:**
- `vLLM/vllm/v1/worker/gpu/input_batch.py`

**구현 과제:**
1. 동적 배치 구성기 구현
2. 가변 길이 시퀀스 처리
3. 처리량 벤치마크

### 4.3 LLM Engine

**목표:** 전체 추론 파이프라인 관리 엔진 구현

**학습 내용:**
- Request lifecycle
- Token generation loop
- Output streaming

**vLLM 참고 파일:**
- `vLLM/vllm/v1/engine/llm_engine.py`
- `vLLM/vllm/engine/arg_utils.py`

**구현 과제:**
1. Engine 클래스 설계
2. Generate 함수 구현
3. Streaming 지원

---

## Phase 5: 분산 추론 (핵심)

### 5.1 분산 병렬화 기초

**목표:** Tensor/Pipeline Parallelism 이해

**학습 내용:**
- Tensor Parallelism (TP): 레이어 내 분할
- Pipeline Parallelism (PP): 레이어 간 분할
- Data Parallelism (DP): 배치 분할
- PyTorch Distributed 기초

**vLLM 참고 파일:**
- `vLLM/vllm/distributed/parallel_state.py`
- `vLLM/vllm/config/parallel.py`

**구현 과제:**
1. ProcessGroup 초기화 구현
2. 간단한 TP 예제 구현
3. 간단한 PP 예제 구현

### 5.2 Ray 분산 실행 분석

**목표:** vLLM의 Ray 기반 분산 실행 완전 이해

**학습 내용:**
- Ray Actor 모델
- Placement Group (리소스 배치)
- Compiled DAG (최적화된 실행 그래프)
- Worker 통신 패턴

**핵심 파일:**
```
vLLM/vllm/v1/executor/
├── abstract.py          # Executor 추상 인터페이스
├── ray_executor.py      # Ray 분산 Executor (624줄)
├── ray_utils.py         # Worker Wrapper (469줄)
├── multiproc_executor.py # 멀티프로세스 Executor
└── uniproc_executor.py  # 단일 프로세스 Executor

vLLM/vllm/distributed/
├── parallel_state.py    # 병렬 상태 관리
├── device_communicators/
│   ├── ray_communicator.py  # Ray PP 통신
│   └── custom_all_reduce.py # 커스텀 AllReduce
└── kv_transfer/         # KV Cache 전송
```

**분석 과제:**
1. `RayDistributedExecutor` 클래스 분석
2. Worker 생성 및 Rank 할당 로직 분석
3. Compiled DAG 실행 흐름 분석
4. 중간 텐서 전송 메커니즘 분석

### 5.3 Kubernetes 기반 오케스트레이션 설계

**목표:** Ray를 Kubernetes로 대체하기 위한 설계

**Ray에서 대체해야 할 핵심 기능:**

| Ray 기능 | Kubernetes 대체 방안 |
|---------|---------------------|
| `ray.remote` Actor | StatefulSet + gRPC Service |
| Placement Group | Pod Affinity/Anti-Affinity |
| `collective_rpc()` | gRPC Fan-out/Fan-in |
| Compiled DAG | Custom Controller + CRD |
| Worker Discovery | K8s API + DNS |
| Env Propagation | ConfigMap/Secret |

**설계 과제:**
1. Worker Pod 사양 설계 (GPU 리소스, 네트워크)
2. gRPC 서비스 인터페이스 정의
3. Custom Resource Definition (CRD) 설계
4. Controller 로직 설계

### 5.4 분산 추론 구현

**목표:** Kubernetes 기반 분산 추론 구현

**구현 단계:**

```
Step 1: 단일 노드 멀티 GPU (TP)
├── torch.distributed 기반 통신
├── NCCL AllReduce/AllGather
└── 단일 프로세스 그룹

Step 2: 멀티 노드 (PP)
├── gRPC 기반 중간 텐서 전송
├── 파이프라인 스테이지 동기화
└── 노드 간 통신 최적화

Step 3: Kubernetes 통합
├── Custom Controller 구현
├── Pod 라이프사이클 관리
└── 동적 스케일링
```

**구현 과제:**
1. TP-aware Linear Layer 구현
2. PP 스테이지 통신 구현
3. gRPC Worker Service 구현
4. Kubernetes Controller 구현

---

## Phase 6: API 서버

### 6.1 HTTP API 서버

**목표:** OpenAI 호환 API 서버 구현

**학습 내용:**
- FastAPI 비동기 처리
- OpenAI API 스펙
- Request/Response 스키마

**vLLM 참고 파일:**
- `vLLM/vllm/entrypoints/serve/openai/api_server.py`
- `vLLM/vllm/entrypoints/serve/openai/protocol.py`

**구현 과제:**
1. FastAPI 서버 셋업
2. `/v1/completions` 구현
3. `/v1/chat/completions` 구현

### 6.2 스트리밍 응답

**목표:** SSE 기반 토큰 스트리밍 구현

**학습 내용:**
- Server-Sent Events (SSE)
- 비동기 제너레이터
- 연결 관리

**vLLM 참고 파일:**
- `vLLM/vllm/entrypoints/serve/openai/serving_chat.py`

**구현 과제:**
1. SSE 스트리밍 구현
2. 토큰별 응답 전송
3. 에러 핸들링

---

## Phase 7: 고급 주제 (향후)

### 7.1 양자화
- INT8/INT4 양자화 지원
- GPTQ, AWQ 포맷
- 참고: `vLLM/vllm/model_executor/layers/quantization/`

### 7.2 Speculative Decoding
- Draft 모델 기반 가속
- 토큰 검증 메커니즘
- 참고: `vLLM/vllm/v1/spec_decode/`

### 7.3 Heterogeneous GPU
- 다른 종류의 GPU 혼합 사용
- 로드 밸런싱 전략
- 메모리 이종성 처리

---

## 마일스톤

### Milestone 1: 기초 완료
- [ ] CUDA 커널 3개 이상 구현
- [ ] PyTorch Extension 빌드 성공
- [ ] 단위 테스트 통과

### Milestone 2: Flash Attention 통합
- [ ] flash-attn 패키지 사용법 숙지
- [ ] 백엔드 추상화 레이어 구현
- [ ] Flash Attention vs Standard Attention 성능 비교
- [ ] (선택) Triton으로 Simplified Flash Attention 구현

### Milestone 3: 단일 GPU 추론
- [ ] LLaMA 모델 로딩 성공
- [ ] 단일 GPU에서 텍스트 생성 성공
- [ ] HuggingFace와 출력 일치 확인

### Milestone 4: Paged Attention
- [ ] Paged Attention 구현
- [ ] Flash Attention + Paged KV Cache 연동
- [ ] 메모리 효율성 개선 확인
- [ ] 다중 요청 처리 가능

### Milestone 5: 분산 추론 (TP)
- [ ] 단일 노드 멀티 GPU 동작
- [ ] NCCL 통신 성공
- [ ] 스케일링 효율성 측정

### Milestone 6: 분산 추론 (PP + K8s)
- [ ] Kubernetes Controller 동작
- [ ] 멀티 노드 추론 성공
- [ ] gRPC 통신 안정성 확인

### Milestone 7: 프로덕션 준비
- [ ] OpenAI 호환 API 서버
- [ ] 스트리밍 응답 동작
- [ ] 부하 테스트 통과

---

## 참고 자료

### 논문
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023)
- "Megatron-LM: Training Multi-Billion Parameter Language Models"

### 문서
- vLLM 공식 문서: https://docs.vllm.ai/
- Flash Attention GitHub: https://github.com/Dao-AILab/flash-attention
- Triton 튜토리얼: https://triton-lang.org/main/getting-started/tutorials/
- CUDA Programming Guide
- PyTorch Distributed Tutorial
- Kubernetes Operator Pattern

### 코드 참조
- vLLM GitHub: https://github.com/vllm-project/vllm
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- Triton Flash Attention 예제: https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
- HuggingFace Transformers
