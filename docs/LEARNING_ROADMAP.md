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
│                      LightvLLM 학습 로드맵 (수정본)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: LLaMA 기초 추론                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1.1 CUDA 기초    →  1.2 전체 커널    →  1.3 레이어    →  1.4 LLaMA  │   │
│  │  ┌───────────┐       ┌───────────┐      ┌───────────┐   ┌───────────┐│   │
│  │  │ CUDA 기초  │       │ RMSNorm   │      │ Linear    │   │ 모델 로더 ││   │
│  │  │ PyTorch   │  ──▶  │ RoPE      │  ──▶ │ MLP       │──▶│ LLaMA 구현││   │
│  │  │ Extension │       │ SiLU+Mul  │      │ Attention │   │ HF 검증   ││   │
│  │  └───────────┘       └───────────┘      └───────────┘   └───────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  Phase 2: 최적화                Phase 3: 스케줄링        Phase 4: 분산        │
│  ┌───────────────┐             ┌───────────────┐       ┌───────────────┐   │
│  │ Flash Attn    │             │ Scheduler     │       │ Ray 분석      │   │
│  │ KV Cache      │      ──▶    │ Batching      │  ──▶  │ TP/PP 구현    │   │
│  │ Paged Attn    │             │ Engine        │       │ K8s 전환      │   │
│  └───────────────┘             └───────────────┘       └───────────────┘   │
│                                                                 ↓           │
│                                                        Phase 5: 서버        │
│                                                        ┌───────────────┐   │
│                                                        │ API Server    │   │
│                                                        │ OpenAI 호환   │   │
│                                                        │ Streaming     │   │
│                                                        └───────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**핵심 변경사항:**
- LLaMA 추론에 필요한 모든 커널/레이어를 먼저 구현하여 **정확성 검증** 가능
- Flash Attention, Paged Attention 등 최적화는 기본 추론 검증 후 진행

---

## Phase 1: LLaMA 기초 추론 (검증 가능한 최소 구현)

### 1.1 CUDA 및 PyTorch Extension 기초

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

### 1.2 LLaMA 필수 커널 구현

**목표:** LLaMA 추론에 필요한 모든 기본 커널 구현

**LLaMA 추론 흐름:**
```
Embedding → (RMSNorm → Attention → RMSNorm → MLP) × N → RMSNorm → LM Head → Sampling
```

**필수 커널 목록:**

| 커널 | 용도 | vLLM 참고 파일 |
|------|------|----------------|
| RMSNorm | 정규화 | `csrc/layernorm_kernels.cu` |
| Fused Add + RMSNorm | 잔차 연결 + 정규화 | `csrc/layernorm_kernels.cu` |
| Rotary Embedding (RoPE) | 위치 인코딩 | `csrc/pos_encoding_kernels.cu` |
| SiLU + Mul | MLP 활성화 | `csrc/activation_kernels.cu` |

**구현 과제:**
1. RMSNorm 커널 구현
   - 입력: `[batch, seq_len, hidden_size]`
   - 연산: `x / sqrt(mean(x^2) + eps) * weight`
2. Fused Add + RMSNorm 구현
   - 잔차 연결과 정규화를 한 번에 처리
3. Rotary Position Embedding 구현
   - Query, Key에 회전 변환 적용
4. SiLU + Mul 커널 구현
   - `SiLU(gate) * up` fused 연산

**검증:** 각 커널의 출력을 PyTorch 순수 구현과 비교

### 1.3 모델 레이어 구현

**목표:** LLaMA 구성 레이어 구현

**레이어 목록:**

| 레이어 | 설명 | 사용하는 커널 |
|--------|------|---------------|
| RMSNorm | 정규화 레이어 | RMSNorm 커널 |
| Linear | 선형 변환 | (PyTorch 기본) |
| RotaryEmbedding | RoPE 레이어 | RoPE 커널 |
| LlamaMLP | Gate + Up + Down | SiLU + Mul 커널 |
| LlamaAttention | Multi-Head Attention | RoPE, (naive attention) |

**vLLM 참고 파일:**
- `vLLM/vllm/model_executor/layers/linear.py`
- `vLLM/vllm/model_executor/layers/rotary_embedding.py`
- `vLLM/vllm/model_executor/layers/activation.py`
- `vLLM/vllm/model_executor/layers/layernorm.py`

**구현 과제:**
1. RMSNorm 레이어 (커널 래핑)
2. Linear 레이어 (추상화 준비)
3. RotaryEmbedding 레이어
4. LlamaMLP 레이어 (Gate + Up + Down)
5. LlamaAttention 레이어 (**naive PyTorch 구현**)
   - 최적화 없이 정확성 검증용
   - 기본 KV Cache 지원

### 1.4 LLaMA 모델 구현 및 검증

**목표:** 완전한 LLaMA 모델 구현 및 HuggingFace와 출력 비교

**LLaMA 아키텍처:**
```
LlamaForCausalLM
├── LlamaModel
│   ├── Embedding (token → hidden)
│   ├── LlamaDecoderLayer × N
│   │   ├── RMSNorm (input)
│   │   ├── LlamaAttention
│   │   ├── Residual + RMSNorm
│   │   └── LlamaMLP + Residual
│   └── RMSNorm (final)
└── LMHead (hidden → vocab)
```

**vLLM 참고 파일:**
- `vLLM/vllm/model_executor/models/llama.py`

**구현 과제:**
1. HuggingFace 모델 로더 구현
   - Safetensors 파일 읽기
   - 가중치 변환 및 매핑
2. LlamaDecoderLayer 구현
3. LlamaModel 전체 조립
4. LlamaForCausalLM (LM Head 포함)

**검증 방법:**
```python
# HuggingFace와 출력 비교
hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
light_model = LightvLLM.from_pretrained("meta-llama/Llama-2-7b-hf")

input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids

hf_output = hf_model(input_ids).logits
light_output = light_model(input_ids).logits

assert torch.allclose(hf_output, light_output, atol=1e-4)
```

---

## Phase 2: 최적화 (Attention & KV Cache)

> **전제조건:** Phase 1 완료 후 HuggingFace와 출력 일치 확인

### 2.1 Flash Attention 통합

**목표:** Flash Attention 패키지 사용법 습득 및 통합

**학습 내용:**
- Standard Attention의 메모리 복잡도 문제 (O(N²) 메모리)
- Tiling 기반 접근법 (Block-wise 계산)
- Online Softmax 알고리즘

**핵심 논문:**
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023)

**vLLM 참고 파일:**
- `vLLM/vllm/attention/utils/fa_utils.py`
- `vLLM/vllm/v1/attention/backends/flash_attn.py`

**구현 과제:**
1. flash-attn 패키지 설치 및 기본 사용법 익히기
2. Attention 백엔드 추상화 레이어 구현
3. Flash Attention 백엔드 구현
4. Naive vs Flash Attention 성능 비교

### 2.2 KV Cache 고도화

**목표:** 효율적인 KV Cache 관리 구현

**학습 내용:**
- Prefill vs Decode 단계
- Cache 메모리 관리
- Cache 재사용 전략

**vLLM 참고 파일:**
- `vLLM/vllm/config/cache.py`
- `vLLM/csrc/cache_kernels.cu`

**구현 과제:**
1. KV Cache 관리자 구현
2. Cache 할당/재사용 로직 구현
3. 메모리 사용량 분석

### 2.3 Paged Attention (vLLM 핵심)

**목표:** vLLM의 핵심 기술인 Paged Attention 완전 이해

**학습 내용:**
- Virtual Memory 개념의 KV Cache 적용
- Block 기반 메모리 관리
- Block Table 구조
- 메모리 단편화 해결

**vLLM 참고 파일:**
- `vLLM/csrc/attention/paged_attention_v1.cu`
- `vLLM/csrc/attention/paged_attention_v2.cu`
- `vLLM/vllm/v1/core/block_pool.py`

**구현 과제:**
1. Block Table 자료구조 구현
2. Block Allocator 구현
3. Paged Attention 커널 구현 (또는 통합)
4. Flash Attention + Paged KV Cache 연동

---

## Phase 3: 스케줄링 및 배칭

### 3.1 요청 스케줄링

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

### 3.2 배치 처리

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

### 3.3 LLM Engine

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

## Phase 4: 분산 추론

### 4.1 분산 병렬화 기초

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

### 4.2 Ray 분산 실행 분석

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
├── ray_executor.py      # Ray 분산 Executor
├── ray_utils.py         # Worker Wrapper
├── multiproc_executor.py # 멀티프로세스 Executor
└── uniproc_executor.py  # 단일 프로세스 Executor
```

**분석 과제:**
1. `RayDistributedExecutor` 클래스 분석
2. Worker 생성 및 Rank 할당 로직 분석
3. Compiled DAG 실행 흐름 분석

### 4.3 Kubernetes 기반 오케스트레이션 설계

**목표:** Ray를 Kubernetes로 대체하기 위한 설계

**Ray에서 대체해야 할 핵심 기능:**

| Ray 기능 | Kubernetes 대체 방안 |
|---------|---------------------|
| `ray.remote` Actor | StatefulSet + gRPC Service |
| Placement Group | Pod Affinity/Anti-Affinity |
| `collective_rpc()` | gRPC Fan-out/Fan-in |
| Compiled DAG | Custom Controller + CRD |
| Worker Discovery | K8s API + DNS |

**구현 과제:**
1. TP-aware Linear Layer 구현
2. PP 스테이지 통신 구현
3. gRPC Worker Service 구현
4. Kubernetes Controller 구현

---

## Phase 5: API 서버

### 5.1 HTTP API 서버

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

### 5.2 스트리밍 응답

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

## Phase 6: 고급 주제 (향후)

### 6.1 양자화
- INT8/INT4 양자화 지원
- GPTQ, AWQ 포맷
- 참고: `vLLM/vllm/model_executor/layers/quantization/`

### 6.2 Speculative Decoding
- Draft 모델 기반 가속
- 토큰 검증 메커니즘
- 참고: `vLLM/vllm/v1/spec_decode/`

### 6.3 Heterogeneous GPU
- 다른 종류의 GPU 혼합 사용
- 로드 밸런싱 전략
- 메모리 이종성 처리

---

## 마일스톤

### Milestone 1: LLaMA 기초 추론 ⭐ (핵심)
- [ ] RMSNorm, RoPE, SiLU+Mul 커널 구현
- [ ] PyTorch Extension 빌드 성공
- [ ] 모든 레이어 구현 (Linear, MLP, Attention)
- [ ] LLaMA 모델 로더 구현
- [ ] **HuggingFace와 출력 일치 확인** ✓

### Milestone 2: Attention 최적화
- [ ] Flash Attention 통합
- [ ] 백엔드 추상화 레이어 구현
- [ ] Flash vs Naive 성능 비교

### Milestone 3: Paged Attention
- [ ] Paged Attention 구현
- [ ] Flash Attention + Paged KV Cache 연동
- [ ] 메모리 효율성 개선 확인

### Milestone 4: 다중 요청 처리
- [ ] Continuous Batching 구현
- [ ] Scheduler 구현
- [ ] Engine 클래스 완성

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
