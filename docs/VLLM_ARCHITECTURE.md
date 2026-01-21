# vLLM 아키텍처 분석

이 문서는 vLLM의 전체 아키텍처를 Low-level(커널)부터 High-level(API 서버)까지 분석한 내용을 담고 있다.

---

## 전체 아키텍처 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL USER API                      │
│                   vllm/entrypoints/llm.py                   │
│                  (LLM class - main entry point)             │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────────┐ ┌────▼──────┐ ┌──────▼──────────┐
│  Chat Utils    │ │ Completion│ │ Embedding       │
│  Score Utils   │ │ Server    │ │ OpenAI Protocol │
└────────────────┘ └───────────┘ └─────────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │      V1 ENGINE LAYER            │
        │   vllm/v1/engine/llm_engine.py  │
        │   (Request scheduling & batching)│
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │    EXECUTOR LAYER               │
        │  vllm/v1/executor/              │
        │ (Ray/Multiproc/Uniproc)         │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │    WORKER LAYER                 │
        │  vllm/v1/worker/gpu/            │
        │ (GPU Model Runner)              │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┼────────────────────────┐
        │                │                        │
┌───────▼───────┐ ┌──────▼────────┐ ┌───────────▼─┐
│ Model Executor│ │ Attention Ops │ │ KV Cache Mgr│
│ (vllm/        │ │ (vllm/        │ │ (vllm/v1/   │
│  model_       │ │  attention/)  │ │  core/)     │
│  executor/)   │ └───────┬────────┘ └─────────────┘
└───────┬───────┘         │
        │                 │
        ├────────┬────────┘
        │        │
        │        └──────────────┐
        │                       │
┌───────▼───────────────────────▼──────────┐
│           ATTENTION BACKENDS              │
│         vllm/attention/backends/          │
│ (FlashAttn, Custom, vLLM Kernels, etc)   │
└────────┬─────────────────────────┬────────┘
         │                         │
┌────────▼────────┐     ┌──────────▼───────────┐
│  CUDA Kernels   │     │   PyTorch/Triton     │
│  csrc/attention/│     │   Operations         │
│  (Paged Attn)   │     │   (Compilation)      │
└─────────────────┘     └──────────────────────┘
```

---

## 1. Low-Level: CUDA 커널 (csrc/)

### 디렉토리 구조
```
vLLM/csrc/
├── attention/              # Attention 커널
│   ├── paged_attention_v1.cu
│   ├── paged_attention_v2.cu
│   ├── attention_kernels.cuh
│   └── mla/               # Multi-head Latent Attention
├── quantization/          # 양자화 커널
│   ├── gptq_marlin/
│   ├── awq/
│   └── fp4/
├── core/                  # 핵심 유틸리티
│   └── scalar_type.hpp
├── moe/                   # Mixture of Experts
├── layernorm_kernels.cu
├── activation_kernels.cu
├── cache_kernels.cu
├── pos_encoding_kernels.cu
├── sampler.cu
└── torch_bindings.cpp     # PyTorch 바인딩
```

### 핵심 커널들
- **paged_attention_v1/v2**: vLLM의 핵심 - 페이지 기반 KV Cache Attention
- **layernorm_kernels**: RMSNorm, LayerNorm 구현
- **activation_kernels**: SiLU, GELU 등 활성화 함수
- **cache_kernels**: KV Cache 복사/관리 연산

---

## 2. Attention 계층 (vllm/attention/)

### 파일 구조
```
vllm/attention/
├── layer.py           # 메인 Attention 레이어
├── selector.py        # 백엔드 선택 로직
├── backends/          # 다양한 Attention 백엔드
│   ├── abstract.py
│   ├── flash_attn.py
│   ├── xformers.py
│   └── ...
├── layers/            # Attention 관련 레이어
└── ops/               # Attention 연산
```

### 역할
- Python과 CUDA 커널 사이의 브릿지
- 다양한 Attention 백엔드 지원 (FlashAttention, xFormers 등)
- 하드웨어에 따른 최적 백엔드 자동 선택

---

## 3. Model Executor (vllm/model_executor/)

### 디렉토리 구조
```
vllm/model_executor/
├── models/            # 200+ 모델 구현
│   ├── llama.py
│   ├── mistral.py
│   └── ...
├── layers/            # 재사용 가능한 레이어
│   ├── linear.py
│   ├── rotary_embedding.py
│   ├── layernorm.py
│   ├── activation.py
│   └── quantization/
├── model_loader/      # 모델 로딩
└── parameter.py       # 파라미터 관리
```

### 핵심 레이어
- **Linear**: 양자화 지원 선형 레이어
- **Rotary Embedding**: RoPE 위치 인코딩
- **LayerNorm**: RMSNorm/LayerNorm
- **MLP**: Gate-Up-Down 구조

---

## 4. V1 Engine (vllm/v1/)

### 디렉토리 구조
```
vllm/v1/
├── engine/
│   └── llm_engine.py      # 메인 엔진
├── core/
│   ├── sched/             # 스케줄러
│   ├── block_pool.py      # 블록 풀 관리
│   └── kv_cache_coordinator.py
├── executor/
│   ├── abstract.py        # Executor 인터페이스
│   ├── ray_executor.py    # Ray 기반 분산
│   ├── multiproc_executor.py
│   └── uniproc_executor.py
├── worker/
│   └── gpu/
│       ├── model_runner.py
│       └── input_batch.py
├── attention/             # V1 Attention
├── sample/                # 샘플링
└── spec_decode/           # Speculative Decoding
```

### V1 엔진 특징
- 새로운 아키텍처 (V0 대비 개선)
- 더 효율적인 스케줄링
- Compiled DAG 지원

---

## 5. 분산 추론 (vllm/distributed/)

### 디렉토리 구조
```
vllm/distributed/
├── parallel_state.py      # 병렬 상태 관리
├── device_communicators/
│   ├── ray_communicator.py
│   ├── custom_all_reduce.py
│   └── pynccl.py
├── kv_transfer/           # KV Cache 전송
└── eplb/                  # Expert 로드 밸런싱
```

### 병렬화 전략
- **Tensor Parallelism (TP)**: 레이어 내 분할 (NCCL)
- **Pipeline Parallelism (PP)**: 레이어 간 분할 (Ray/gRPC)
- **Data Parallelism (DP)**: 배치 분할

### GroupCoordinator 구조
```python
GroupCoordinator:
├── rank: int              # 전역 rank
├── world_size: int        # 그룹 크기
├── local_rank: int        # 로컬 장치 ID
├── cpu_group: ProcessGroup  # Gloo 백엔드
├── device_group: ProcessGroup  # NCCL 백엔드
└── device_communicator: DeviceCommunicatorBase
```

---

## 6. Ray Executor 분석 (vllm/v1/executor/ray_executor.py)

### 핵심 클래스: RayDistributedExecutor

```python
class RayDistributedExecutor(Executor):
    """Ray 기반 분산 실행기"""

    def _init_workers_ray(self):
        # 1. Placement Group 생성
        # 2. Worker Actor 생성
        # 3. Rank 할당
        # 4. 환경변수 전파
        # 5. 모델 로딩
        pass

    def _compiled_ray_dag(self):
        # Ray Compiled DAG 생성
        # PP 스테이지 간 텐서 전송 최적화
        pass

    def collective_rpc(self, method, args):
        # 모든 Worker에 동시 호출
        pass
```

### Worker 구조 (PP=2, TP=4 예시)
```
pp_tp_workers[PP rank][TP rank]
├── PP 0: [Worker 0, Worker 1, Worker 2, Worker 3]
└── PP 1: [Worker 4, Worker 5, Worker 6, Worker 7]
```

### Compiled DAG 실행 흐름
```
Input: (SchedulerOutput, GrammarOutput)
  ↓
TP Workers (PP rank 0)
  ↓ [NCCL/SharedMemory]
TP Workers (PP rank 1)
  ↓
Output: ModelRunnerOutput
```

---

## 7. API 서버 (vllm/entrypoints/)

### 디렉토리 구조
```
vllm/entrypoints/
├── llm.py                 # LLM 클래스 (Python API)
├── serve/
│   ├── openai/
│   │   ├── api_server.py  # FastAPI 서버
│   │   ├── protocol.py    # OpenAI 프로토콜
│   │   ├── serving_chat.py
│   │   └── serving_completion.py
│   └── anthropic/         # Anthropic API 호환
├── cli/                   # CLI 인터페이스
└── launcher.py            # 서버 런처
```

### OpenAI 호환 엔드포인트
- `POST /v1/completions` - 텍스트 완성
- `POST /v1/chat/completions` - 채팅 완성
- `POST /v1/embeddings` - 임베딩 생성

---

## 8. 설정 (vllm/config/)

### 주요 설정 파일
```
vllm/config/
├── model.py           # 모델 설정
├── attention.py       # Attention 설정
├── cache.py           # KV Cache 설정
├── parallel.py        # 병렬화 설정
├── scheduler.py       # 스케줄러 설정
├── device.py          # 장치 설정
├── load.py            # 모델 로딩 설정
└── vllm.py            # 통합 설정
```

### ParallelConfig 주요 필드
```python
@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    distributed_executor_backend: str = "ray"  # ray, mp, uni
```

---

## 9. 핵심 개념 요약

### Paged Attention
- KV Cache를 고정 크기 블록으로 관리
- 가상 메모리처럼 Block Table 사용
- 메모리 단편화 최소화

### Continuous Batching
- 요청별로 토큰 생성 완료시 즉시 새 요청 추가
- GPU 활용률 극대화

### Tensor Parallelism
- 레이어 내 가중치를 GPU들에 분산
- AllReduce/AllGather로 결과 동기화

### Pipeline Parallelism
- 모델 레이어를 여러 노드에 분산
- 중간 텐서를 다음 스테이지로 전송

---

## 10. 주요 진입점

### Python API
```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(["Hello, world!"])
```

### 서버 실행
```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 4
```

### 핵심 클래스 진입점
- `vllm/entrypoints/llm.py`: `LLM` 클래스
- `vllm/v1/engine/llm_engine.py`: V1 엔진
- `vllm/v1/executor/ray_executor.py`: 분산 실행기
- `vllm/v1/worker/gpu/model_runner.py`: GPU 워커
