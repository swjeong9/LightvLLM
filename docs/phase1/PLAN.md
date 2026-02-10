# Phase 1: LightvLLM 기본 커널 구현 계획

## 목표
자연어 입력을 받아 LLaMA 모델로 추론하여 출력하는 기본 파이프라인 구현 (학습 목적)

---

## 전체 파이프라인

```
Text → Tokenizer → Embedding (PyTorch) → RoPE → Attention → MLP (SiLU+Mul) → Output
```

- **Embedding**: PyTorch `F.embedding()` 사용 (vLLM도 동일)
- **RoPE, Attention, MLP**: CUDA 커널로 구현

---

## 완료된 작업

### 1. GPU 메모리 최적화 매크로 — `csrc/cuda_compat.h` (182줄) ✅

GPU 메모리 계층 구조(레지스터 → 공유메모리 → L1 → L2 → 글로벌)에 대한 상세 교육 문서와 함께 두 가지 핵심 매크로를 구현:

- **`VLLM_LDG(arg)`**: `__ldg()` 래핑 매크로. 읽기 전용 데이터(모델 가중치, cos/sin 캐시 등)를 텍스처 캐시 경로로 로드하여 L1 캐시 오염(cache pollution)을 방지. 메모리 바운드 커널에서 5~20% 성능 향상 가능.
- **`WARP_SIZE = 32`**: NVIDIA GPU 워프 크기 상수. 리덕션 연산, 블록 크기 설정 등에 활용.

### 2. 타입 디스패치 매크로 — `csrc/dispatch_utils.h` (279줄) ✅

PyTorch 텐서의 런타임 dtype과 C++ 템플릿 컴파일타임 타입 간의 브릿지 역할을 하는 매크로 구현:

- **`VLLM_DISPATCH_CASE_FLOATING_TYPES`**: float32, float16, bfloat16 세 가지 타입에 대한 switch case 생성
- **`VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)`**: 메인 디스패치 매크로. `AT_DISPATCH_SWITCH`를 래핑하여 런타임에 적절한 타입의 커널을 선택적으로 호출

### 3. RoPE 순수 CUDA 헤더 — `csrc/pos_encoding_kernels.cuh` (219줄) ✅

PyTorch 의존성 없이 사용 가능한 RoPE 커널 헤더. `lightvllm` 네임스페이스 내에 3개 함수/커널 구현:

- **`apply_token_rotary_embedding<scalar_t, IS_NEOX>()`**: 단일 (x, y) 쌍에 2D 회전 변환 적용. GPT-NeoX(전반부/후반부 쌍) 및 GPT-J(인접 쌍) 스타일 모두 지원. `VLLM_LDG` 매크로로 cos/sin 값을 텍스처 캐시 경로로 로드.
- **`apply_rotary_embedding<scalar_t, IS_NEOX>()`**: 하나의 토큰에 대해 모든 Query/Key head에 RoPE 적용. 블록 내 스레드들이 `threadIdx.x + blockDim.x` stride로 협력 처리. GQA 지원 (`num_kv_heads < num_heads`).
- **`rotary_embedding_kernel<scalar_t, IS_NEOX>()`**: 메인 CUDA 커널. Grid=(num_tokens), Block=(min(num_heads*rot_dim/2, 512)) 병렬화 전략.

### 4. RoPE PyTorch 연동 — `csrc/pos_encoding_kernels.cu` (209줄) ✅

RoPE의 수학적 원리(2D 회전 변환, 상대 위치 인코딩 증명, Q/K에만 적용하는 이유 등)에 대한 상세 교육 문서와 함께 PyTorch 진입점 함수 구현:

- **`rotary_embedding()`**: Python에서 호출되는 C++ 함수. 텐서 크기에서 num_heads, num_kv_heads, rot_dim 등을 자동 추출하고, `VLLM_DISPATCH_FLOATING_TYPES`로 dtype 디스패치 후 커널 launch. `CUDAGuard`로 멀티 GPU 지원.

### 5. Low-level RoPE 테스트 — `tests/test_rope_kernel.cu` (512줄) ✅

PyTorch 없이 CUDA 커널을 직접 테스트하는 C++ 테스트 스위트:

- **CPU 참조 구현**: GPU 결과 비교용 `rope_cpu_reference()` 함수 (GPT-NeoX 스타일)
- **cos/sin 캐시 생성**: `generate_cos_sin_cache()` — θ_i = pos × base^(-2i/d) 공식으로 CPU에서 캐시 생성
- **테스트 유틸리티**: `compare_arrays()` (오차 비교), `compute_norm()` (L2 norm)
- **테스트 케이스 3개**:
  - Test 1 — Basic Correctness: GPU vs CPU 참조 구현 결과 비교 (1024토큰, 64헤드, 8 KV헤드, 128 head_size). CPU/GPU 실행시간 및 speedup 측정.
  - Test 2 — Norm Preservation: RoPE 적용 전후 head 벡터의 L2 norm이 보존되는지 검증 (회전 행렬의 직교성).
  - Test 3 — Position 0 Identity: position=0일 때 cos(0)=1, sin(0)=0이므로 입력=출력인지 검증.
- **컴파일**: `nvcc -O2 -std=c++17 -I csrc tests/test_rope_kernel.cu -o tests/test_rope_kernel`

### 6. 프로젝트 인프라 ✅

- **`pyproject.toml`**: PEP 621 표준, lightvllm 0.1.0, torch>=2.8.0, pytest 설정
- **패키지 구조**: `lightvllm/{kernels,layers,attention,models,utils}/` 스켈레톤 구성 완료
- **`lightvllm/utils/logging.py`**: 로깅 유틸리티 완성

### 7. PyTorch 바인딩 — `csrc/torch_bindings.cpp` ✅

PYBIND11을 통해 CUDA 커널을 Python에서 호출 가능하게 연결:

- **`rotary_embedding`** 함수 선언 및 `m.def()` 바인딩 활성화
- Python에서 `import lightvllm._C` → `_C.rotary_embedding(...)` 호출 가능

### 8. CUDA Extension 빌드 설정 — `setup.py` ✅

`torch.utils.cpp_extension.CUDAExtension`을 사용한 빌드 스크립트:

- **빌드**: `uv run python setup.py build_ext --inplace`
- **결과물**: `lightvllm/_C.cpython-*.so`
- `pyproject.toml`의 `build-system.requires`에 `torch>=2.8.0` 추가하여 빌드 환경에서도 torch 사용 가능

### 9. Python RoPE 테스트 — `tests/kernels/test_rope.py` ✅

CUDA 커널을 PyTorch GPU 참조 구현과 비교하는 테스트 (기본 dtype: **bf16**):

- **참조 구현**: `rope_reference()` — GPU에서 동일 dtype으로 실행 (CPU bf16은 내부적으로 float32 변환하여 결과가 달라지므로 GPU 필수)
- **cos/sin 캐시 생성**: `generate_cos_sin_cache()` — float32로 계산 후 target dtype으로 변환
- **테스트 케이스 6개** (모두 bit-exact 일치, `atol=0, rtol=0`):
  - Test 1 — Basic Correctness: bf16, 32토큰, 8헤드, 2 KV헤드
  - Test 2 — Norm Preservation: 회전 전후 L2 norm 보존 확인
  - Test 3 — Position 0 Identity: position=0에서 항등 변환 확인
  - Test 4 — Half Dtypes: float16, bfloat16 파라미터화 테스트
  - Test 5 — Large Scale: LLaMA-like 설정 (1024토큰, 32헤드, 8 KV헤드, 128 head_size)

---

## 다음 작업 (예정)

| 순서 | 작업 | 파일 | 내용 |
|------|------|------|------|
| 3 | RMSNorm 커널 | `csrc/layernorm_kernels.cu` | RMSNorm + Fused Add+RMSNorm CUDA 커널, Low-level 테스트 |
| 4 | Activation 커널 | `csrc/activation_kernels.cu` | SiLU, SiLU+Mul fused 연산 (LLaMA MLP용), Low-level 테스트 |
| 5 | Attention | `lightvllm/attention/` | Self-Attention 커널 또는 naive PyTorch 구현 |
| 6 | Python Integration | `lightvllm/models/llama.py` | 전체 파이프라인 연결: Tokenizer → Embedding → RoPE → Attention → MLP → Output |

---

## 참조

- vLLM 원본: `/home/ubuntu/LightvLLM/vLLM/csrc/`
- 학습 로드맵: `/home/ubuntu/LightvLLM/docs/LEARNING_ROADMAP.md`
