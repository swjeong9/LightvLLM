# Phase 1 작업 이력 총정리

Phase 1의 목표는 LLaMA 기본 추론 파이프라인을 구현하는 것이다.
이 문서는 Phase 1 전체의 작업 흐름과 각 단계에서 무엇을 했는지를 시간순으로 정리한다.

---

## 타임라인

### 2026-01-16 — 프로젝트 초기화

**커밋**: `fb884fe`, `f360944`

- uv로 Python 프로젝트 초기화 (`pyproject.toml`, `.gitignore`, `uv.lock`)
- vLLM을 git submodule로 추가 (레퍼런스 구현 참조용)
- 기본 디렉토리 구조 생성

### 2026-01-21 — 문서 작성 및 Phase 1 골격

**커밋**: `9bf72e8`, `0a93504`

- 프로젝트 핵심 문서 작성:
  - `docs/LEARNING_ROADMAP.md` — 전체 학습 로드맵 (Phase 1~6)
  - `docs/VLLM_ARCHITECTURE.md` — vLLM 아키텍처 분석
  - `docs/RAY_TO_K8S_MIGRATION.md` — Ray→Kubernetes 마이그레이션 가이드
  - `README.md` — 프로젝트 개요
- Phase 1 패키지 스켈레톤 생성:
  - `lightvllm/` 하위 모듈 구조 (kernels, layers, attention, models, utils)
  - 각 모듈에 `__init__.py` 및 플레이스홀더 파일 생성
  - `lightvllm/utils/logging.py` 구현
- 테스트 스켈레톤 생성:
  - `tests/kernels/` (test_rope.py, test_layernorm.py, test_activation.py)
  - `tests/models/test_llama.py`

### 2026-01-26 — CUDA 인프라 및 RoPE 시작

**커밋**: `4974b47`, `e72f1d9`

- **GPU 메모리 최적화 매크로** — `csrc/cuda_compat.h` (182줄)
  - `VLLM_LDG()`: 텍스처 캐시 경로 로드 매크로
  - `WARP_SIZE = 32`: 워프 크기 상수
  - GPU 메모리 계층 구조에 대한 상세 한국어 교육 주석

- **타입 디스패치 매크로** — `csrc/dispatch_utils.h` (279줄)
  - `VLLM_DISPATCH_FLOATING_TYPES()`: 런타임 dtype → 컴파일타임 템플릿 브릿지
  - float32, float16, bfloat16 지원

- **RoPE 순수 CUDA 헤더 작성 시작** — `csrc/pos_encoding_kernels.cuh`

### 2026-02-09 — RoPE 완성

**커밋**: `5bec475`, `eec5573`, `ee312c5`

- **torch 버전 의존성 완화** — `pyproject.toml`에서 torch>=2.8.0 설정
- **RoPE 순수 CUDA 헤더 완성** — `csrc/pos_encoding_kernels.cuh` (219줄)
  - `apply_token_rotary_embedding()`: 단일 (x,y) 쌍 회전
  - `apply_rotary_embedding()`: 전체 head에 RoPE 적용 (GQA 지원)
  - `rotary_embedding_kernel()`: 메인 CUDA 커널
  - GPT-NeoX, GPT-J 두 스타일 지원
- **RoPE PyTorch 연동** — `csrc/pos_encoding_kernels.cu` (314줄)
  - RoPE 수학적 원리에 대한 상세 교육 문서 (한국어)
  - `rotary_embedding()` PyTorch 진입점 함수
  - dtype 디스패치, 멀티GPU CUDAGuard 지원
- **PyTorch 바인딩** — `csrc/torch_bindings.cpp`
  - pybind11로 `lightvllm._C.rotary_embedding()` 노출
- **CUDA Extension 빌드** — `setup.py`
  - `CUDAExtension`으로 .cu 파일 컴파일
- **C++ Low-level 테스트** — `tests/test_rope_kernel.cu` (512줄)
  - CPU 참조 구현, GPU vs CPU 비교
  - 3개 테스트: 정확성, 노름 보존, Position 0 항등성
  - 성능 벤치마크 (cudaEvent 타이밍)

### 2026-02-10 — RoPE Python 테스트 완성

**커밋**: `d6d4900`

- **Python 통합 테스트** — `tests/kernels/test_rope.py` (273줄)
  - GPU에서 동일 dtype으로 실행하는 참조 구현 (`rope_reference()`)
  - 6개 테스트: 기본 정확성, 노름 보존, Position 0, dtype별, 대규모
  - 모든 테스트 **bit-exact 일치** (atol=0, rtol=0)

### 2026-02-11 — RMSNorm 커널 구현

(상세 내용은 `2026-02-11_rmsnorm.md` 참조)

- **RMSNorm + Fused Add+RMSNorm CUDA 커널** 전체 구현
- CUB BlockReduce 기반 병렬 리덕션
- 순수 CUDA 헤더 + PyTorch wrapper + Python 래퍼 + nn.Module 클래스
- C++ 테스트 4개 + Python 테스트 14개, 전체 25개 테스트 통과
- GPU 커널 CPU 대비 168x 속도 향상

### 2026-02-12 — Activation 커널 구현 + Python 래퍼 계층 완성

(상세 내용은 `2026-02-12_activation.md` 참조)

- **SiLU + Fused silu_and_mul CUDA 커널** 전체 구현
- 128-bit 벡터화(`int4`) 메모리 접근으로 대역폭 최적화
- 순수 CUDA 헤더 + PyTorch wrapper + Python 래퍼 + nn.Module 클래스
- C++ 테스트 5개 + Python 테스트 31개, 전체 63개 테스트 통과
- Fused 커널 Non-Fused 대비 1.69x 속도 향상
- **RoPE Python 래퍼 스텁 구현** (kernels/pos_encoding.py, layers/rotary_embedding.py)
- **전 커널 Python 래퍼 테스트 13개 추가** (Activation 5개, RMSNorm 4개, RoPE 4개)
- **벤치마크에 PyTorch inplace 최적화 비교 추가** (Non-Fused vs Fused 체계적 측정)

---

## 현재 완료 상태

| 구성 요소 | 상태 | 핵심 파일 |
|-----------|------|-----------|
| CUDA 인프라 (메모리 매크로, 타입 디스패치) | ✅ 완료 | `cuda_compat.h`, `dispatch_utils.h` |
| CUB 호환 헤더 | ✅ 완료 | `cub_compat.h` |
| RoPE (Rotary Position Embedding) | ✅ 완료 | `pos_encoding_kernels.cuh/.cu` |
| RMSNorm + Fused Add+RMSNorm | ✅ 완료 | `layernorm_kernels.cuh/.cu` |
| Activation (SiLU, Fused SiLU+Mul) | ✅ 완료 | `activation_kernels.cuh/.cu` |
| PyTorch 바인딩 | ✅ 완료 | `torch_bindings.cpp` |
| 빌드 시스템 | ✅ 완료 | `setup.py`, `pyproject.toml` |
| Python 커널 래퍼 | ✅ 완료 (RoPE, RMSNorm, Activation) | `lightvllm/kernels/` |
| Python 레이어 | ✅ 완료 (RoPE, RMSNorm, Activation) | `lightvllm/layers/` |
| Attention | ⏳ 미완 | `lightvllm/attention/` |
| LLaMA 모델 통합 | ⏳ 미완 | `lightvllm/models/llama.py` |

---

## 다음 예정 작업

1. **레이어 구현** — Linear, MLP Python 레이어
2. **Attention** — naive Self-Attention 구현
3. **LLaMA 모델 조립** — 전체 파이프라인 연결 + HuggingFace 검증
