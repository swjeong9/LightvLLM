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

### 1. `csrc/cuda_compat.h`
- GPU 메모리 계층 구조 설명 (레지스터 → 공유메모리 → L1 → L2 → 글로벌)
- `VLLM_LDG` 매크로: 텍스처 캐시를 통한 읽기 전용 메모리 접근
- `WARP_SIZE` 상수: 워프 크기 (32)

### 2. `csrc/dispatch_utils.h`
- dtype별 커널 디스패치 설명 (float32, float16, bfloat16)
- C++ 템플릿과 런타임 타입 문제 해결
- `VLLM_DISPATCH_FLOATING_TYPES` 매크로

### 3. `.vscode/c_cpp_properties.json`
- VS Code IntelliSense 설정
- PyTorch, CUDA 헤더 경로 추가

---

## 다음 작업: RoPE (Rotary Position Embedding)

### 구현 순서

1. **`csrc/pos_encoding_kernels.cu`** - RoPE CUDA 커널
   - RoPE 수학적 원리 (2D 회전 변환)
   - Query, Key에 위치 정보 인코딩

2. **`csrc/torch_bindings.cpp`** - PyTorch 바인딩

3. **Low-level C++ 테스트** - PyTorch 없이 커널 직접 테스트

4. **컴파일 및 테스트**

---

## 이후 작업 (예정)

| 순서 | 파일 | 내용 |
|------|------|------|
| 3 | `layernorm_kernels.cu` | RMSNorm |
| 4 | `activation_kernels.cu` | SiLU + Mul |
| 5 | Attention | Self-Attention |
| 6 | Python Integration | 전체 파이프라인 연결 |

---

## 참조

- vLLM 원본: `/home/ubuntu/LightvLLM/vLLM/csrc/`
- 학습 로드맵: `/home/ubuntu/LightvLLM/docs/LEARNING_ROADMAP.md`
