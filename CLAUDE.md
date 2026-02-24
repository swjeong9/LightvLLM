# LightvLLM 프로젝트 규칙

vLLM을 참고한 교육용 LLM 추론 엔진. CUDA 커널부터 모델 서빙까지 단계별 구현.

## 빌드 & 테스트

```bash
# CUDA Extension 빌드
uv run python setup.py build_ext --inplace

# Python 테스트
uv run pytest tests/ -v                          # 전체
uv run pytest tests/kernels/test_activation.py -v # 특정 파일
uv run pytest tests/ -k "silu" -v                 # 키워드

# C++ Low-level 테스트 (PyTorch 없이 커널 직접 테스트)
nvcc -O2 -std=c++17 -I csrc tests/test_xxx_kernel.cu -o tests/test_xxx_kernel
./tests/test_xxx_kernel
```

## 아키텍처: 3-layer 래퍼 구조

모든 커널은 동일한 3단 구조를 따른다:

```
csrc/xxx_kernels.cuh      → 순수 CUDA 커널 (PyTorch 의존성 없음)
csrc/xxx_kernels.cu       → PyTorch wrapper (CUDAGuard, 타입 디스패치)
lightvllm/kernels/xxx.py  → Python 함수 래퍼 (출력 텐서 자동 생성)
lightvllm/layers/xxx.py   → nn.Module 클래스 (파라미터 관리, model.state_dict())
tests/test_xxx_kernel.cu  → C++ Low-level 테스트
tests/kernels/test_xxx.py → Python 통합 테스트
```

새 커널 추가 시: `csrc/torch_bindings.cpp`에 바인딩 추가, `setup.py`에 소스 추가.

## 코드 스타일

- `.cu/.cuh` 파일: **한국어 교육 주석** (수학적 원리, 설계 이유 상세 설명)
- `.py` 파일: **영어 docstring**, 변수/함수는 snake_case
- CUDA 커널: `lightvllm` 네임스페이스, `VLLM_DISPATCH_FLOATING_TYPES` 매크로로 dtype 디스패치
- 지원 dtype: float32, float16, bfloat16

## 테스트 컨벤션

- **참조 구현은 GPU에서 동일 dtype으로** 실행 (CPU bf16은 내부적으로 float32 변환되어 결과가 달라짐)
- **허용 오차**: float32(atol=1e-5), float16(atol=1e-3), bfloat16(atol=1e-2). 리덕션 없는 element-wise 연산은 더 타이트하게 가능
- **벤치마크**: PyTorch 최적화 옵션(inplace 등)도 포함하여 공정 비교. Non-Fused vs Fused 체계적 비교
- **테스트 클래스**: `TestXxx` (커널 테스트), `TestXxxWrappers` (래퍼+Module 테스트)

## Git 컨벤션

- 브랜치: `phase1` (현재), `main` (안정)
- 커밋 메시지: `complete activation(SiLU)`, `start layernorm study` 형식

## 현재 진행 상태

- 완료/미완 작업 목록: `docs/phase1/PLAN.md`
- 작업 타임라인: `docs/phase1/history/OVERVIEW.md`
- 전체 로드맵: `docs/LEARNING_ROADMAP.md`
- vLLM 참조: `vLLM/csrc/` (git submodule)
