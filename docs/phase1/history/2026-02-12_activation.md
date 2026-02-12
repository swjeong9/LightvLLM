# 2026-02-12: Activation CUDA 커널 구현 + Python 래퍼 계층 완성

## 작업 개요

Phase 1의 네 번째 커널인 **SiLU**와 **Fused SiLU+Mul (silu_and_mul)**을 구현하였다.
RoPE, RMSNorm과 동일한 아키텍처 패턴(순수 CUDA 헤더 + PyTorch wrapper + Python 래퍼 + nn.Module + 테스트)을 따르며,
128-bit 벡터화를 도입하여 메모리 대역폭 활용을 최적화하였다.

추가로, 기존에 비어있던 **RoPE Python 래퍼 스텁을 구현**하고,
세 커널(Activation, RMSNorm, RoPE) 모두에 대해 **Python 래퍼 계층 테스트**를 추가하여
`_C` → `kernels/` → `layers/` 3단 래퍼 구조의 정확성을 검증하였다.

또한 모든 벤치마크에 **PyTorch inplace 최적화 비교**를 추가하여
Non-Fused vs Fused 커널 퓨전의 효과를 체계적으로 측정하였다.

---

## 배경

### SiLU (Sigmoid Linear Unit)

SiLU는 LLaMA, Mistral 등 최신 LLM에서 MLP 블록의 활성화 함수로 사용된다.

```
SiLU(x) = x × σ(x) = x / (1 + exp(-x))
```

ReLU → GELU → SiLU로 이어지는 활성화 함수 발전의 최신 단계이며,
GELU와 달리 error function 근사가 필요 없어 **정확한 closed-form 계산**이 가능하다.

### Fused silu_and_mul이 필요한 이유

LLaMA MLP는 SwiGLU 구조를 사용한다:

```
output = down_proj(silu(gate_proj(x)) × up_proj(x))
```

별도 커널 2개를 사용하면:

```
Step 1: gate_activated = silu(gate)  → gate 읽기 + silu + 쓰기   (커널 1)
Step 2: output = gate_activated × up → gate_activated 읽기 + up 읽기 + 곱셈 + 쓰기 (커널 2)
총 메모리 접근: 4회 읽기 + 2회 쓰기 = 132KB (d=11008, bf16)
```

Fused 커널은 이를 하나로 합친다:

```
output = silu(gate) × up  → gate 읽기 + up 읽기 + silu + 곱셈 + 쓰기 (커널 1)
총 메모리 접근: 2회 읽기 + 1회 쓰기 = 66KB
```

**메모리 트래픽이 절반**으로 줄어 ~1.7x 속도 향상을 달성한다.

---

## 설계 결정

### Approach B (교육적 + 실용 최적화) 채택

| 결정 사항 | 선택 | 이유 |
|-----------|------|------|
| 벡터화 | 128-bit `int4` 로드 포함 | element-wise 커널의 핵심 최적화, GPU 메모리 대역폭 학습 |
| 파일 구조 | `.cuh` 헤더 분리 | RoPE/RMSNorm과 동일 패턴, C++ 테스트에서도 재사용 |
| 범위 | SiLU + silu_and_mul만 | LLaMA는 SiLU만 사용, GELU는 나중에 추가 가능 |
| vLLM 대비 단순화 | `act_first` 파라미터 제거, 매크로 대신 인라인 | LLaMA는 activation-first만 필요 |

### 128-bit 벡터화 전략

```cpp
// 16바이트 = int4 = bf16 8개 또는 float32 4개를 한 번에 로드
constexpr int VEC_SIZE = 16 / sizeof(scalar_t);

// 정렬 확인: 포인터가 16바이트 경계에 있는지
bool aligned = is_16byte_aligned(out) && is_16byte_aligned(input);

if (aligned && d % VEC_SIZE == 0) {
    // Fast path: reinterpret_cast<int4*>로 벡터화 로드/스토어
} else {
    // Fallback: 스칼라 접근
}
```

- 스칼라 접근: bf16 2바이트씩 → 메모리 버스 활용율 6.25%
- int4 벡터화: 16바이트씩 → 메모리 버스 활용율 50~100%

---

## 구현 상세

### activation_kernel (SiLU 단독)

```
Grid: (num_tokens,)  — 각 블록이 하나의 토큰(행) 처리
Block: (min(d, 1024),) — 스레드들이 d 원소를 협력 처리

각 스레드:
  ┌─ stride 패턴으로 d 순회 (threadIdx.x, threadIdx.x + blockDim.x, ...)
  ├─ int4 벡터화: 8개 bf16 원소를 한 번에 로드
  ├─ #pragma unroll로 8개 원소에 silu() 적용
  └─ int4로 한 번에 스토어
```

### act_and_mul_kernel (Fused silu_and_mul)

```
입력: [num_tokens, 2*d] — 전반부 gate, 후반부 up
출력: [num_tokens, d]

각 스레드:
  ┌─ stride 패턴으로 d 순회
  ├─ gate[i] = input[token, i]          ← 읽기
  ├─ up[i]   = input[token, d + i]      ← 읽기
  ├─ result  = silu(gate[i]) × up[i]    ← 레지스터에서 계산
  └─ output[token, i] = result          ← 쓰기
```

핵심: `silu(gate)` 결과를 글로벌 메모리에 쓰지 않고 **레지스터에 유지**한 채 곱셈을 수행한다.

---

## 생성/수정된 파일

### Activation 커널 (신규)

| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `csrc/activation_kernels.cuh` | 412 | 순수 CUDA 커널 + 128-bit 벡터화 + 한국어 교육 주석 |
| `csrc/activation_kernels.cu` | 517 | PyTorch wrapper + 활성화 함수 발전사 + 메모리 대역폭 분석 교육 주석 |
| `lightvllm/kernels/activation.py` | 47 | Python 커널 래퍼 (`silu()`, `silu_and_mul()`) |
| `lightvllm/layers/activation.py` | 33 | `SiluAndMul` nn.Module 클래스 |
| `tests/test_activation_kernel.cu` | 531 | C++ Low-level 테스트 5개 (정확성, 수학적 성질, 안정성, Non-Fused vs Fused 벤치마크) |
| `tests/kernels/test_activation.py` | 570 | Python 통합 테스트 31개 (정확성, 성능, 래퍼 검증) |

### RoPE Python 래퍼 (스텁 → 구현)

| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `lightvllm/kernels/pos_encoding.py` | 34 | `rotary_embedding()` Python 래퍼 함수 (기존 스텁 교체) |
| `lightvllm/layers/rotary_embedding.py` | 92 | `RotaryEmbedding` nn.Module (cos/sin 캐시 생성 + forward) |

### 기존 파일 수정

| 파일 | 변경 내용 |
|------|-----------|
| `csrc/torch_bindings.cpp` | `silu`, `silu_and_mul` 전방 선언 + pybind11 바인딩 활성화 |
| `setup.py` | sources 리스트에 `"csrc/activation_kernels.cu"` 추가 |
| `tests/kernels/test_activation.py` | 래퍼 테스트 5개 추가 (`TestActivationWrappers`), 벤치마크에 inplace 비교 추가 |
| `tests/kernels/test_layernorm.py` | 래퍼 테스트 4개 추가 (`TestLayerNormWrappers`) |
| `tests/kernels/test_rope.py` | 래퍼 테스트 4개 추가 (`TestRoPEWrappers`) |

---

## 추가 작업: Python 래퍼 계층 완성

### 문제 발견

기존 테스트 파일들은 모두 `lightvllm._C`를 직접 호출하여 커널을 테스트하고 있었다.
`kernels/` 폴더의 Python 래퍼 함수와 `layers/` 폴더의 nn.Module 클래스가
올바르게 동작하는지 검증하는 테스트가 없었다.

또한 RoPE의 Python 래퍼(`kernels/pos_encoding.py`, `layers/rotary_embedding.py`)는
빈 스텁 상태였지만, 테스트가 `_C`를 직접 호출하므로 문제가 드러나지 않았다.

### 해결

1. **RoPE 스텁 구현**: `kernels/pos_encoding.py`와 `layers/rotary_embedding.py`를 완전 구현
2. **래퍼 테스트 추가**: 세 커널 모두에 `TestXxxWrappers` 클래스 추가

### 3-layer 아키텍처 검증

```
_C (C 바인딩) → kernels/ (Python 함수 래퍼) → layers/ (nn.Module)
```

| 계층 | 역할 | 예시 |
|------|------|------|
| `_C` | 원시 C 함수 호출 | `_C.silu(out, input)` — 출력 텐서를 직접 전달 |
| `kernels/` | 편의 래퍼 | `silu(input)` — 출력 텐서 자동 생성 후 반환 |
| `layers/` | nn.Module | `SiluAndMul()(x)` — 모델 조립용 (파라미터 관리, state_dict, .cuda()) |

---

## 추가 작업: 성능 벤치마크 개선

### PyTorch inplace 최적화 비교

PyTorch에서 제공하는 `F.silu(inplace=True)` 등의 최적화 옵션을 벤치마크에 포함하여
공정한 비교를 수행하였다.

### SiLU 단독 벤치마크 결과

```
설정: num_tokens=4096, d=11008, dtype=torch.bfloat16
[1] PyTorch F.silu():              775.2 us
[2] PyTorch F.silu(inplace=True):   750.3 us
[3] CUDA _C.silu():                768.3 us
F.silu vs inplace 속도 비율:       1.03x
F.silu vs CUDA 속도 비율:          1.01x
inplace vs CUDA 속도 비율:         0.98x
```

**분석**: 셋 다 거의 동일한 성능. SiLU는 element-wise 연산이므로 순수 메모리 대역폭 바운드이며,
PyTorch의 `F.silu()`도 내부적으로 CUDA 커널을 호출하기 때문에 차이가 없다.

### silu_and_mul 벤치마크 결과

```
설정: num_tokens=4096, d=11008, dtype=torch.bfloat16
[1] PyTorch (F.silu + mul, 할당 O):         1931.1 us
[2] PyTorch (inplace silu + mul_):          1936.3 us
[3] CUDA Non-Fused (silu + mul, 커널 2회):  1932.4 us
[4] CUDA Fused (silu_and_mul, 커널 1회):    1141.7 us
[1] PyTorch vs [4] Fused:   1.69x
[2] inplace vs [4] Fused:   1.70x
[3] Non-Fused vs [4] Fused: 1.69x
```

**분석**: [1]~[3]은 모두 ~1930 us로 동일한 성능.
PyTorch든 CUDA든 inplace든, Non-Fused 방식은 동일한 메모리 접근 패턴(커널 2회 + 중간 버퍼)이므로
성능도 동일하다. **유일한 해결책은 커널 퓨전**이며, Fused 커널이 1.69x 빠르다.

### C++ 벤치마크 결과 (Test 5)

```
설정: num_tokens=1024, d=4096
[1] CPU 참조 (silu_and_mul): ~4500 us/call
[2] GPU Non-Fused (커널 2회): ~110 us/call
[3] GPU Fused (커널 1회):     ~73 us/call
Non-Fused vs Fused 속도 향상: 1.49x
```

---

## 테스트 결과

### C++ Low-level 테스트 (5/5 통과)

```
[Test 1] SiLU 기본 정확성
  최대 절대 오차: < 1e-5 (허용: 1e-5)  → PASS

[Test 2] silu_and_mul 기본 정확성
  최대 절대 오차: < 1e-5  → PASS

[Test 3] SiLU 수학적 성질
  silu(0) == 0, silu(-1.278) ≈ -0.278, silu(100) ≈ 100  → PASS

[Test 4] 수치 안정성
  극단값 (1e38, -1e38 등)에서 NaN/Inf 없음  → PASS

[Test 5] 성능 벤치마크
  Non-Fused vs Fused: 1.49x  → PASS
```

### Python 통합 테스트

**Activation 테스트 (31/31 통과)**:

| 테스트 클래스 | 테스트 수 | 내용 |
|--------------|-----------|------|
| `TestSiLU` | 12 | 기본 정확성, zero, dtype별, 큰 값, 다양한 크기, PyTorch Module 비교, 성능 벤치마크 |
| `TestSiLUAndMul` | 10 | 기본 정확성, shape 검증, dtype별, 대규모, 다양한 크기, 3D 입력, zero gate, unit up, 성능 벤치마크 |
| `TestActivationWrappers` | 5 | silu 래퍼, silu_and_mul 래퍼, 3D 래퍼, Module, Module 3D |

**래퍼 테스트 (13개 추가, 전체 테스트에 통합)**:

| 파일 | 테스트 클래스 | 테스트 수 | 내용 |
|------|--------------|-----------|------|
| `test_activation.py` | `TestActivationWrappers` | 5 | silu/silu_and_mul 래퍼 + SiluAndMul Module |
| `test_layernorm.py` | `TestLayerNormWrappers` | 4 | rms_norm/fused_add_rms_norm 래퍼 + RMSNorm Module |
| `test_rope.py` | `TestRoPEWrappers` | 4 | rotary_embedding 래퍼 + RotaryEmbedding Module + 캐시 검증 |

### 전체 테스트 스위트

```
tests/kernels/test_activation.py    31 passed
tests/kernels/test_layernorm.py     20 passed
tests/kernels/test_rope.py          10 passed
tests/models/test_llama.py           2 passed
================================== 63 passed ===
```

기존 RoPE, RMSNorm 테스트 및 모든 기존 테스트에 영향 없음 확인.

---

## 허용 오차 참고

Activation 커널은 **리덕션이 없는 순수 element-wise 연산**이므로,
RMSNorm(CUB BlockReduce 사용)보다 오차가 작다.
주된 오차 원인은 `expf()` 구현의 미세한 차이뿐이다.

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-6 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |

RMSNorm 대비 float32 atol이 10배 타이트하다 (1e-6 vs 1e-5).

---

## 교육 문서 내용 요약

### activation_kernels.cuh (커널 헤더)

1. **활성화 함수의 역할**: 비선형성이 없으면 깊은 네트워크도 하나의 선형 변환과 동일
2. **ReLU → GELU → SiLU 발전사**: dead neuron 문제, error function 근사, self-gating
3. **128-bit 벡터화 메모리 접근**: `int4` 타입, `reinterpret_cast`, 16바이트 정렬 조건
4. **`__forceinline__` 함수 포인터 템플릿**: 컴파일타임에 활성화 함수를 플러그인
5. **`#pragma unroll`**: 루프 언롤링으로 instruction overhead 제거
6. **`__restrict__`**: 포인터 앨리어싱 없음을 보장하여 컴파일러 최적화 허용

### activation_kernels.cu (PyTorch wrapper)

1. **GELU 상세 분석**: Gaussian CDF 기반, error function, tanh 근사 vs erff() 구현
2. **SiLU/Swish 유래**: AutoML 탐색 (Ramachandran et al., 2017)에서 발견
3. **SiLU 수학적 성질**: 매끄러움, 비단조성, self-gating, 수치 안정성
4. **LLaMA MLP SwiGLU 구조**: 3-projection (gate, up, down), Shazeer (2020)
5. **Fused 커널 메모리 절약 분석**: 132KB → 66KB, ~2x throughput 향상
6. **element-wise 연산의 본질**: 연산 대비 메모리가 병목 (A100: 연산 240x 빠름)

---

## 핵심 학습 포인트

1. **커널 퓨전의 본질은 중간 버퍼 제거**이다.
   Non-Fused 방식은 PyTorch든, CUDA 커널이든, inplace든 모두 동일한 성능을 보인다.
   메모리 대역폭이 병목인 element-wise 연산에서 유일한 최적화는 중간 결과를 레지스터에 유지하는 퓨전이다.

2. **128-bit 벡터화는 메모리 바운드 커널의 핵심 최적화**이다.
   bf16 원소를 하나씩 로드하면 메모리 버스의 6.25%만 활용한다.
   `int4` (16바이트)로 8개를 한 번에 로드하면 버스 활용율이 최대 100%까지 올라간다.

3. **PyTorch의 F.silu()도 내부적으로 CUDA 커널**이다.
   따라서 `_C.silu()` vs `F.silu()`는 성능 차이가 없다.
   커스텀 CUDA 커널의 가치는 단독 연산이 아니라 **퓨전에서 나온다**.

4. **3-layer Python 래퍼 아키텍처**(`_C` → `kernels/` → `layers/`)는
   각 계층이 명확한 역할을 갖는다:
   - `_C`: 원시 C 바인딩 (출력 텐서를 호출자가 전달)
   - `kernels/`: 편의 함수 (출력 텐서 자동 생성)
   - `layers/`: nn.Module (파라미터 관리, `model.state_dict()`, `model.cuda()`, `print(model)`)
