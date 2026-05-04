# 2026-02-24: Linear 레이어 + LlamaMLP 코드 작성

## 작업 개요

Phase 1.3의 레이어 구현 단계로, **Linear**, **MergedLinear**, **LlamaMLP** 코드를 작성하였다.
Linear/MergedLinear는 학습 완료, LlamaMLP는 코드 작성만 완료되었으며 학습이 필요한 상태이다.

---

## 배경

### 왜 Linear에 CUDA 커널이 없는가?

Linear 연산(행렬 곱셈, GEMM)은 **compute-bound** 연산이다.
SiLU, RMSNorm 같은 element-wise/reduction 연산은 memory-bound이므로
커스텀 CUDA 커널로 메모리 접근을 최적화하면 큰 효과를 본다.

하지만 GEMM은 이미 cuBLAS가 최고 수준으로 최적화되어 있으며,
PyTorch의 `F.linear()`이 내부적으로 cuBLAS를 호출한다.
따라서 커스텀 커널 없이 `F.linear()`을 그대로 사용한다.

### MergedLinear: GEMM 융합

LLaMA MLP의 gate_proj와 up_proj는 같은 입력(hidden_states)에 대해
별도 가중치로 별도 GEMM을 수행한다:

```
gate = F.linear(x, W_gate)   # GEMM 1
up   = F.linear(x, W_up)     # GEMM 2
```

MergedLinear는 두 가중치를 하나로 합쳐 1회 GEMM으로 처리한다:

```
gate_up = F.linear(x, W_merged)  # GEMM 1회 (W_merged = [W_gate; W_up])
```

커널 launch 오버헤드 제거 + GPU utilization 향상 효과.

### SwiGLU 구조

LLaMA MLP는 SwiGLU (Shazeer, 2020)를 사용한다:

```
output = down_proj(silu(gate_proj(x)) * up_proj(x))
```

기존 FFN의 `ReLU(xW1)W2` 대비 gate 메커니즘을 추가하여 학습 품질 향상.

---

## 설계 결정

| 결정 사항 | 선택 | 이유 |
|-----------|------|------|
| weight 관리 | nn.Parameter 직접 생성 | torch.nn.Linear 대신, weight_loader와 통일된 로딩 인터페이스 |
| MergedLinear | output_sizes 리스트 | gate+up (2 shard), Q+K+V (3 shard) 모두 지원 |
| shard_id | int(0,1) 또는 str("q","k","v") | MLP는 정수, Attention은 문자열로 가독성 확보 |
| SHARD_GATE, SHARD_UP | 상수 분리 | 매직 넘버 방지, 코드 가독성 |
| LlamaMLP 위치 | `models/llama.py` | CLAUDE.md 규칙: "MLP 구성 등 모델 아키텍처는 models/에" |
| bias | False 기본값 | LLaMA는 bias 미사용 |

---

## 구현 상세

### Linear (`lightvllm/layers/linear.py`)

```python
class Linear(nn.Module):
    # F.linear() 래핑 + weight nn.Parameter 관리
    # torch.nn.Linear 대신 사용하여 weight 로딩 방식 통일
```

### MergedLinear (`lightvllm/layers/linear.py`)

```python
class MergedLinear(nn.Module):
    # output_sizes로 shard별 크기 정의
    # weight: [sum(output_sizes), input_size]
    # weight_loader(param, loaded_weight, shard_id):
    #   offset 계산 → param.data[offset:offset+size] = loaded_weight
```

### LlamaMLP (`lightvllm/models/llama.py`)

```
x: [num_tokens, hidden_size]
    |
gate_up_proj (MergedLinear)  ->  [num_tokens, 2 * intermediate_size]
    |
SiluAndMul (CUDA 커널)       ->  [num_tokens, intermediate_size]
    |
down_proj (Linear)           ->  [num_tokens, hidden_size]
```

---

## 생성/수정된 파일

| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `lightvllm/layers/linear.py` | 175 | Linear, MergedLinear, weight_loader |
| `lightvllm/models/llama.py` | 107 | LlamaMLP (SwiGLU) |
| `tests/layers/test_linear.py` | ~200 | Linear 20개 테스트 |
| `tests/models/test_llama.py` | ~170 | LlamaMLP 7개 테스트 + LLaMA 모델 2개 스텁 |

---

## 학습 상태

- **Linear, MergedLinear**: 학습 완료. cuBLAS GEMM 활용, weight_loader shard 메커니즘 이해 완료.
- **LlamaMLP**: 코드 작성 완료, **학습 필요**. SwiGLU 구조, MergedLinear 활용, 테스트 검증 미실시.
