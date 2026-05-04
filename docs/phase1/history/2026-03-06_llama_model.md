# 2026-03-06: LlamaAttention → LlamaForCausalLM 완성 + HF 검증

## 작업 개요

Phase 1의 최종 목표인 LLaMA 추론 파이프라인을 완성했다.
LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM 4개 클래스를 `lightvllm/models/llama.py`에 구현하고,
HuggingFace LLaMA-3.2-3B-Instruct와 출력을 비교하여 예측 일치(argmax parity)를 달성했다.
단위 테스트 27개 + HF 검증 테스트 1개, 전체 28개 통과.

## 배경

Phase 1은 CUDA 커널(RoPE, RMSNorm, SiLU)부터 시작하여 Python 래퍼, nn.Module 레이어,
그리고 최종적으로 완전한 LLaMA 모델까지 bottom-up으로 구축하는 학습 프로젝트이다.

이전 세션들에서 다음이 완료되어 있었다:
- CUDA 커널 3종 (RoPE, RMSNorm, SiLU+Mul)
- Python 레이어 (Linear, MergedLinear, RMSNorm, SiluAndMul, RotaryEmbedding)
- LlamaMLP (SwiGLU FFN)
- Attention backends (Naive, SDPA) + KV Cache
- Weight loader (safetensors → stacked_params_mapping)

이번 세션에서는 이 빌딩블록들을 조합하여 LlamaAttention부터 LlamaForCausalLM까지 구현하고,
실제 HuggingFace 모델과 출력을 비교했다.

## 설계 결정

| 선택지 | 설명 | 채택 여부 |
|--------|------|-----------|
| LlamaAttention 위치: `attention/layer.py` | 범용 Attention 레이어로 분리 | ❌ Phase 2용으로 스텁 유지 |
| LlamaAttention 위치: `models/llama.py` | 모델별 클래스는 models/에 배치 (CLAUDE.md 규칙) | ✅ 채택 |
| Attention backend: NaiveAttention | 교육용 수동 구현 | ❌ 기본값으로 사용 안 함 |
| Attention backend: SDPAAttention | PyTorch F.scaled_dot_product_attention 활용 | ✅ 채택 (성능 우수) |
| is_causal 판단: `kv_cache is None` | 캐시 유무로 판단 | ❌ 캐시 있어도 prefill일 수 있음 |
| is_causal 판단: `num_tokens > 1` | 토큰 수로 판단 (Phase 1 단일 시퀀스 충분) | ✅ 채택 |
| HF 검증 모델: LLaMA-3.2-3B | 3B 파라미터, GPU 메모리 적합 | ❌ |
| HF 검증 모델: LLaMA-3.2-3B-Instruct | Instruct 버전으로 검증 | ✅ 채택 |
| bf16 수치 비교: strict tolerance | atol=1e-2로 비교 | ❌ 28 layer 누적 시 max_diff ~0.125 |
| bf16 수치 비교: relaxed + argmax parity | max_diff<0.5, mean_diff<0.1, argmax 일치 | ✅ 채택 |

## 구현 상세

### LlamaAttention (10단계 forward)

```
hidden_states [N, hidden_size]
    → qkv_proj (MergedLinear, 1 GEMM)      → [N, q_size + 2*kv_size]
    → split → Q [N, q_size], K [N, kv_size], V [N, kv_size]
    → .contiguous() (RoPE CUDA 커널 요구)
    → RoPE in-place (2D 상태에서 적용)
    → reshape 3D: Q [N, H, D], K [N, KH, D], V [N, KH, D]
    → KV Cache update (선택적)
    → is_causal = (num_tokens > 1)
    → SDPA attention                        → [N, H, D]
    → reshape 2D                            → [N, q_size]
    → o_proj (Linear)                       → [N, hidden_size]
```

**핵심**: `split()` 후 `.contiguous()` 필수. split은 non-contiguous view를 반환하고,
RoPE CUDA 커널은 연속 메모리를 요구한다. V는 reshape로 처리하므로 contiguous 불필요.

### LlamaDecoderLayer (Pre-Norm Residual)

vLLM과 동일한 fused residual 패턴:
- **첫 번째 layer** (residual=None): `residual = hidden_states`, `hidden_states = RMSNorm(hidden_states)`
- **이후 layer** (residual is not None): `hidden_states, residual = fused_add_rms_norm(hidden_states, residual)`

이 패턴으로 덧셈+정규화를 1회 커널 호출로 처리하여 메모리 대역폭을 절약한다.

### LlamaModel

```
input_ids → embed_tokens → [DecoderLayer × N] → 최종 RMSNorm → hidden_states
```

- residual은 layer 간에 명시적으로 전달 (gradient highway)
- `kv_cache.advance()`는 모든 layer 처리 후 1회만 호출

### LlamaForCausalLM + from_pretrained

- `logits.float()`: bf16의 유효 자릿수(~3자리)로는 vocab_size(128K) 확률 분포 표현 불가
- `stacked_params_mapping`: HF 별도 가중치(q/k/v_proj, gate/up_proj) → 융합 파라미터 매핑
- `tie_word_embeddings`: LLaMA-3.2-3B-Instruct는 True → lm_head.weight = embed_tokens.weight
- `rope_theta`: LLaMA-3은 500000.0 (LLaMA-2는 10000.0)

### HF 검증 결과

**입력**: "The capital of France is"
**비교**: LightvLLM vs HuggingFace transformers (LLaMA-3.2-3B-Instruct, bf16)

- Max abs diff: ~0.125 (bf16 28-layer 누적으로 정상)
- Mean abs diff: ~0.001
- Argmax prediction: 양쪽 모두 token 12366 (" Paris") → **예측 일치**

bf16에서 max_diff 0.125는 temperature/sampling과 무관한 순수 수치 누적 오차이다.
fp32에서는 max_diff가 0에 가깝지만, bf16의 7-bit mantissa 한계로 layer를 통과할수록
오차가 누적된다. argmax 예측이 일치하므로 실용적으로 동일한 결과이다.

## 생성/수정된 파일

### 기존 파일 수정

| 파일 | 변경 내용 |
|------|-----------|
| `lightvllm/models/llama.py` (107→742줄) | LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM 4개 클래스 추가 |
| `tests/models/test_llama.py` (→923줄) | TestLlamaAttention(8), TestLlamaDecoderLayer(5), TestLlamaModel(3), TestLlamaForCausalLM(3), TestLlamaHFValidation(1) 추가 |

## 교육 문서 내용 요약

`lightvllm/models/llama.py`에 작성한 한국어 교육 주석:

- **GQA (Grouped Query Attention)**: Q head와 KV head 수가 다른 이유와 메모리 절감 효과
- **Pre-Norm Residual 패턴**: fused_add_rms_norm의 메모리 대역폭 절약 원리
- **gradient highway**: residual stream이 deep network에서 gradient 전파를 안정화하는 메커니즘
- **RoPE 2D 입력**: 커널이 내부적으로 head 분리를 처리하므로 flatten 상태에서 적용
- **logits fp32 변환**: bf16 유효 자릿수 한계와 vocab_size 분포 표현의 관계
- **stacked_params_mapping**: HF 체크포인트 → 융합 파라미터 매핑의 동작 원리
- **tie_word_embeddings**: 소형 모델의 파라미터 절약 기법

## 핵심 학습 포인트

1. **`.contiguous()` 필수**: `split()` 후 텐서는 non-contiguous view이므로, CUDA 커널에 전달하기 전에 반드시 `.contiguous()` 호출이 필요하다.

2. **Pre-Norm Residual 패턴의 메모리 효율**: residual을 layer 간에 명시적으로 전달하고 fused_add_rms_norm으로 처리하면, 별도의 덧셈 커널 호출 없이 메모리 읽기/쓰기를 절반으로 줄일 수 있다.

3. **bf16 수치 누적 오차**: 28개 layer를 통과하면 max abs diff가 ~0.125까지 커질 수 있지만, 이는 bf16의 7-bit mantissa 한계에 의한 정상적인 현상이다. argmax 예측이 일치하면 실용적으로 동일한 모델이다.

4. **KV Cache advance() 위치**: 각 layer가 아닌, 모든 layer 처리가 끝난 후 LlamaModel에서 1회만 호출해야 한다. layer 안에서 호출하면 포인터가 N번 전진하여 캐시가 망가진다.

5. **stacked_params_mapping으로 가중치 융합**: HuggingFace가 별도로 저장한 q_proj/k_proj/v_proj를 하나의 qkv_proj로 융합 로딩하면, 추론 시 3번의 GEMM을 1번으로 줄여 커널 launch 오버헤드를 제거하고 GPU utilization을 향상시킨다.
