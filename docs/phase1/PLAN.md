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

### 10. CUB 호환 헤더 — `csrc/cub_compat.h` (26줄) ✅

CUB 2.0.8 이상에서 `cub::Sum()`이 `cuda::std::plus<>`로 변경된 호환성 문제 해결:

- **`CubAddOp`**: CUB 버전에 따라 `cub::Sum()` 또는 `cuda::std::plus<void>()` 자동 선택

### 11. RMSNorm 순수 CUDA 헤더 — `csrc/layernorm_kernels.cuh` (245줄) ✅

PyTorch 의존성 없이 사용 가능한 RMSNorm 커널 헤더. CUB `BlockReduce`를 사용한 효율적인 병렬 리덕션:

- **`rms_norm_kernel<scalar_t>()`**: Two-Pass 알고리즘. Pass 1에서 fp32로 제곱합 누적 + BlockReduce + rsqrt 계산 및 `__shared__` 브로드캐스트, Pass 2에서 정규화 적용. Grid=(num_tokens), Block=(block_size).
- **`fused_add_rms_norm_kernel<scalar_t>()`**: 잔차 덧셈 + RMSNorm을 하나의 커널에서 수행. Pass 1에서 `residual += input`과 제곱합 누적을 동시에 처리하여 글로벌 메모리 읽기 1회 절약. 두 텐서 모두 in-place 수정.

### 12. RMSNorm PyTorch 연동 — `csrc/layernorm_kernels.cu` (249줄) ✅

RMSNorm의 수학적 원리(LLaMA Pre-Norm 구조, LayerNorm 비교, Fused 커널의 메모리 대역폭 분석)에 대한 상세 교육 문서와 함께 PyTorch 진입점 함수 구현:

- **`rms_norm()`**: 출력 텐서에 정규화 결과 저장. 블록 크기 휴리스틱(토큰 수 < 256이면 1024, 아니면 256) 적용.
- **`fused_add_rms_norm()`**: input과 residual을 in-place로 수정. `VLLM_DISPATCH_FLOATING_TYPES`로 dtype 디스패치.

### 13. Python 래퍼 및 nn.Module ✅

- **`lightvllm/kernels/layernorm.py`** (61줄): `rms_norm()`, `fused_add_rms_norm()` Python 래퍼 함수
- **`lightvllm/layers/normalization.py`** (70줄): `RMSNorm` nn.Module 클래스. forward에서 residual 유무에 따라 `rms_norm` / `fused_add_rms_norm` 자동 분기.

### 14. RMSNorm 테스트 ✅

**C++ Low-level 테스트** — `tests/test_layernorm_kernel.cu`:
- Test 1 — RMSNorm 기본 정확성 (CPU 참조 vs GPU)
- Test 2 — Fused Add+RMSNorm 정확성 (CPU 참조 재사용)
- Test 3 — 영벡터 입력 안정성 (NaN/Inf 없음)
- Test 4 — Fused vs Non-Fused 성능 비교 (1.56x 속도 향상)

**Python 통합 테스트** — `tests/kernels/test_layernorm.py` (16개 테스트):
- 기본 정확성 (bf16), Fused 커널 정확성, residual bit-exact 검증
- 단위 가중치 수학적 성질, 영벡터 안정성
- half dtype별 (fp16, bf16) 검증, 대규모 텐서 (1024×4096)
- 다양한 hidden_size (128~8192) 파라미터화
- HuggingFace `LlamaRMSNorm`과의 정확성 비교
- 성능 벤치마크: HuggingFace vs PyTorch 참조 vs CUDA 커널 vs Non-Fused vs Fused

### 15. Activation 순수 CUDA 헤더 — `csrc/activation_kernels.cuh` (412줄) ✅

128-bit 벡터화를 도입한 SiLU 활성화 커널 헤더:

- **`silu_kernel<T>()`**: SiLU 활성화 함수. `__device__ __forceinline__`, fp32 중간 연산으로 수치 안정성 확보.
- **`activation_kernel<scalar_t, ACT_FN>()`**: element-wise 활성화 커널. 128-bit `int4` 벡터화 + 정렬 체크 + scalar fallback 이중 경로.
- **`act_and_mul_kernel<scalar_t, ACT_FN>()`**: Fused silu_and_mul 커널. 입력 `[..., 2*d]` → 출력 `[..., d]`. `silu(gate)` 결과를 레지스터에 유지한 채 up과 곱셈.

### 16. Activation PyTorch 연동 — `csrc/activation_kernels.cu` (517줄) ✅

활성화 함수 발전사(ReLU→GELU→SiLU), SwiGLU 구조, 메모리 대역폭 분석에 대한 상세 교육 문서와 함께 PyTorch 진입점 구현:

- **`silu()`**: element-wise SiLU 적용. `VLLM_DISPATCH_FLOATING_TYPES`, `CUDAGuard` 사용.
- **`silu_and_mul()`**: Fused gate 연산. 입력의 전반부에 SiLU 적용 후 후반부와 곱셈.

### 17. Activation Python 래퍼 및 nn.Module ✅

- **`lightvllm/kernels/activation.py`** (47줄): `silu()`, `silu_and_mul()` Python 래퍼 함수
- **`lightvllm/layers/activation.py`** (33줄): `SiluAndMul` nn.Module 클래스 (학습 파라미터 없음)

### 18. RoPE Python 래퍼 완성 ✅

기존 빈 스텁을 완전 구현:

- **`lightvllm/kernels/pos_encoding.py`** (34줄): `rotary_embedding()` Python 래퍼 함수
- **`lightvllm/layers/rotary_embedding.py`** (92줄): `RotaryEmbedding` nn.Module (cos/sin 캐시 생성 + forward)

### 19. Activation 테스트 ✅

**C++ Low-level 테스트** — `tests/test_activation_kernel.cu` (531줄):
- Test 1 — SiLU 기본 정확성 (GPU vs CPU 참조)
- Test 2 — silu_and_mul 기본 정확성
- Test 3 — SiLU 수학적 성질 (silu(0)=0, 하한값 ≈ -0.278)
- Test 4 — 수치 안정성 (극단값에서 NaN/Inf 없음)
- Test 5 — Non-Fused vs Fused 성능 벤치마크 (1.49x 속도 향상)

**Python 통합 테스트** — `tests/kernels/test_activation.py` (31개 테스트):
- SiLU: 기본 정확성, zero, dtype별, 큰 값, 다양한 크기, PyTorch Module 비교, 성능 벤치마크
- silu_and_mul: 기본 정확성, shape 검증, dtype별, 대규모, 3D 입력, zero gate, unit up, 성능 벤치마크
- 래퍼: silu/silu_and_mul 래퍼, 3D 래퍼, SiluAndMul Module

### 20. 전 커널 Python 래퍼 테스트 추가 ✅

기존 테스트 파일에 래퍼 계층 검증 테스트 추가 (13개):

- `tests/kernels/test_activation.py`: `TestActivationWrappers` (5개)
- `tests/kernels/test_layernorm.py`: `TestLayerNormWrappers` (4개)
- `tests/kernels/test_rope.py`: `TestRoPEWrappers` (4개)

### 21. Linear 레이어 — `lightvllm/layers/linear.py` (128줄) ✅

LLaMA 모델의 모든 프로젝션(QKV, Output, Gate, Up, Down)의 기본 빌딩블록:

- **`Linear`**: `F.linear` + 직접 weight/bias `nn.Parameter` 관리. `torch.nn.Linear` 대신 사용하여 weight 로딩 방식을 통일. LLaMA는 bias=False 기본.
- **`MergedLinear`**: 여러 별도 프로젝션을 하나의 큰 weight 행렬로 융합하여 GEMM 1회로 처리.
  - `output_sizes` 리스트로 각 shard 크기 정의 (예: gate_up_proj=[intermediate, intermediate], qkv_proj=[q_size, kv_size, kv_size])
  - **`weight_loader(param, loaded_weight, shard_id)`**: HuggingFace 체크포인트의 별도 가중치(q_proj, k_proj, v_proj 등)를 융합 파라미터의 올바른 오프셋에 로딩
  - shard_id: int(0=gate, 1=up) 또는 str("q", "k", "v")
- **왜 Fuse하는가**: 2~3번의 별도 CUDA 커널 launch → 1번의 큰 GEMM. GPU utilization 향상, 커널 launch 오버헤드 제거

**Python 통합 테스트** — `tests/layers/test_linear.py` (20개 테스트):
- Linear: 기본 forward, 다양한 shape, bias 유무, dtype별, weight shape, 3D 입력
- MergedLinear: gate_up 2 shard, qkv 3 shard, output shape, weight_loader 정확성(gate_up/qkv), dtype, offsets, bias

---

## 다음 작업 (예정) — Phase 1.3 남은 세션

### Session 2: LlamaMLP

**파일: `lightvllm/layers/mlp.py`** (~60줄)

`LlamaMLP` nn.Module — SwiGLU 구조의 Feed-Forward Network:

```
forward(x):
    gate_up = gate_up_proj(x)        # MergedLinear → [num_tokens, 2 * intermediate_size]
    activated = silu_and_mul(gate_up) # SiluAndMul → [num_tokens, intermediate_size]
    output = down_proj(activated)     # Linear → [num_tokens, hidden_size]
    return output
```

**구성 요소:**
- `gate_up_proj`: `MergedLinear(hidden_size, [intermediate_size, intermediate_size])`
- `down_proj`: `Linear(intermediate_size, hidden_size)`
- `act_fn`: `SiluAndMul()` (기존 `lightvllm/layers/activation.py` 재사용)

**학습 포인트:**
- SwiGLU 구조: `silu(gate) * up`이 ReLU보다 좋은 이유 (Shazeer 2020)
- Fused gate_up_proj: 1 GEMM이 2 GEMM보다 빠른 이유 (GPU occupancy, kernel launch overhead)
- LLaMA 크기: hidden=4096, intermediate=11008 (≈ 2.7× hidden, 8의 배수로 정렬)

**테스트 (`tests/layers/test_mlp.py`):**
- output shape 검증, 별도 gate/up/down과 fused 결과 비교, dtype, LLaMA-7B 크기, zero 입력

**재사용:** `lightvllm/layers/linear.py`, `lightvllm/layers/activation.py`

---

### Session 3: Attention Backend + KV Cache

#### 3-1. AttentionBackend ABC (`lightvllm/attention/backends/base.py`, ~40줄)

```python
class AttentionBackend(ABC):
    @abstractmethod
    def forward(self, query, key, value, num_heads, num_kv_heads, head_dim, is_causal=True):
        """
        query:  [num_tokens, num_heads * head_dim]
        key:    [seq_len, num_kv_heads * head_dim]
        value:  [seq_len, num_kv_heads * head_dim]
        Returns: [num_tokens, num_heads * head_dim]
        """
        ...
```

#### 3-2. NaiveAttention (`lightvllm/attention/backends/naive.py`, ~80줄)

수동 `softmax(Q @ K^T / √d_k) @ V` 구현:
- GQA: KV heads를 Q heads 수만큼 `unsqueeze(2).expand().reshape()` 패턴
- Causal mask: `torch.triu(full(-inf), diagonal=1)` — prefill 시에만 적용
- Softmax fp32 계산 후 원래 dtype 복원 (수치 안정성, HF 동일 패턴)

**학습 포인트:**
- Q@K^T가 "유사도 행렬"인 이유 (내적 = 벡터 간 코사인 유사도 * 크기)
- Causal masking: 미래 토큰 참조 방지 (autoregressive decoding 전제)
- GQA: KV heads < Q heads일 때 메모리 절감 (LLaMA-2 70B: 8 KV, 64 Q)
- Softmax fp32: half precision에서 overflow 방지 (exp(x)에서 x가 크면 inf)

#### 3-3. SDPAAttention (`lightvllm/attention/backends/sdpa.py`, ~50줄)

`F.scaled_dot_product_attention` 래핑:
- GQA expand 후 `[1, num_heads, seq_len, head_dim]` shape으로 변환
- `is_causal` 플래그 직접 전달
- PyTorch 2.0+에서 FlashAttention/Memory-Efficient 자동 선택

**학습 포인트:**
- SDPA 내부에서 FlashAttention vs Memory-Efficient Attention 선택 기준
- Naive O(N²) 메모리 vs FlashAttention O(N) 메모리
- Backend 추상화 패턴: 같은 인터페이스, 다른 구현

#### 3-4. KVCache (`lightvllm/attention/kv_cache.py`, ~80줄)

```python
class KVCache:
    """[num_layers, max_seq_len, num_kv_heads, head_dim] contiguous buffer"""
    def update(layer_idx, key, value) -> (all_keys, all_values)
    def advance(num_tokens) -> None
    @property current_seq_len -> int
```

**학습 포인트:**
- Prefill vs Decode: prefill에서 전체 시퀀스 캐시, decode에서 1토큰씩 추가
- Q는 캐시 안하는 이유: Q는 현재 토큰만 필요, K/V는 이전 전체 필요
- Contiguous vs Paged: 이 단계는 contiguous (Phase 2에서 Paged Attention으로 발전)

**테스트 (`tests/attention/test_attention.py`):**
- NaiveAttention: 기본, causal mask, GQA, single-token decode, dtype
- SDPAAttention: Naive와 비교하여 동일 결과 확인
- KVCache: 기본, incremental, multi-layer, attention 결합

---

### Session 4: LlamaAttention (통합 레이어)

**파일: `lightvllm/attention/layer.py`** (~120줄)

```python
class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, ...):
        self.qkv_proj = MergedLinear(hidden_size, [q_size, kv_size, kv_size])
        self.o_proj = Linear(q_size, hidden_size)
        self.rotary_emb = RotaryEmbedding(head_size=head_dim, ...)
        self.attn_backend = NaiveAttention()

    def forward(self, positions, hidden_states, kv_cache=None, layer_idx=0):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = q.contiguous(), k.contiguous()  # split 후 contiguous 필수!
        self.rotary_emb(positions, q, k)        # in-place RoPE

        if kv_cache is not None:
            k_cached, v_cached = kv_cache.update(layer_idx, k_3d, v_3d)
            k, v = k_cached.flatten(), v_cached.flatten()

        attn_out = self.attn_backend.forward(q, k, v, ..., is_causal=(kv_cache is None))
        return self.o_proj(attn_out)
```

**핵심 설계:**
- `is_causal=(kv_cache is None)`: prefill에서만 causal mask 적용. Decode 시 전체 캐시에 attend
- `.contiguous()` 필수: `split()`이 non-contiguous view 반환, CUDA 커널은 contiguous 필요
- RoPE 적용 위치: QKV 프로젝션 후 → Attention 계산 전

**테스트 (`tests/layers/test_attention_layer.py`):**
- output shape, KV cache prefill→decode, Naive vs SDPA backend 비교, dtype

**재사용:** `linear.py`, `rotary_embedding.py`, `naive.py`, `kv_cache.py`

---

## Phase 1.4 (모델 조립, 별도 계획)

| 순서 | 작업 | 파일 | 내용 |
|------|------|------|------|
| 1 | LlamaDecoderLayer | `lightvllm/models/llama.py` | Attention + MLP + RMSNorm 결합, fused residual |
| 2 | LlamaModel | `lightvllm/models/llama.py` | Embedding → DecoderLayer × N → Final Norm |
| 3 | LlamaForCausalLM | `lightvllm/models/llama.py` | LlamaModel + LM Head, from_pretrained() |
| 4 | Weight Loader | `lightvllm/models/loader.py` | HF safetensors → stacked_params_mapping |
| 5 | End-to-End Tests | `tests/models/test_llama.py` | HuggingFace와 출력 비교 검증 |

---

## 참조

- vLLM 원본: `/home/ubuntu/LightvLLM/vLLM/csrc/`
- vLLM 레이어 참조: `vLLM/vllm/model_executor/layers/linear.py`
- vLLM 모델 참조: `vLLM/vllm/model_executor/models/llama.py`
- 학습 로드맵: `/home/ubuntu/LightvLLM/docs/LEARNING_ROADMAP.md`
