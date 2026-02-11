/**
 * =============================================================================
 * LightvLLM LayerNorm CUDA 커널
 * =============================================================================
 *
 * 이 파일은 RMSNorm과 Fused Add+RMSNorm을 구현합니다.
 * RMSNorm은 LLaMA, Mistral 등 최신 LLM에서 정규화 레이어로 사용됩니다.
 *
 * 목차:
 * 1. RMSNorm의 역할
 * 2. RMSNorm의 수학적 원리
 * 3. Fused 커널의 필요성
 * 4. CUDA 커널 래퍼 코드
 */

/**
 * =============================================================================
 * 1. RMSNorm의 역할
 * =============================================================================
 *
 * [LLM에서 정규화가 필요한 이유]
 *
 * 딥러닝에서 레이어를 깊게 쌓으면 두 가지 문제가 발생합니다:
 *
 * 1. Internal Covariate Shift:
 *    각 레이어의 입력 분포가 학습 중에 계속 변화함
 *    → 후속 레이어가 변화하는 분포에 적응해야 하므로 학습이 불안정
 *
 * 2. Gradient Vanishing/Exploding:
 *    깊은 네트워크에서 역전파 시 gradient가 소실되거나 폭발
 *    → 특히 hidden_size가 큰 LLM에서 심각
 *
 * 정규화는 각 레이어의 출력을 일정한 스케일로 유지하여 이 문제를 완화합니다.
 *
 *
 * [LLaMA에서 RMSNorm의 위치]
 *
 * LLaMA의 각 Transformer 블록:
 *
 *   hidden_states ─┬─ RMSNorm → Attention → + ─┬─ RMSNorm → MLP → + ─→
 *                  │                           │                     │
 *                  └──── (residual) ─────────→ └──── (residual) ────→
 *
 * RMSNorm은 Attention과 MLP의 "입력"에 적용됩니다 (Pre-Norm 구조).
 * 이것은 원래 Transformer (Post-Norm)와 다른 점입니다:
 *   - Post-Norm: output = LayerNorm(x + Attention(x))
 *   - Pre-Norm:  output = x + Attention(RMSNorm(x))
 *
 * Pre-Norm이 학습 안정성이 더 좋다는 것이 실험적으로 밝혀졌습니다.
 *
 *
 * =============================================================================
 * 2. RMSNorm의 수학적 원리
 * =============================================================================
 *
 * [수식]
 *
 * 입력 벡터 x = [x_0, x_1, ..., x_{d-1}] (d = hidden_size)
 *
 * RMS(x) = sqrt(mean(x²) + ε)
 *        = sqrt((x_0² + x_1² + ... + x_{d-1}²) / d + ε)
 *
 * RMS는 Root Mean Square의 약자입니다:
 *   x²    → Square:  각 원소를 제곱
 *   mean  → Mean:    평균
 *   sqrt  → Root:    제곱근
 * 즉 "제곱의 평균의 제곱근"이고, 거꾸로 읽으면 Root-Mean-Square입니다.
 * RMSNorm은 이 RMS 값으로 입력을 나누는 정규화 방식이라서 RMS + Norm = RMSNorm.
 *
 * output = x / RMS(x) * γ
 *
 * 여기서:
 * - ε (epsilon): 수치 안정성 (보통 1e-6, 분모가 0이 되는 것을 방지)
 * - γ (weight): 학습 가능한 스케일 파라미터 [hidden_size], 초기값 = 1
 *
 *
 * [구현에서 rsqrt를 사용하는 이유]
 *
 * 수식대로 구현하면:
 *   rms = sqrt(mean(x²) + ε)
 *   output = x / rms * weight
 *
 * 나누기(/)는 GPU에서 비용이 큰 연산입니다.
 * 대신 rsqrt (1/sqrt)를 사용하면:
 *   inv_rms = rsqrt(mean(x²) + ε)   ← rsqrtf()는 GPU 하드웨어 명령어
 *   output = x * inv_rms * weight     ← 나누기 대신 곱하기
 *
 * 용어 정리:
 *   - rsqrt: "reciprocal square root"의 약자. 1/sqrt(x), 즉 제곱근의 역수.
 *   - rsqrtf(): rsqrt의 float32 버전 C/CUDA 함수.
 *     C에서 수학 함수의 'f' 접미사는 float32를 의미합니다.
 *     (sqrtf, sinf, cosf 등도 동일한 관례. 접미사 없는 sqrt, sin 등은 double 버전)
 *   - SFU (Special Function Unit): SM 안에 있는 특수 연산 전용 하드웨어 회로.
 *     일반 ALU(덧셈, 곱셈 등)와는 별도로, 수학적으로 복잡하지만 자주 쓰이는
 *     연산을 빠르게 처리하기 위해 GPU 칩에 내장된 전용 회로입니다.
 *     SFU가 처리하는 연산: rsqrtf, sinf, cosf, expf, logf, __fdividef 등.
 *     SM당 SFU 수는 ALU보다 적으므로 (보통 4~8개 vs ALU 수십~수백 개),
 *     모든 스레드가 동시에 SFU 연산을 하면 병목이 될 수 있습니다.
 *
 * 성능 비교:
 *   x / sqrt(...)   → sqrt 1번 (SFU) + 나누기 1번 (비싼 연산) = 느림
 *   x * rsqrtf(...) → rsqrt 1번 (SFU) + 곱하기 1번 (ALU, 1 클록) = 빠름
 *
 *
 * [LayerNorm과의 비교]
 *
 * LayerNorm:
 *   μ = mean(x)                    ← 리덕션 1회
 *   σ² = mean((x - μ)²)            ← 리덕션 1회 (+ mean centering)
 *   output = (x - μ) / sqrt(σ² + ε) * γ + β
 *   learnable parameters: γ, β
 *
 * RMSNorm:
 *   RMS² = mean(x²)                ← 리덕션 1회 (mean centering 없음!)
 *   output = x / sqrt(RMS² + ε) * γ
 *   learnable parameters: γ만
 *
 * RMSNorm의 핵심 통찰:
 * - mean centering (x - μ)는 학습 품질에 큰 영향을 미치지 않음
 * - re-centering (+ β)도 불필요 → bias term 제거
 * - 리덕션 1회 절약 → 약 30% 속도 향상
 * - 파라미터 절반 절약 → 메모리 효율적
 *
 *
 * =============================================================================
 * 3. Fused 커널의 필요성
 * =============================================================================
 *
 * [LLaMA Pre-Norm 구조에서의 Fused 지점]
 *
 * LLaMA는 Pre-Norm 구조를 사용합니다. 하나의 Transformer 블록:
 *
 *   입력: residual (이전 블록에서 넘어온 값)
 *
 *   ① normed     = RMSNorm(residual)           ← 정규화
 *   ② attn_out   = Attention(normed)            ← 어텐션 연산
 *   ③ residual   = residual + attn_out          ← 잔차 연결
 *   ④ normed     = RMSNorm(residual)            ← 정규화
 *   ⑤ mlp_out    = MLP(normed)                  ← MLP 연산
 *   ⑥ residual   = residual + mlp_out           ← 잔차 연결
 *
 *   출력: residual (다음 블록으로)
 *
 * ③과 ④를 보면: 덧셈(잔차 연결) 직후에 RMSNorm이 옵니다.
 * 이 두 단계를 하나의 커널로 합치는 것이 Fused Add+RMSNorm입니다.
 *
 * Fused 커널의 출력이 2개인 이유:
 *   - residual (③의 결과): ⑥에서 다시 + mlp_out 할 때 원본 그대로 필요
 *   - normed   (④의 결과): ⑤의 MLP 입력으로 사용
 *   → 이 두 값은 서로 다르므로 각각 저장해야 합니다.
 *
 *
 * [별도 커널 vs Fused 커널의 메모리 접근 비교]
 *
 * 별도 커널로 ③과 ④를 실행하면:
 *
 *   커널 1 (③ 덧셈):
 *     - 글로벌 메모리에서 residual 읽기          (8KB per token)
 *     - 글로벌 메모리에서 attn_out 읽기           (8KB per token)
 *     - 덧셈 결과를 글로벌 메모리에 쓰기           (8KB per token)
 *
 *   커널 2 (④ RMSNorm):
 *     - 글로벌 메모리에서 residual 다시 읽기       (8KB per token) ← 낭비!
 *     - 정규화 결과를 글로벌 메모리에 쓰기           (8KB per token)
 *
 * Fused 커널 (③+④를 하나로):
 *     - 글로벌 메모리에서 residual, attn_out 읽기
 *     - 레지스터에서 덧셈 수행:  sum = residual + attn_out
 *     - 같은 레지스터에서 바로 제곱합 누적:  variance += sum * sum
 *       (sum이 아직 레지스터에 있으므로 글로벌 메모리 재읽기 불필요!)
 *     - 글로벌 메모리에 residual 저장   (⑥에서 재사용)
 *     - 글로벌 메모리에 normed 저장     (⑤ MLP의 입력)
 *     → 커널 2에서 residual을 다시 읽던 8KB/토큰이 제거됨!
 *
 * GPU는 연산(compute)보다 메모리 대역폭(bandwidth)이 병목인 경우가 많습니다.
 * 이런 메모리 바운드 커널에서 불필요한 글로벌 메모리 접근을 줄이는 것이
 * 가장 효과적인 최적화입니다.
 *
 *
 * =============================================================================
 * 4. 커널 런칭 전략
 * =============================================================================
 *
 * [블록 크기 선택 휴리스틱]
 *
 * block_size = min(hidden_size, max_block_size)
 *
 * max_block_size 결정:
 * - num_tokens < 256일 때: max_block_size = 1024
 *   → 토큰이 적으면 블록당 더 많은 스레드를 사용하여 SM 활용도 극대화
 *   → 적은 수의 블록으로도 GPU를 충분히 채울 수 있음
 *
 * - num_tokens >= 256일 때: max_block_size = 256
 *   → 토큰이 많으면 작은 블록으로 SM 동시성(concurrency) 증대
 *   → 여러 블록이 하나의 SM에서 동시 실행되어 메모리 레이턴시 은닉
 *
 * [레이턴시 은닉(latency hiding)이란?]
 *
 * 글로벌 메모리 읽기는 ~200~400 클록이 걸립니다.
 * 블록이 1개만 SM에 있으면, 메모리를 기다리는 동안 SM이 놀게 됩니다:
 *
 *   블록A: [연산] → [메모리 대기... 200클록 동안 아무것도 안 함...] → [연산]
 *                    └─ SM이 놀고 있음 (낭비)
 *
 * 블록이 여러 개 있으면, 하나가 메모리를 기다리는 동안 다른 블록을 실행합니다:
 *
 *   블록A: [연산] → [메모리 대기...............................] → [연산]
 *   블록B:          [연산] → [메모리 대기.....................] → [연산]
 *   블록C:                   [연산] → [메모리 대기..........] → [연산]
 *                   ↑                 ↑
 *               A가 기다리는 동안   B가 기다리는 동안
 *               B를 실행           C를 실행
 *
 * SM의 warp 스케줄러가 매 클록마다 "실행 가능한 워프"를 골라서 실행합니다.
 * 메모리를 기다리는 워프는 건너뛰고, 준비된 워프를 즉시 실행합니다.
 * 블록이 많을수록 "준비된 워프"가 많아져서 SM이 쉬는 시간이 줄어듭니다.
 *
 * 따라서:
 *   블록 크기 1024 → SM에 블록 1~2개만 올라감 → 대기 시 대체할 워프가 적음
 *   블록 크기 256  → SM에 블록 4~8개 올라감 → 대기 시 다른 워프로 전환 가능
 *
 * 이 휴리스틱은 vLLM에서 사용하는 것과 동일합니다.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "layernorm_kernels.cuh"


/**
 * =============================================================================
 * PyTorch 연동 함수
 * =============================================================================
 */

/**
 * rms_norm - RMSNorm Python 진입점
 *
 * @param out       출력 텐서 [num_tokens, hidden_size]
 * @param input     입력 텐서 [num_tokens, hidden_size]
 * @param weight    학습 가능한 가중치 [hidden_size]
 * @param epsilon   수치 안정성 상수 (보통 1e-6)
 */
void rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    double epsilon) {

    // 차원 추출
    int hidden_size = input.size(-1);
    int num_tokens = input.numel() / hidden_size;

    // 블록 크기 결정 (휴리스틱)
    const int max_block_size = (num_tokens < 256) ? 1024 : 256;
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, max_block_size));

    // CUDA 디바이스 가드 (멀티 GPU 지원)
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // dtype에 따라 적절한 커널 호출
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
        lightvllm::rms_norm_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                static_cast<float>(epsilon),
                num_tokens,
                hidden_size);
    });
}


/**
 * fused_add_rms_norm - Fused 잔차 덧셈 + RMSNorm Python 진입점
 *
 * 두 텐서를 in-place로 수정합니다:
 * - input:    정규화 결과로 덮어씌워짐
 * - residual: (원래 input + 원래 residual)로 업데이트됨
 *
 * @param input     입력/출력 텐서 [num_tokens, hidden_size]
 * @param residual  잔차 텐서 [num_tokens, hidden_size]
 * @param weight    학습 가능한 가중치 [hidden_size]
 * @param epsilon   수치 안정성 상수 (보통 1e-6)
 */
void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon) {

    int hidden_size = input.size(-1);
    int num_tokens = input.numel() / hidden_size;

    const int max_block_size = (num_tokens < 256) ? 1024 : 256;
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, max_block_size));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_add_rms_norm_kernel", [&] {
        lightvllm::fused_add_rms_norm_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                static_cast<float>(epsilon),
                num_tokens,
                hidden_size);
    });
}
