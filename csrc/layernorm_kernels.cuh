/**
 * =============================================================================
 * 커널 알고리즘
 * =============================================================================
 *
 * [병렬화 전략]
 *
 * 입력: [num_tokens, hidden_size]
 * Grid: (num_tokens,) — 각 블록이 하나의 토큰(행) 처리
 * Block: (block_size,) — 블록 내 스레드들이 협력하여 hidden_size 원소 처리
 *
 * [Two-Pass 알고리즘]
 *
 * Pass 1: 제곱합(sum of squares) 계산
 *   - 각 스레드가 hidden_size의 일부를 순회하며 x² 누적
 *   - **fp32로 변환 후 누적** (half precision의 수치 안정성 확보)
 *   - CUB BlockReduce로 블록 전체의 합 계산
 *   - Thread 0이 rsqrt(variance / hidden_size + ε) 계산
 *   - 결과를 __shared__ 메모리로 브로드캐스트
 *
 * Pass 2: 정규화 적용
 *   - output[i] = (float(input[i]) * s_variance) * weight[i]
 *   - weight와 곱하는 것은 scalar_t 타입에서 수행 (원래 정밀도 유지)
 *
 *
 * [왜 fp32로 누적하는가?]
 *
 * bfloat16의 가수부(mantissa)는 7비트 → 유효 자릿수 ~2.4자리
 * hidden_size=4096인 경우, 4096개의 x²를 bf16으로 누적하면
 * 큰 값에 작은 값을 더할 때 작은 값이 소실됨 (catastrophic cancellation)
 *
 * 해결: 각 스레드가 float(32bit, 유효 자릿수 ~7자리)로 변환 후 누적
 * → 최종 결과만 원래 dtype으로 변환
 *
 *
 * [CUB BlockReduce 동작 원리]
 *
 * CUB의 BlockReduce는 내부적으로 warp shuffle 명령어를 사용합니다:
 *
 * 1단계: Warp 내 리덕션 (32개 스레드)
 *   __shfl_xor_sync()를 사용하여 레지스터 간 직접 데이터 교환
 *   → 공유 메모리 사용 없이 매우 빠른 리덕션
 *
 * 2단계: Warp 간 리덕션
 *   각 warp의 결과를 공유 메모리(TempStorage)에 저장
 *   첫 번째 warp가 이 값들을 다시 리듀스
 *
 * BlockReduce<float, 1024>는 최대 1024개 스레드를 지원합니다.
 * 실제 block 크기가 더 작을 수 있으므로 Reduce()에 유효 스레드 수를 전달합니다.
 *
 *
 * [__shared__ 브로드캐스트 패턴]
 *
 * Thread 0이 리덕션 결과를 받아 rsqrt()를 계산합니다.
 * 이 결과를 모든 스레드가 사용해야 하므로:
 *   1. Thread 0이 __shared__ 변수에 저장
 *   2. __syncthreads()로 동기화
 *   3. 모든 스레드가 __shared__ 변수에서 읽기
 *
 * 이것은 CUDA에서 "broadcast" 패턴의 표준적인 구현입니다.
 */

#pragma once

#include <cstdint>
#include <cub/cub.cuh>
#include "cub_compat.h"
#include "cuda_compat.h"

namespace lightvllm {

/**
 * rms_norm_kernel - RMSNorm 메인 CUDA 커널
 *
 * 병렬화 전략:
 * - Grid: (num_tokens,) - 각 블록이 하나의 토큰(행) 담당
 * - Block: (block_size,) - 스레드들이 협력하여 hidden_size 원소 처리
 *
 * @tparam scalar_t   데이터 타입 (float, half, bfloat16)
 *
 * @param out          출력 텐서 [num_tokens, hidden_size]
 * @param input        입력 텐서 [num_tokens, hidden_size]
 * @param weight       학습 가능한 가중치 [hidden_size]
 * @param epsilon      수치 안정성을 위한 작은 상수 (보통 1e-6)
 * @param num_tokens   토큰 수 (행 수)
 * @param hidden_size  은닉 차원 크기 (열 수)
 */
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {

    // 각 블록이 하나의 토큰(행) 처리
    const int token_idx = blockIdx.x;
    const scalar_t* input_row = input + token_idx * hidden_size;

    // =========================================================================
    // Pass 1: 제곱합 계산 (Sum of Squares)
    // =========================================================================
    // 각 스레드가 stride 패턴으로 hidden_size의 일부를 순회
    // fp32로 변환하여 누적 → 수치 안정성 확보
    float variance = 0.0f;
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float x = static_cast<float>(input_row[idx]);
        variance += x * x;
    }

    // =========================================================================
    // CUB BlockReduce: 블록 내 모든 스레드의 variance를 합산
    // =========================================================================
    //
    // 위의 for 루프에서 각 스레드는 자기 담당 원소들의 부분합만 갖고 있습니다.
    // 예: hidden_size=4096, 블록=256스레드이면 각 스레드가 16개 원소의 부분합을 보유.
    // 이제 256개 스레드의 부분합을 하나의 전체합으로 합쳐야 합니다.
    // 이것이 CUB BlockReduce가 하는 일입니다.
    //
    // using BlockReduce = cub::BlockReduce<float, 1024>;
    //   - float: 리덕션할 값의 타입
    //   - 1024: 지원할 최대 블록 크기 (컴파일타임 상수)
    //     CUDA에서 개별 블록의 최대 스레드 수는 1024입니다 (CC 2.0 이후 모든 GPU 공통).
    //     ※ SM당 최대 스레드 수(예: A100은 2048)와는 다른 제한입니다.
    //        SM당 2048 = 블록 1024×2개 또는 블록 256×8개 등 (여러 블록의 합).
    //        개별 블록은 어떤 GPU에서든 1024를 넘을 수 없습니다.
    //     1024로 설정하면 가능한 모든 블록 크기를 하나의 인스턴스로 커버.
    //
    // __shared__ typename BlockReduce::TempStorage reduceStore;
    //   - CUB가 내부적으로 사용하는 __shared__ 메모리 공간.
    //     워프 간 부분합을 교환할 때 사용됨.
    //   - typename 키워드: C++ 문법 규칙. BlockReduce::TempStorage가
    //     "값"이 아니라 "타입"임을 컴파일러에게 명시. 없으면 컴파일 에러.
    //
    // variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
    //   - BlockReduce(reduceStore): 공유 메모리를 전달하여 임시 객체 생성
    //   - .Reduce(variance, CubAddOp{}, blockDim.x):
    //       variance:    이 스레드의 부분합 (입력값)
    //       CubAddOp{}:  덧셈 연산자 (어떤 연산으로 합칠지. cub_compat.h 참조)
    //       blockDim.x:  유효 스레드 수 (실제 참여하는 스레드 개수)
    //   - 반환값: 스레드 0에만 전체 합이 들어옴.
    //     나머지 스레드의 반환값은 미정의(undefined)이므로 사용하면 안 됨.
    //     그래서 아래에서 Thread 0이 __shared__를 통해 결과를 브로드캐스트함.
    //
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

    // =========================================================================
    // rsqrt 계산 및 브로드캐스트
    // =========================================================================
    // Thread 0만 리덕션 결과를 받으므로, __shared__를 통해 모든 스레드에 전파
    __shared__ float s_variance;
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    // =========================================================================
    // Pass 2: 정규화 적용
    // =========================================================================
    // output[i] = (float(input[i]) * rsqrt_value) * weight[i]
    // float로 변환 → rsqrt 곱하기 → scalar_t로 변환 → weight 곱하기
    scalar_t* out_row = out + token_idx * hidden_size;
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float x = static_cast<float>(input_row[idx]);
        out_row[idx] = static_cast<scalar_t>(x * s_variance) * weight[idx];
    }
}


/**
 * fused_add_rms_norm_kernel - 잔차 덧셈 + RMSNorm Fused 커널
 *
 * 하나의 커널에서 두 가지 연산을 수행합니다:
 * 1. residual += input  (잔차 연결)
 * 2. input = rms_norm(residual) * weight  (정규화)
 *
 * 두 텐서 모두 in-place로 수정됩니다:
 * - residual: 덧셈 결과 저장 (다음 레이어의 잔차 입력으로 사용)
 * - input: 정규화 결과 저장 (다음 레이어의 입력으로 사용)
 *
 * @tparam scalar_t   데이터 타입 (float, half, bfloat16)
 *
 * @param input        입력/출력 텐서 [num_tokens, hidden_size] (정규화 결과로 덮어씌워짐)
 * @param residual     잔차 텐서 [num_tokens, hidden_size] (input + residual로 업데이트됨)
 * @param weight       학습 가능한 가중치 [hidden_size]
 * @param epsilon      수치 안정성을 위한 작은 상수
 * @param num_tokens   토큰 수 (행 수)
 * @param hidden_size  은닉 차원 크기 (열 수)
 */
template <typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {

    const int token_idx = blockIdx.x;

    // =========================================================================
    // Pass 1: 잔차 덧셈 + 제곱합 계산 (Fused)
    // =========================================================================
    // 핵심 최적화: input과 residual을 레지스터에서 더하고,
    // 바로 sum_sq에 누적한 후, residual에 저장
    // → 글로벌 메모리 읽기 1회 절약
    float variance = 0.0f;
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float inp = static_cast<float>(input[token_idx * hidden_size + idx]);
        float res = static_cast<float>(residual[token_idx * hidden_size + idx]);
        float sum = inp + res;
        variance += sum * sum;
        // residual에 덧셈 결과 저장 (다음 레이어의 잔차 입력)
        residual[token_idx * hidden_size + idx] = static_cast<scalar_t>(sum);
    }

    // =========================================================================
    // CUB BlockReduce + rsqrt (rms_norm_kernel과 동일)
    // =========================================================================
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

    __shared__ float s_variance;
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    // =========================================================================
    // Pass 2: 정규화 적용 (residual에서 읽어서 input에 저장)
    // =========================================================================
    // residual은 Pass 1에서 이미 업데이트되었으므로,
    // 여기서 읽는 값은 (원래 input + 원래 residual)
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float x = static_cast<float>(residual[token_idx * hidden_size + idx]);
        input[token_idx * hidden_size + idx] =
            static_cast<scalar_t>(x * s_variance) * weight[idx];
    }
}

}  // namespace lightvllm
