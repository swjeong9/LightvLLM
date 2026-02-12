/**
 * =============================================================================
 * LightvLLM Activation 커널 — SiLU, SiLU+Mul (Fused Gate)
 * =============================================================================
 *
 * 이 파일은 PyTorch 의존성 없이 사용할 수 있는 순수 CUDA 활성화 커널입니다.
 * C++ 단위 테스트에서도 이 헤더만 include하여 커널을 직접 테스트할 수 있습니다.
 *
 * 사용처:
 *   - csrc/activation_kernels.cu   (PyTorch wrapper에서 include)
 *   - tests/test_activation_kernel.cu  (low-level 테스트에서 include)
 *
 * SiLU의 수학적 배경과 LLaMA MLP 아키텍처에 대한 설명은
 * activation_kernels.cu 파일의 주석을 참조하세요.
 *
 *
 * =============================================================================
 * 1. 커널 구조 개요
 * =============================================================================
 *
 * 이 헤더에 구현된 커널은 두 가지입니다:
 *
 * (a) activation_kernel — element-wise 활성화 (SiLU 단독)
 *     입력 [..., d] → 출력 [..., d]
 *     Grid=(num_tokens), Block=(min(d, 1024))
 *
 * (b) act_and_mul_kernel — fused gate 연산 (silu_and_mul)
 *     입력 [..., 2*d] → 출력 [..., d]
 *     Grid=(num_tokens), Block=(min(d, 1024))
 *
 * 두 커널 모두 리덕션이 전혀 없는 순수 element-wise 연산입니다.
 * RMSNorm처럼 CUB BlockReduce나 __shared__ 메모리가 필요 없으며,
 * 각 스레드가 완전히 독립적으로 원소를 처리합니다.
 * → 성능은 오직 메모리 대역폭에 의해 결정되므로,
 *   128-bit 벡터화가 유일하고 핵심적인 최적화입니다.
 *
 *
 * =============================================================================
 * 2. 128-bit 벡터화 메모리 접근 (int4)
 * =============================================================================
 *
 * [왜 bf16을 하나씩 로드하면 안 되는가?]
 *
 * GPU 메모리 시스템은 스레드가 아무리 작은 데이터를 요청해도,
 * 한 번의 메모리 트랜잭션이 항상 32바이트 이상을 읽어옵니다.
 *
 * 스레드당 bf16 원소 1개(2바이트)를 요청하면:
 *   → 하드웨어는 32바이트를 읽음 → 2바이트만 사용 → 효율 6.25%
 *   → 나머지 30바이트는 버려짐
 *
 * 스레드당 int4(16바이트, bf16 × 8개)를 요청하면:
 *   → 하드웨어는 32바이트를 읽음 → 16바이트 사용 → 효율 50%
 *
 * 더 중요한 것은 warp(32개 스레드) 단위의 합산입니다:
 *   스레드당 2바이트 × 32 = 64바이트 → 1개 128바이트 트랜잭션, 50% 활용
 *   스레드당 16바이트 × 32 = 512바이트 → 4개 128바이트 트랜잭션, 100% 활용
 *
 * element-wise 커널은 연산이 거의 없어 메모리 대역폭이 성능을 결정하므로,
 * 이 차이가 직접적으로 처리 속도에 반영됩니다.
 *
 *
 * [int4란? — 왜 int4이고 int8이 아닌가?]
 *
 * CUDA 내장 벡터 타입: int1(4B), int2(8B), int4(16B)
 * int8(32바이트)은 존재하지 않습니다.
 *
 * 이유: GPU의 단일 메모리 로드 명령어(LDG.128)가 지원하는
 * 최대 크기가 128비트(= 16바이트)이기 때문입니다.
 * 32바이트를 한 번에 로드하는 하드웨어 명령어 자체가 없으므로,
 * int4(16바이트)가 스레드당 최대 로드 단위입니다.
 *
 * int4를 사용하면 한 번의 LDG.128 명령어로:
 *   - bf16: 16 / 2 = 8개 원소를 한 번에 로드
 *   - fp16: 16 / 2 = 8개 원소
 *   - fp32: 16 / 4 = 4개 원소
 *
 * [사용 방법: reinterpret_cast]
 * 1. 원래 타입 포인터를 int4*로 캐스팅:
 *      const int4* vec_ptr = reinterpret_cast<const int4*>(scalar_ptr);
 * 2. int4 단위로 로드:
 *      int4 chunk = VLLM_LDG(&vec_ptr[i]);
 * 3. 다시 원래 타입으로 캐스팅하여 개별 원소 접근:
 *      scalar_t* elems = reinterpret_cast<scalar_t*>(&chunk);
 *      scalar_t val = elems[j];
 *
 * reinterpret_cast는 메모리의 비트 패턴을 다른 타입으로 "재해석"합니다.
 * 실제 데이터 변환은 일어나지 않고, 컴파일러에게 "이 메모리를 이 타입으로
 * 읽어라"고 알려주는 것입니다. C의 포인터 캐스팅과 유사하지만 더 명시적입니다.
 *
 * [정렬(Alignment) 조건]
 * int4 로드는 포인터 주소가 16바이트 경계에 정렬되어야 합니다.
 * 정렬되지 않은 주소에서 int4를 로드하면 하드웨어 예외가 발생합니다.
 * 따라서 반드시 정렬 검사를 하고, 정렬되지 않은 경우 스칼라 폴백을 사용합니다.
 * (아래 is_16byte_aligned() 함수 참조)
 *
 * [VEC_SIZE 결정]
 * constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
 *
 * constexpr: "이 값은 컴파일할 때 이미 확정된다"는 의미입니다.
 * sizeof(scalar_t)는 타입 크기이므로 컴파일 시점에 알 수 있고,
 * 따라서 VEC_SIZE도 컴파일 시점에 계산이 끝납니다:
 *   bf16 → 16/2 = 8,  fp32 → 16/4 = 4
 *
 * 이것이 중요한 이유는 #pragma unroll 때문입니다.
 * #pragma unroll은 "루프를 풀어서 나열하라"는 컴파일러 지시입니다:
 *
 *   // 원래 루프 (매 반복: j<8 비교 → 본문 실행 → j++ → 점프)
 *   for (int j = 0; j < 8; j++) { rp[j] = ACT_FN(vp[j]); }
 *
 *   // unroll 후 (비교/증감/점프 제거, 명령어만 나열)
 *   rp[0] = ACT_FN(vp[0]);
 *   rp[1] = ACT_FN(vp[1]);
 *   ...
 *   rp[7] = ACT_FN(vp[7]);
 *
 * 루프 제어 오버헤드가 사라지고, GPU가 명령어를 파이프라인으로
 * 연속 실행할 수 있어 성능이 향상됩니다.
 *
 * unroll하려면 "몇 번 풀 것인가"를 컴파일 시점에 알아야 합니다.
 * VEC_SIZE가 constexpr이므로 컴파일러가 정확히 풀 수 있습니다.
 * 런타임 변수였다면 컴파일러가 몇 번 풀지 모르므로 unroll이 불가능합니다.
 *
 *
 * =============================================================================
 * 3. 함수 포인터 템플릿 패턴
 * =============================================================================
 *
 * 커널 템플릿에 활성화 함수를 함수 포인터로 전달합니다:
 *
 *   template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
 *   __global__ void activation_kernel(...) {
 *       out[i] = ACT_FN(input[i]);
 *   }
 *
 * 호출 시: activation_kernel<half, silu_kernel<half>><<<grid, block>>>(...)
 *
 * 이 패턴의 장점:
 * - 컴파일타임에 함수가 결정됨 → __forceinline__과 결합하면 오버헤드 제로
 *   (함수 호출 대신 커널 코드에 직접 삽입됨)
 * - 새로운 활성화 함수 추가 시 커널 코드 변경 없이 함수만 정의하면 됨
 *   (예: GELU를 추가하려면 gelu_kernel<T>만 정의)
 *
 * 주의: __device__ 함수 포인터는 호스트에서 사용할 수 없습니다.
 * 반드시 커널 launch 시 템플릿 인자로 전달해야 합니다 (런타임 함수 포인터 전달 불가).
 * 이는 GPU 코드가 별도의 명령어 집합(PTX)으로 컴파일되기 때문입니다.
 *
 *
 * =============================================================================
 * 4. __restrict__ 키워드
 * =============================================================================
 *
 * 포인터에 __restrict__를 붙이면 "이 포인터를 통해 접근하는 메모리는
 * 다른 포인터와 겹치지 않는다"고 컴파일러에게 보장합니다.
 * C99의 restrict와 동일한 의미이며, CUDA에서도 지원합니다.
 *
 * 효과: 컴파일러가 더 공격적으로 최적화할 수 있음
 * - 메모리 로드/스토어 순서를 자유롭게 재배열
 * - 동일 값을 여러 번 읽지 않고 레지스터에 캐싱
 * - 벡터화 명령어 생성 가능
 *
 * out과 input이 같은 메모리를 가리키지 않는다는 보장이 있으므로
 * (PyTorch 래퍼에서 별도 텐서를 전달) 안전하게 사용할 수 있습니다.
 */

#pragma once

#include <cstdint>
#include "cuda_compat.h"

namespace lightvllm {

// =============================================================================
// 유틸리티: 16바이트 정렬 검사
// =============================================================================
// int4 (128-bit) 로드를 안전하게 수행하려면 포인터가 16바이트 경계에
// 정렬되어 있어야 합니다. 정렬되지 않은 주소에서 int4를 로드하면
// 하드웨어 예외(misaligned address error)가 발생합니다.
//
// uintptr_t: 포인터를 정수로 변환하는 타입 (비트 연산 가능)
// & 15: 하위 4비트를 검사 → 0이면 16의 배수 → 정렬됨
__device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}


// =============================================================================
// SiLU 디바이스 함수
// =============================================================================
//
// SiLU(x) = x / (1 + exp(-x))
//
// [구현 상의 선택]
//
// 1. fp32로 변환 후 계산:
//    half/bfloat16에서 expf()의 정밀도가 떨어지므로
//    (float)로 변환하여 중간 계산을 수행합니다.
//    최종 결과만 (T)로 캐스팅하여 원래 정밀도로 복원합니다.
//
// 2. exp(-x)를 사용하는 이유:
//    exp(x)는 x가 큰 양수일 때 overflow → Inf가 됩니다.
//    exp(-x)는 x가 큰 양수일 때 0에 수렴하여 안전합니다.
//    x가 큰 음수일 때 exp(-x)가 overflow될 수 있지만,
//    분모 (1 + exp(-x))가 매우 커지므로 결과는 0에 수렴 → 안전.
//
// 3. GPU SFU(Special Function Unit):
//    expf()는 GPU의 전용 하드웨어 유닛(SFU)에서 실행됩니다.
//    SFU는 exp, sin, cos, rsqrt 등 초월 함수를 하드웨어 회로로
//    빠르게 계산하므로, 일반 산술 연산과 거의 동일한 속도입니다.
//    (rsqrtf()도 SFU 명령어 — RMSNorm에서 사용)
//
// [__forceinline__ + __device__]
// __device__: 이 함수는 GPU에서만 호출 가능함을 표시
// __forceinline__: 컴파일러에게 반드시 인라인하도록 강제 지시
// → 함수 호출 오버헤드를 완전히 제거하고 커널 코드에 직접 삽입됩니다.
// 특히 함수 포인터 템플릿과 결합하면, 커널 안에서 ACT_FN(x) 호출이
// 그냥 ((T)(((float)x) / (1.0f + expf((float)-x))))로 치환됩니다.
template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
    return (T)(((float)x) / (1.0f + expf((float)-x)));
}


// =============================================================================
// Element-wise 활성화 커널 (SiLU 단독 사용)
// =============================================================================
//
// 입력 [..., d]에 대해 각 원소에 활성화 함수를 적용합니다.
// 출력도 [..., d]로 동일한 shape입니다.
//
// [병렬화 전략]
// Grid:  (num_tokens,) — 각 블록이 하나의 토큰(행) 처리
// Block: (min(d, 1024),) — 스레드들이 stride 패턴으로 d개 원소 처리
//
// stride 패턴: d > blockDim.x인 경우, 각 스레드가
//   threadIdx.x, threadIdx.x + blockDim.x, threadIdx.x + 2*blockDim.x, ...
// 를 순회하며 원소를 처리합니다. 이렇게 하면 인접 스레드가 인접 메모리를
// 접근하여 메모리 coalescing이 보장됩니다.
//
// @tparam scalar_t  데이터 타입 (float, half, bfloat16)
// @tparam ACT_FN    활성화 함수 포인터 (silu_kernel 등)
//
// @param out    출력 텐서 [..., d]
// @param input  입력 텐서 [..., d]
// @param d      마지막 차원 크기
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const int d) {

    // 16 / sizeof(scalar_t) = bf16이면 8개, fp32이면 4개
    constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
    const int64_t token_idx = blockIdx.x;
    const scalar_t* in_ptr = input + token_idx * d;
    scalar_t* out_ptr = out + token_idx * d;

    // 세 포인터 모두 16바이트 정렬이고 d가 VEC_SIZE 이상이면 벡터화 경로 사용
    const bool aligned = is_16byte_aligned(in_ptr) && is_16byte_aligned(out_ptr);

    if (aligned && d >= VEC_SIZE) {
        // =================================================================
        // Fast Path: 128-bit 벡터화 루프
        // =================================================================
        // int4 단위(16바이트)로 한 번에 여러 원소를 로드/저장합니다.
        //
        // 예) bf16이면 VEC_SIZE=8:
        //   for (i = threadIdx.x; i < num_vecs; i += blockDim.x)
        //     한 번에 8개 bf16 원소를 로드 → 활성화 적용 → 저장
        //
        const int4* in_vec = reinterpret_cast<const int4*>(in_ptr);
        int4* out_vec = reinterpret_cast<int4*>(out_ptr);
        const int num_vecs = d / VEC_SIZE;
        const int vec_end = num_vecs * VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
            // VLLM_LDG: __ldg() 래퍼. 읽기 전용 데이터를 텍스처 캐시
            // 경로로 로드하여 L1 캐시 압박을 줄입니다.
            int4 v = VLLM_LDG(&in_vec[i]);
            int4 r;
            auto* vp = reinterpret_cast<scalar_t*>(&v);
            auto* rp = reinterpret_cast<scalar_t*>(&r);
            // #pragma unroll: 컴파일러에게 이 루프를 완전히 풀어서
            // VEC_SIZE개의 독립적인 명령어로 변환하도록 지시합니다.
            // 루프 제어 오버헤드(분기, 증감, 비교)를 제거하고,
            // 명령어 수준 병렬성(ILP)을 극대화합니다.
            // VEC_SIZE가 constexpr이므로 컴파일러가 정확히 풀 수 있습니다.
#pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                rp[j] = ACT_FN(vp[j]);
            }
            out_vec[i] = r;
        }

        // 나머지 원소 처리 (d가 VEC_SIZE로 나누어 떨어지지 않는 경우)
        // 예: d=130, VEC_SIZE=8이면 128개는 벡터화, 나머지 2개는 스칼라
        for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
            out_ptr[i] = ACT_FN(VLLM_LDG(&in_ptr[i]));
        }
    } else {
        // =================================================================
        // Scalar Fallback: 정렬되지 않았거나 d가 작은 경우
        // =================================================================
        // 정렬되지 않은 메모리에서 int4를 강제로 로드하면
        // CUDA 런타임 에러(misaligned address)가 발생합니다.
        // 안전을 위해 원소 단위로 처리합니다.
        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = VLLM_LDG(&in_ptr[idx]);
            out_ptr[idx] = ACT_FN(x);
        }
    }
}


// =============================================================================
// Fused Activation + Gate 커널 (silu_and_mul)
// =============================================================================
//
// 입력 [..., 2*d]를 절반으로 나누어:
//   gate = input[..., :d]   → ACT_FN 적용 (SiLU)
//   up   = input[..., d:]   → 그대로 곱셈
//   output = ACT_FN(gate) * up
//
// [왜 Fused 커널인가? — 커널 융합의 원리]
// 별도 커널 2개로 구현하면:
//   커널 1: temp = silu(gate)              → 글로벌 메모리 쓰기
//   커널 2: output = temp * up             → temp 글로벌 메모리 읽기
// 중간 결과 temp를 글로벌 메모리에 썼다가 다시 읽어야 합니다.
//
// Fused 커널에서는:
//   silu(gate) 결과를 레지스터에 보관한 채 바로 up과 곱합니다.
//   → 글로벌 메모리 쓰기 1회 + 읽기 1회 절약
//   → 이 커널은 완전히 메모리 바운드이므로 이 절약이 직접적인 성능 향상
//
// [메모리 레이아웃과 포인터 산술]
//   x_ptr = input + token_idx * 2 * d      → gate 시작점
//   y_ptr = x_ptr + d                       → up 시작점 (gate 바로 뒤)
//
// [벡터화 정렬 조건]
// gate(x_ptr)와 up(y_ptr) 포인터가 모두 정렬되어야 벡터화를 사용합니다.
// y_ptr = x_ptr + d이므로, d * sizeof(scalar_t)가 16의 배수이면
// x_ptr 정렬 시 y_ptr도 자동으로 정렬됩니다.
// LLaMA-7B의 intermediate_size=11008에서: 11008 * 2(bf16) = 22016 = 16 * 1376 ✓
//
// @tparam scalar_t  데이터 타입 (float, half, bfloat16)
// @tparam ACT_FN    활성화 함수 포인터 (silu_kernel 등)
//
// @param out    출력 텐서 [..., d]
// @param input  입력 텐서 [..., 2*d]  (gate와 up이 연결됨)
// @param d      출력의 마지막 차원 크기 (입력의 절반)
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const int d) {

    constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
    const int64_t token_idx = blockIdx.x;
    // 입력을 절반으로 나누기: 앞쪽이 gate, 뒤쪽이 up
    const scalar_t* x_ptr = input + token_idx * 2 * d;   // gate 시작점
    const scalar_t* y_ptr = x_ptr + d;                    // up 시작점
    scalar_t* out_ptr = out + token_idx * d;

    // 세 포인터 모두 16바이트 정렬 검사
    const bool aligned = is_16byte_aligned(x_ptr)
                      && is_16byte_aligned(y_ptr)
                      && is_16byte_aligned(out_ptr);

    if (aligned && d >= VEC_SIZE) {
        // =================================================================
        // Fast Path: 128-bit 벡터화 루프
        // =================================================================
        // gate와 up에서 각각 int4를 로드하여 처리
        const int4* x_vec = reinterpret_cast<const int4*>(x_ptr);
        const int4* y_vec = reinterpret_cast<const int4*>(y_ptr);
        int4* out_vec = reinterpret_cast<int4*>(out_ptr);
        const int num_vecs = d / VEC_SIZE;
        const int vec_end = num_vecs * VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
            int4 x = VLLM_LDG(&x_vec[i]);
            int4 y = VLLM_LDG(&y_vec[i]);
            int4 r;
            auto* xp = reinterpret_cast<scalar_t*>(&x);
            auto* yp = reinterpret_cast<scalar_t*>(&y);
            auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                // silu(gate) * up → 레지스터에서 바로 계산
                // 중간 결과가 글로벌 메모리를 거치지 않음
                rp[j] = ACT_FN(xp[j]) * yp[j];
            }
            out_vec[i] = r;
        }

        // 나머지 원소 처리
        for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
            out_ptr[i] = ACT_FN(VLLM_LDG(&x_ptr[i])) * VLLM_LDG(&y_ptr[i]);
        }
    } else {
        // =================================================================
        // Scalar Fallback
        // =================================================================
        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = VLLM_LDG(&x_ptr[idx]);
            const scalar_t y = VLLM_LDG(&y_ptr[idx]);
            out_ptr[idx] = ACT_FN(x) * y;
        }
    }
}

}  // namespace lightvllm
