/**
 * =============================================================================
 * LightvLLM 타입 디스패치 유틸리티
 * =============================================================================
 *
 * 이 파일은 PyTorch 텐서의 dtype에 따라 적절한 CUDA 커널을 호출하는
 * 매크로를 정의합니다.
 *
 * 목차:
 * 1. "디스패치(Dispatch)"란 무엇인가?
 * 2. 왜 dtype에 따라 다른 커널이 필요한가?
 * 3. C++ 템플릿과 런타임 타입의 문제
 * 4. PyTorch의 AT_DISPATCH 매크로
 * 5. 실제 매크로 정의
 */

#pragma once

#include <torch/all.h>

/**
 * =============================================================================
 * 1. "디스패치(Dispatch)"란 무엇인가?
 * =============================================================================
 *
 * 디스패치는 "적절한 곳으로 보내다"라는 의미입니다.
 *
 * 프로그래밍에서 타입 디스패치란:
 *   "데이터의 타입을 확인하고, 해당 타입에 맞는 함수/코드를 실행하는 것"
 *
 * 예를 들어:
 *   - float32 텐서가 들어오면 → float32용 커널 실행
 *   - float16 텐서가 들어오면 → float16용 커널 실행
 *   - bfloat16 텐서가 들어오면 → bfloat16용 커널 실행
 *
 *
 * =============================================================================
 * 2. 왜 dtype에 따라 다른 커널이 필요한가?
 * =============================================================================
 *
 * [이유 1: 메모리 크기가 다름]
 *
 *   타입별 크기:
 *   - float32:  4 바이트 (32비트)
 *   - float16:  2 바이트 (16비트)
 *   - bfloat16: 2 바이트 (16비트)
 *
 *   메모리 주소 계산이 달라짐:
 *     float32: ptr + idx * 4
 *     float16: ptr + idx * 2
 *
 *
 * [이유 2: 연산 명령어가 다름]
 *
 *   GPU는 타입별로 다른 하드웨어 유닛을 사용합니다:
 *
 *   - float32 덧셈: FADD (FP32 유닛)
 *   - float16 덧셈: HADD (FP16 유닛, Tensor Core 활용 가능)
 *
 *   최신 GPU(A100, H100 등)는 float16/bfloat16 연산이 float32보다 2배 이상 빠릅니다.
 *
 *
 * [이유 3: 정밀도와 범위가 다름]
 *
 *   float32 (단정밀도):
 *   - 지수: 8비트, 가수: 23비트
 *   - 범위: ±3.4 × 10^38
 *   - 정밀도: 약 7자리
 *
 *   float16 (반정밀도):
 *   - 지수: 5비트, 가수: 10비트
 *   - 범위: ±65,504
 *   - 정밀도: 약 3자리
 *   - 주의: 오버플로우 발생하기 쉬움
 *
 *   bfloat16 (Brain Floating Point):
 *   - 지수: 8비트, 가수: 7비트
 *   - 범위: float32와 동일 (±3.4 × 10^38)
 *   - 정밀도: 약 2자리
 *   - 장점: float32와 범위가 같아서 오버플로우 적음
 *
 *
 * [LLM에서 dtype 선택]
 *
 *   - 학습: bfloat16 선호 (넓은 범위로 안정적)
 *   - 추론: float16 선호 (Tensor Core 최적화, 메모리 절약)
 *   - 디버깅: float32 사용 (정확한 수치 확인)
 *
 *
 * =============================================================================
 * 3. C++ 템플릿과 런타임 타입의 문제
 * =============================================================================
 *
 * [문제 상황]
 *
 * CUDA 커널은 C++ 템플릿으로 작성됩니다:
 *
 *   template <typename T>
 *   __global__ void my_kernel(T* data, int n) {
 *       int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *       if (idx < n) {
 *           data[idx] = data[idx] * 2;
 *       }
 *   }
 *
 * 그런데 PyTorch 텐서의 dtype은 **런타임**에 결정됩니다:
 *
 *   void launch_kernel(torch::Tensor input) {
 *       // input.scalar_type()은 런타임에 결정됨
 *       // 하지만 템플릿 파라미터 T는 컴파일 타임에 결정되어야 함!
 *
 *       my_kernel<???><<<grid, block>>>(input.data_ptr<???>(), n);
 *       //        ^^^                                  ^^^
 *       //        컴파일 타임에 타입을 알아야 함
 *   }
 *
 *
 * [해결책: 모든 타입에 대해 미리 인스턴스화]
 *
 * 컴파일 타임에 지원할 모든 타입의 커널을 미리 만들어두고,
 * 런타임에 switch문으로 적절한 것을 선택합니다:
 *
 *   void launch_kernel(torch::Tensor input) {
 *       switch (input.scalar_type()) {
 *           case at::ScalarType::Float:
 *               my_kernel<float><<<grid, block>>>(input.data_ptr<float>(), n);
 *               break;
 *           case at::ScalarType::Half:
 *               my_kernel<__half><<<grid, block>>>(input.data_ptr<__half>(), n);
 *               break;
 *           case at::ScalarType::BFloat16:
 *               my_kernel<__nv_bfloat16><<<grid, block>>>(input.data_ptr<__nv_bfloat16>(), n);
 *               break;
 *           default:
 *               throw std::runtime_error("Unsupported dtype");
 *       }
 *   }
 *
 * 이 switch문을 매번 작성하는 것은 번거롭고 오류가 발생하기 쉽습니다.
 * → 그래서 매크로로 자동화합니다!
 *
 *
 * =============================================================================
 * 4. PyTorch의 AT_DISPATCH 매크로
 * =============================================================================
 *
 * PyTorch는 AT_DISPATCH_* 매크로를 제공합니다.
 *
 * [기본 사용법]
 *
 *   AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "kernel_name", [&] {
 *       // 이 블록 안에서 'scalar_t'가 실제 타입으로 정의됨
 *       // float32 텐서면 scalar_t = float
 *       // float64 텐서면 scalar_t = double
 *
 *       my_kernel<scalar_t><<<grid, block>>>(
 *           tensor.data_ptr<scalar_t>(),
 *           n
 *       );
 *   });
 *
 *
 * [매크로 확장 과정]
 *
 * AT_DISPATCH_FLOATING_TYPES(Float, "my_kernel", [&] { ... })
 *
 * 는 대략 다음과 같이 확장됩니다:
 *
 *   switch(Float) {
 *       case at::ScalarType::Float: {
 *           using scalar_t = float;
 *           { ... }  // 람다 실행
 *           break;
 *       }
 *       case at::ScalarType::Double: {
 *           using scalar_t = double;
 *           { ... }  // 람다 실행
 *           break;
 *       }
 *       default:
 *           AT_ERROR("my_kernel not implemented for ", toString(Float));
 *   }
 *
 *
 * [왜 람다([&] { ... })를 사용하는가?]
 *
 * 람다를 사용하면:
 * 1. 외부 변수를 캡처할 수 있음 ([&]는 참조로 캡처)
 * 2. 코드 블록을 여러 번 인스턴스화할 수 있음
 * 3. 각 타입별로 다른 scalar_t 정의 가능
 *
 *
 * =============================================================================
 * 5. 실제 매크로 정의
 * =============================================================================
 */

/**
 * VLLM_DISPATCH_CASE_FLOATING_TYPES
 *
 * LLM 추론에서 사용하는 세 가지 부동소수점 타입에 대한 케이스를 정의합니다:
 * - Float (float32): 디버깅, 정밀도가 중요할 때
 * - Half (float16): 추론 최적화, Tensor Core 활용
 * - BFloat16: 학습 및 추론, 넓은 동적 범위
 *
 * 내부 동작:
 *   AT_DISPATCH_CASE(타입, 코드)는 switch문의 case절을 생성합니다.
 *   각 case에서 scalar_t가 해당 타입으로 정의됩니다.
 *
 * 주의:
 *   Double(float64)은 포함하지 않음 - LLM에서 거의 사용하지 않고 느림
 */
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)              \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)


/**
 * VLLM_DISPATCH_FLOATING_TYPES - 메인 디스패치 매크로
 *
 * 사용법:
 *   VLLM_DISPATCH_FLOATING_TYPES(
 *       tensor.scalar_type(),  // 런타임에 확인할 타입
 *       "kernel_name",         // 오류 메시지에 사용할 이름
 *       [&] {                  // 실행할 코드 블록 (람다)
 *           // 여기서 scalar_t가 실제 타입으로 정의됨
 *           my_kernel<scalar_t><<<grid, block>>>(
 *               tensor.data_ptr<scalar_t>(),
 *               ...
 *           );
 *       }
 *   );
 *
 * 전체 예시 (RoPE 커널 호출):
 *
 *   void apply_rope(torch::Tensor query, torch::Tensor key,
 *                   torch::Tensor cos, torch::Tensor sin,
 *                   torch::Tensor positions) {
 *
 *       int num_tokens = query.size(0);
 *       int num_heads = query.size(1);
 *       int head_dim = query.size(2);
 *
 *       dim3 grid(num_tokens, num_heads);
 *       dim3 block(head_dim);
 *
 *       VLLM_DISPATCH_FLOATING_TYPES(
 *           query.scalar_type(),
 *           "rope_kernel",
 *           [&] {
 *               rope_kernel<scalar_t><<<grid, block>>>(
 *                   query.data_ptr<scalar_t>(),
 *                   key.data_ptr<scalar_t>(),
 *                   cos.data_ptr<scalar_t>(),
 *                   sin.data_ptr<scalar_t>(),
 *                   positions.data_ptr<int64_t>(),
 *                   num_tokens,
 *                   num_heads,
 *                   head_dim
 *               );
 *           }
 *       );
 *   }
 *
 *
 * 파라미터:
 *   TYPE - 텐서의 스칼라 타입 (tensor.scalar_type())
 *   NAME - 커널 이름 (오류 발생 시 표시됨)
 *   ...  - 실행할 코드가 포함된 람다
 *
 * 오류 처리:
 *   지원하지 않는 타입(예: int, double)이 들어오면
 *   "kernel_name not implemented for <타입>" 오류 발생
 */
#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                   \
    AT_DISPATCH_SWITCH(TYPE, NAME,                                      \
        VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
