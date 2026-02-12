/**
 * =============================================================================
 * RMSNorm 커널 Low-level C++ 테스트
 * =============================================================================
 *
 * PyTorch 없이 CUDA 커널을 직접 테스트합니다.
 *
 * 컴파일:
 *   nvcc -O2 -std=c++17 -I csrc tests/test_layernorm_kernel.cu -o tests/test_layernorm_kernel
 *
 * 실행:
 *   ./tests/test_layernorm_kernel
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>

#include "layernorm_kernels.cuh"

// =============================================================================
// CUDA 에러 체크 매크로
// =============================================================================
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)


// =============================================================================
// CPU 참조 구현 (검증용)
// =============================================================================

/**
 * CPU에서 RMSNorm을 직접 계산하여 GPU 결과와 비교하기 위한 참조 구현
 *
 * output = input / sqrt(mean(input²) + epsilon) * weight
 */
void rms_norm_cpu_reference(
    float* out,
    const float* input,
    const float* weight,
    float epsilon,
    int num_tokens,
    int hidden_size) {

    for (int t = 0; t < num_tokens; t++) {
        const float* row = input + t * hidden_size;
        float* out_row = out + t * hidden_size;

        // sum of squares
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum_sq += row[i] * row[i];
        }
        float rsqrt_val = 1.0f / sqrtf(sum_sq / hidden_size + epsilon);

        // normalize with weight
        for (int i = 0; i < hidden_size; i++) {
            out_row[i] = row[i] * rsqrt_val * weight[i];
        }
    }
}




// =============================================================================
// 유틸리티 함수
// =============================================================================

/**
 * 두 배열을 비교하여 최대 절대 오차를 반환
 */
float compare_arrays(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}


/**
 * 배열에 NaN 또는 Inf가 있는지 확인
 */
bool has_nan_or_inf(const float* arr, int n) {
    for (int i = 0; i < n; i++) {
        if (isnan(arr[i]) || isinf(arr[i])) return true;
    }
    return false;
}


/**
 * 간단한 랜덤 초기화 (시드 기반)
 */
void fill_random(float* arr, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;  // [-1, 1]
    }
}


// =============================================================================
// Non-Fused 비교용 elementwise add 커널
// =============================================================================

/**
 * 단순 원소별 덧셈: residual[i] += input[i]
 * Fused 커널과의 성능 비교에서 "별도 커널 2개" 시나리오의 첫 번째 단계로 사용
 */
template <typename scalar_t>
__global__ void elementwise_add_kernel(
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ input,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        residual[idx] += input[idx];
    }
}


// =============================================================================
// 테스트 케이스
// =============================================================================

/**
 * Test 1: RMSNorm 기본 정확성
 *
 * GPU 커널 결과를 CPU 참조 구현과 비교
 */
bool test_rms_norm_correctness() {
    printf("\n[Test 1] RMSNorm 기본 정확성\n");

    const int num_tokens = 64;
    const int hidden_size = 256;
    const float epsilon = 1e-6f;
    const int total = num_tokens * hidden_size;

    // CPU 메모리 할당
    float* h_input  = (float*)malloc(total * sizeof(float));
    float* h_weight = (float*)malloc(hidden_size * sizeof(float));
    float* h_out_gpu = (float*)malloc(total * sizeof(float));
    float* h_out_cpu = (float*)malloc(total * sizeof(float));

    // 초기화
    fill_random(h_input, total, 42);
    fill_random(h_weight, hidden_size, 123);

    // CPU 참조 계산
    memcpy(h_out_cpu, h_input, total * sizeof(float));
    rms_norm_cpu_reference(h_out_cpu, h_input, h_weight, epsilon, num_tokens, hidden_size);

    // GPU 메모리 할당 및 복사
    float *d_input, *d_weight, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    // 커널 실행
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    lightvllm::rms_norm_kernel<float><<<grid, block>>>(
        d_out, d_input, d_weight, epsilon, num_tokens, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 결과 복사
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // 비교
    float max_diff = compare_arrays(h_out_gpu, h_out_cpu, total);
    bool passed = max_diff < 1e-5f;
    printf("  최대 절대 오차: %.2e (허용: 1e-5)\n", max_diff);
    printf("  결과: %s\n", passed ? "PASS" : "FAIL");

    // 정리
    free(h_input); free(h_weight); free(h_out_gpu); free(h_out_cpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));

    return passed;
}


/**
 * Test 2: Fused Add+RMSNorm 정확성
 *
 * output과 residual 모두 CPU 참조와 비교
 */
bool test_fused_add_rms_norm_correctness() {
    printf("\n[Test 2] Fused Add+RMSNorm 정확성\n");

    const int num_tokens = 64;
    const int hidden_size = 256;
    const float epsilon = 1e-6f;
    const int total = num_tokens * hidden_size;

    // CPU 메모리 할당
    float* h_input_cpu     = (float*)malloc(total * sizeof(float));
    float* h_residual_cpu  = (float*)malloc(total * sizeof(float));
    float* h_input_gpu     = (float*)malloc(total * sizeof(float));
    float* h_residual_gpu  = (float*)malloc(total * sizeof(float));
    float* h_weight        = (float*)malloc(hidden_size * sizeof(float));

    // 초기화
    fill_random(h_input_cpu, total, 42);
    fill_random(h_residual_cpu, total, 77);
    fill_random(h_weight, hidden_size, 123);

    // GPU용 복사본 생성
    memcpy(h_input_gpu, h_input_cpu, total * sizeof(float));
    memcpy(h_residual_gpu, h_residual_cpu, total * sizeof(float));

    // CPU 참조 계산: 단순 덧셈 후 기존 rms_norm_cpu_reference 재사용
    // (Fused는 GPU 메모리 대역폭 최적화이므로 CPU 참조에서는 불필요)
    for (int i = 0; i < total; i++) {
        h_residual_cpu[i] += h_input_cpu[i];
    }
    rms_norm_cpu_reference(h_input_cpu, h_residual_cpu, h_weight, epsilon, num_tokens, hidden_size);

    // GPU 메모리 할당 및 복사
    float *d_input, *d_residual, *d_weight;
    CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input_gpu, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual_gpu, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    // 커널 실행
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    lightvllm::fused_add_rms_norm_kernel<float><<<grid, block>>>(
        d_input, d_residual, d_weight, epsilon, num_tokens, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 결과 복사
    CUDA_CHECK(cudaMemcpy(h_input_gpu, d_input, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_residual_gpu, d_residual, total * sizeof(float), cudaMemcpyDeviceToHost));

    // 비교
    float max_diff_output = compare_arrays(h_input_gpu, h_input_cpu, total);
    float max_diff_residual = compare_arrays(h_residual_gpu, h_residual_cpu, total);
    bool passed = max_diff_output < 1e-5f && max_diff_residual < 1e-5f;
    printf("  output 최대 오차: %.2e\n", max_diff_output);
    printf("  residual 최대 오차: %.2e\n", max_diff_residual);
    printf("  결과: %s\n", passed ? "PASS" : "FAIL");

    // 정리
    free(h_input_cpu); free(h_residual_cpu);
    free(h_input_gpu); free(h_residual_gpu);
    free(h_weight);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_weight));

    return passed;
}


/**
 * Test 3: 영벡터 입력 안정성
 *
 * 입력이 모두 0일 때 NaN이나 Inf가 발생하지 않아야 함
 */
bool test_zero_input_stability() {
    printf("\n[Test 3] 영벡터 입력 안정성\n");

    const int num_tokens = 8;
    const int hidden_size = 128;
    const float epsilon = 1e-6f;
    const int total = num_tokens * hidden_size;

    // 영벡터 입력
    float* h_input  = (float*)calloc(total, sizeof(float));
    float* h_weight = (float*)malloc(hidden_size * sizeof(float));
    float* h_out    = (float*)malloc(total * sizeof(float));

    for (int i = 0; i < hidden_size; i++) h_weight[i] = 1.0f;

    // GPU 메모리
    float *d_input, *d_weight, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    // 커널 실행
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    lightvllm::rms_norm_kernel<float><<<grid, block>>>(
        d_out, d_input, d_weight, epsilon, num_tokens, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 결과 복사
    CUDA_CHECK(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // 검증: NaN/Inf 없고, 모든 값이 0이어야 함
    bool no_nan_inf = !has_nan_or_inf(h_out, total);
    float max_val = 0.0f;
    for (int i = 0; i < total; i++) {
        if (fabsf(h_out[i]) > max_val) max_val = fabsf(h_out[i]);
    }
    bool all_zero = max_val == 0.0f;
    bool passed = no_nan_inf && all_zero;
    printf("  NaN/Inf 없음: %s\n", no_nan_inf ? "YES" : "NO");
    printf("  모든 값 0: %s (최대 절대값: %.2e)\n", all_zero ? "YES" : "NO", max_val);
    printf("  결과: %s\n", passed ? "PASS" : "FAIL");

    free(h_input); free(h_weight); free(h_out);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));

    return passed;
}


/**
 * Test 4: Fused vs Non-Fused 성능 비교
 *
 * 동일한 연산(residual += input → rms_norm(residual))을 두 가지 방식으로 수행하고
 * latency를 비교하여 Fused 커널의 메모리 대역폭 절약 효과를 실측한다.
 *
 * Non-Fused: elementwise_add_kernel + rms_norm_kernel (커널 2개)
 * Fused:     fused_add_rms_norm_kernel (커널 1개)
 */
bool test_fused_vs_nonfused_performance() {
    printf("\n[Test 4] Fused vs Non-Fused 성능 비교 (LLaMA-like)\n");

    const int num_tokens = 1024;
    const int hidden_size = 4096;
    const float epsilon = 1e-5f;
    const int total = num_tokens * hidden_size;
    const int warmup = 10;
    const int repeat = 100;

    // CPU 메모리 (초기값 보관용)
    float* h_input    = (float*)malloc(total * sizeof(float));
    float* h_residual = (float*)malloc(total * sizeof(float));
    float* h_weight   = (float*)malloc(hidden_size * sizeof(float));
    fill_random(h_input, total, 42);
    fill_random(h_residual, total, 77);
    fill_random(h_weight, hidden_size, 123);

    // GPU 메모리
    float *d_input, *d_residual, *d_weight, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    // RMSNorm 커널 launch 설정
    dim3 norm_grid(num_tokens);
    dim3 norm_block(256);  // num_tokens >= 256이므로 block=256

    // Elementwise add 커널 launch 설정
    dim3 add_block(256);
    dim3 add_grid((total + 255) / 256);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 데이터 초기 전송 (1회)
    CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual, total * sizeof(float), cudaMemcpyHostToDevice));

    // =====================================================================
    // Non-Fused 벤치마크: elementwise_add + rms_norm (커널 2개)
    // =====================================================================
    // in-place 연산이지만, 벤치마크에서는 순수 커널 실행 시간만 측정.
    // 같은 데이터를 반복 실행해도 커널 latency 측정에는 영향 없음.

    // 워밍업
    for (int i = 0; i < warmup; i++) {
        elementwise_add_kernel<float><<<add_grid, add_block>>>(d_residual, d_input, total);
        lightvllm::rms_norm_kernel<float><<<norm_grid, norm_block>>>(
            d_out, d_residual, d_weight, epsilon, num_tokens, hidden_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 측정
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        elementwise_add_kernel<float><<<add_grid, add_block>>>(d_residual, d_input, total);
        lightvllm::rms_norm_kernel<float><<<norm_grid, norm_block>>>(
            d_out, d_residual, d_weight, epsilon, num_tokens, hidden_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float nonfused_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&nonfused_ms, start, stop));
    float avg_nonfused_us = (nonfused_ms / repeat) * 1000.0f;

    // =====================================================================
    // Fused 벤치마크: fused_add_rms_norm (커널 1개)
    // =====================================================================

    // 데이터 리셋
    CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual, total * sizeof(float), cudaMemcpyHostToDevice));

    // 워밍업
    for (int i = 0; i < warmup; i++) {
        lightvllm::fused_add_rms_norm_kernel<float><<<norm_grid, norm_block>>>(
            d_input, d_residual, d_weight, epsilon, num_tokens, hidden_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 측정
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        lightvllm::fused_add_rms_norm_kernel<float><<<norm_grid, norm_block>>>(
            d_input, d_residual, d_weight, epsilon, num_tokens, hidden_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fused_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&fused_ms, start, stop));
    float avg_fused_us = (fused_ms / repeat) * 1000.0f;

    // 결과 출력
    printf("  설정: num_tokens=%d, hidden_size=%d\n", num_tokens, hidden_size);
    printf("  Non-Fused (add + rms_norm): %.1f us/call\n", avg_nonfused_us);
    printf("  Fused (fused_add_rms_norm):  %.1f us/call\n", avg_fused_us);
    printf("  Fused 속도 향상: %.2fx\n", avg_nonfused_us / avg_fused_us);
    printf("  결과: PASS (벤치마크)\n");

    free(h_input); free(h_residual); free(h_weight);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return true;
}


// =============================================================================
// 메인 함수
// =============================================================================

int main() {
    printf("=============================================================================\n");
    printf("LightvLLM RMSNorm 커널 Low-level C++ 테스트\n");
    printf("=============================================================================\n");

    int passed = 0;
    int total = 4;

    if (test_rms_norm_correctness())       passed++;
    if (test_fused_add_rms_norm_correctness()) passed++;
    if (test_zero_input_stability())        passed++;
    if (test_fused_vs_nonfused_performance()) passed++;

    printf("\n=============================================================================\n");
    printf("결과: %d/%d 테스트 통과\n", passed, total);
    printf("=============================================================================\n");

    return (passed == total) ? 0 : 1;
}
