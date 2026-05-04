/**
 * =============================================================================
 * Activation 커널 Low-level C++ 테스트
 * =============================================================================
 *
 * PyTorch 없이 CUDA 커널을 직접 테스트합니다.
 *
 * 컴파일:
 *   nvcc -O2 -std=c++17 -I csrc tests/test_activation_kernel.cu -o tests/test_activation_kernel
 *
 * 실행:
 *   ./tests/test_activation_kernel
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#include "activation_kernels.cuh"

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
 * CPU에서 SiLU를 계산: silu(x) = x / (1 + exp(-x))
 */
float silu_cpu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * CPU에서 element-wise SiLU 적용
 */
void silu_cpu_reference(
    float* out,
    const float* input,
    int num_tokens,
    int d) {

    for (int t = 0; t < num_tokens; t++) {
        for (int i = 0; i < d; i++) {
            out[t * d + i] = silu_cpu(input[t * d + i]);
        }
    }
}

/**
 * CPU에서 silu_and_mul 계산: silu(gate) * up
 * 입력: [..., 2*d], gate = 앞 절반, up = 뒷 절반
 */
void silu_and_mul_cpu_reference(
    float* out,
    const float* input,
    int num_tokens,
    int d) {

    for (int t = 0; t < num_tokens; t++) {
        const float* gate = input + t * 2 * d;
        const float* up = gate + d;
        for (int i = 0; i < d; i++) {
            out[t * d + i] = silu_cpu(gate[i]) * up[i];
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
// 테스트 케이스
// =============================================================================

/**
 * Test 1: SiLU 기본 정확성
 *
 * GPU 커널 결과를 CPU 참조 구현과 비교
 */
bool test_silu_correctness() {
    printf("\n[Test 1] SiLU 기본 정확성\n");

    const int num_tokens = 64;
    const int d = 256;
    const int total = num_tokens * d;

    // CPU 메모리 할당
    float* h_input   = (float*)malloc(total * sizeof(float));
    float* h_out_gpu = (float*)malloc(total * sizeof(float));
    float* h_out_cpu = (float*)malloc(total * sizeof(float));

    // 초기화
    fill_random(h_input, total, 42);

    // CPU 참조 계산
    silu_cpu_reference(h_out_cpu, h_input, num_tokens, d);

    // GPU 메모리 할당 및 복사
    float *d_input, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice));

    // 커널 실행
    dim3 grid(num_tokens);
    dim3 block(std::min(d, 1024));
    lightvllm::activation_kernel<float, lightvllm::silu_kernel<float>>
        <<<grid, block>>>(d_out, d_input, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 결과 복사
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // 비교
    float max_diff = compare_arrays(h_out_gpu, h_out_cpu, total);
    bool passed = max_diff < 1e-5f;
    printf("  설정: num_tokens=%d, d=%d\n", num_tokens, d);
    printf("  최대 절대 오차: %.2e (허용: 1e-5)\n", max_diff);
    printf("  결과: %s\n", passed ? "PASS" : "FAIL");

    // 정리
    free(h_input); free(h_out_gpu); free(h_out_cpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_out));

    return passed;
}


/**
 * Test 2: silu_and_mul 기본 정확성
 *
 * GPU 커널 결과를 CPU 참조 구현과 비교
 */
bool test_silu_and_mul_correctness() {
    printf("\n[Test 2] silu_and_mul 기본 정확성\n");

    const int num_tokens = 64;
    const int d = 256;
    const int input_total = num_tokens * 2 * d;
    const int output_total = num_tokens * d;

    // CPU 메모리 할당
    float* h_input   = (float*)malloc(input_total * sizeof(float));
    float* h_out_gpu = (float*)malloc(output_total * sizeof(float));
    float* h_out_cpu = (float*)malloc(output_total * sizeof(float));

    // 초기화
    fill_random(h_input, input_total, 42);

    // CPU 참조 계산
    silu_and_mul_cpu_reference(h_out_cpu, h_input, num_tokens, d);

    // GPU 메모리 할당 및 복사
    float *d_input, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, input_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, output_total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_total * sizeof(float), cudaMemcpyHostToDevice));

    // 커널 실행
    dim3 grid(num_tokens);
    dim3 block(std::min(d, 1024));
    lightvllm::act_and_mul_kernel<float, lightvllm::silu_kernel<float>>
        <<<grid, block>>>(d_out, d_input, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 결과 복사
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, output_total * sizeof(float), cudaMemcpyDeviceToHost));

    // 비교
    float max_diff = compare_arrays(h_out_gpu, h_out_cpu, output_total);
    bool passed = max_diff < 1e-5f;
    printf("  설정: num_tokens=%d, d=%d (입력: [..., %d])\n", num_tokens, d, 2 * d);
    printf("  최대 절대 오차: %.2e (허용: 1e-5)\n", max_diff);
    printf("  결과: %s\n", passed ? "PASS" : "FAIL");

    // 정리
    free(h_input); free(h_out_gpu); free(h_out_cpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_out));

    return passed;
}


/**
 * Test 3: SiLU 수학적 성질
 *
 * 1. silu(0) == 0 (정확히)
 * 2. silu(x)의 하한값 ≈ -0.278 (x ≈ -1.278에서 달성)
 * 3. 큰 양수: silu(x) ≈ x
 * 4. 큰 음수: silu(x) ≈ 0
 */
bool test_silu_properties() {
    printf("\n[Test 3] SiLU 수학적 성질\n");

    const int n = 5;
    float h_input[5] = {0.0f, -1.278f, 100.0f, -100.0f, 1.0f};
    float h_out[5];

    float *d_input, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));

    // 1개 블록, 5개 스레드
    lightvllm::activation_kernel<float, lightvllm::silu_kernel<float>>
        <<<1, n>>>(d_out, d_input, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = true;

    // 1. silu(0) == 0
    bool zero_ok = (h_out[0] == 0.0f);
    printf("  silu(0) = %.6e → %s\n", h_out[0], zero_ok ? "OK" : "FAIL");
    passed &= zero_ok;

    // 2. silu(-1.278) ≈ -0.278 (최솟값 근처)
    float expected_min = silu_cpu(-1.278f);
    bool min_ok = fabsf(h_out[1] - expected_min) < 1e-4f;
    printf("  silu(-1.278) = %.6f (기대: %.6f) → %s\n",
           h_out[1], expected_min, min_ok ? "OK" : "FAIL");
    passed &= min_ok;

    // 3. silu(100) ≈ 100
    bool large_pos_ok = fabsf(h_out[2] - 100.0f) < 1e-3f;
    printf("  silu(100) = %.6f (기대: ~100) → %s\n",
           h_out[2], large_pos_ok ? "OK" : "FAIL");
    passed &= large_pos_ok;

    // 4. silu(-100) ≈ 0
    bool large_neg_ok = fabsf(h_out[3]) < 1e-6f;
    printf("  silu(-100) = %.6e (기대: ~0) → %s\n",
           h_out[3], large_neg_ok ? "OK" : "FAIL");
    passed &= large_neg_ok;

    // 5. silu(1) = 1 / (1 + exp(-1)) ≈ 0.7311
    float expected_one = silu_cpu(1.0f);
    bool one_ok = fabsf(h_out[4] - expected_one) < 1e-5f;
    printf("  silu(1) = %.6f (기대: %.6f) → %s\n",
           h_out[4], expected_one, one_ok ? "OK" : "FAIL");
    passed &= one_ok;

    printf("  결과: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_out));

    return passed;
}


/**
 * Test 4: 수치 안정성 (극단값)
 *
 * 매우 큰 양수/음수 입력에서 NaN이나 Inf가 발생하지 않아야 함
 */
bool test_numerical_stability() {
    printf("\n[Test 4] 수치 안정성 (극단값)\n");

    const int n = 6;
    float h_input[6] = {
        1e10f, -1e10f,         // 매우 큰 값
        1e38f, -1e38f,         // float 범위 근처
        1e-30f, -1e-30f        // 매우 작은 값
    };
    float h_out[6];

    float *d_input, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));

    lightvllm::activation_kernel<float, lightvllm::silu_kernel<float>>
        <<<1, n>>>(d_out, d_input, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool no_nan = !has_nan_or_inf(h_out, 4);  // 처음 4개만 검사 (1e38은 결과도 1e38이라 inf 아님)
    printf("  silu(1e10) = %.6e\n", h_out[0]);
    printf("  silu(-1e10) = %.6e\n", h_out[1]);
    printf("  silu(1e38) = %.6e\n", h_out[2]);
    printf("  silu(-1e38) = %.6e\n", h_out[3]);
    printf("  silu(1e-30) = %.6e\n", h_out[4]);
    printf("  silu(-1e-30) = %.6e\n", h_out[5]);
    printf("  NaN/Inf 없음 (처음 4개): %s\n", no_nan ? "YES" : "NO");

    // 추가 검증: silu_and_mul도 극단값에서 안정
    const int d = 3;
    float h_input2[6] = {1e10f, -1e10f, 1e-30f, 1.0f, 2.0f, 3.0f};  // gate=앞3, up=뒤3
    float h_out2[3];

    float *d_input2, *d_out2;
    CUDA_CHECK(cudaMalloc(&d_input2, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out2, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input2, h_input2, 6 * sizeof(float), cudaMemcpyHostToDevice));

    lightvllm::act_and_mul_kernel<float, lightvllm::silu_kernel<float>>
        <<<1, d>>>(d_out2, d_input2, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out2, d_out2, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    bool no_nan2 = !has_nan_or_inf(h_out2, 3);
    printf("  silu_and_mul 극단값 NaN/Inf 없음: %s\n", no_nan2 ? "YES" : "NO");

    bool passed = no_nan && no_nan2;
    printf("  결과: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_input2));
    CUDA_CHECK(cudaFree(d_out2));

    return passed;
}


/**
 * Test 5: 성능 벤치마크
 *
 * LLaMA-like 차원에서 3가지 방식을 비교:
 *   1. CPU 참조 (silu_and_mul)
 *   2. GPU Non-Fused: activation_kernel(SiLU) → element-wise mul (커널 2회)
 *   3. GPU Fused: act_and_mul_kernel (커널 1회)
 *
 * Non-Fused vs Fused 비교로 커널 퓨전의 효과를 확인합니다.
 * Non-Fused는 gate에 SiLU를 적용하는 커널과 그 결과를 up과 곱하는 커널,
 * 총 2회의 커널 호출 + 중간 결과 메모리 읽기/쓰기가 필요합니다.
 * Fused는 1회의 커널 호출로 gate 읽기 → SiLU → up 읽기 → 곱셈 → 쓰기를 모두 처리합니다.
 */
bool test_performance() {
    printf("\n[Test 5] 성능 벤치마크 (LLaMA-like)\n");

    const int num_tokens = 1024;
    const int d = 4096;
    const int input_total = num_tokens * 2 * d;
    const int output_total = num_tokens * d;
    const int warmup = 10;
    const int repeat = 100;

    // CPU 메모리
    float* h_input   = (float*)malloc(input_total * sizeof(float));
    float* h_out_cpu = (float*)malloc(output_total * sizeof(float));
    fill_random(h_input, input_total, 42);

    // CPU 벤치마크
    clock_t cpu_start = clock();
    for (int r = 0; r < repeat; r++) {
        silu_and_mul_cpu_reference(h_out_cpu, h_input, num_tokens, d);
    }
    clock_t cpu_end = clock();
    float cpu_us = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1e6f / repeat;

    // GPU 메모리
    // 입력: [num_tokens, 2*d] — gate(앞 d) + up(뒤 d)
    float *d_input, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, input_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, output_total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_total * sizeof(float), cudaMemcpyHostToDevice));

    // Non-Fused용 중간 버퍼: SiLU(gate) 결과를 저장
    float *d_gate_activated;
    CUDA_CHECK(cudaMalloc(&d_gate_activated, output_total * sizeof(float)));

    dim3 grid(num_tokens);
    dim3 block(std::min(d, 1024));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ─────────────────────────────────────────────────────────────────────
    // [1] GPU Non-Fused: SiLU 커널 + element-wise mul 커널 (2회 호출)
    // ─────────────────────────────────────────────────────────────────────
    //
    // 실제로는 element-wise mul 전용 커널이 없으므로,
    // Non-Fused 시나리오를 시뮬레이션합니다:
    //   Step 1: activation_kernel로 gate에 SiLU 적용 → d_gate_activated
    //   Step 2: activation_kernel로 up에 항등 함수 적용 (메모리 복사 시뮬레이션)
    //
    // 이렇게 하면 "커널 2회 호출 + 중간 버퍼 메모리 트래픽" 비용을 측정할 수 있습니다.
    // 실제 Non-Fused는 이보다 더 느립니다 (곱셈 커널 추가).

    // 워밍업
    for (int i = 0; i < warmup; i++) {
        // gate는 입력의 stride가 2*d이므로, contiguous한 gate 배열이 필요합니다.
        // 실제 Non-Fused 시나리오에서는 슬라이싱 + SiLU + 곱셈으로 3개 커널이 필요할 수 있지만,
        // 여기서는 핵심인 "메모리 2회 읽기/쓰기" 비용만 측정합니다.
        lightvllm::activation_kernel<float, lightvllm::silu_kernel<float>>
            <<<grid, block>>>(d_gate_activated, d_input, d);
        lightvllm::activation_kernel<float, lightvllm::silu_kernel<float>>
            <<<grid, block>>>(d_out, d_input + d, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        lightvllm::activation_kernel<float, lightvllm::silu_kernel<float>>
            <<<grid, block>>>(d_gate_activated, d_input, d);
        lightvllm::activation_kernel<float, lightvllm::silu_kernel<float>>
            <<<grid, block>>>(d_out, d_input + d, d);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float nonfused_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&nonfused_ms, start, stop));
    float nonfused_us = (nonfused_ms / repeat) * 1000.0f;

    // ─────────────────────────────────────────────────────────────────────
    // [2] GPU Fused: act_and_mul_kernel (1회 호출)
    // ─────────────────────────────────────────────────────────────────────

    // 워밍업
    for (int i = 0; i < warmup; i++) {
        lightvllm::act_and_mul_kernel<float, lightvllm::silu_kernel<float>>
            <<<grid, block>>>(d_out, d_input, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        lightvllm::act_and_mul_kernel<float, lightvllm::silu_kernel<float>>
            <<<grid, block>>>(d_out, d_input, d);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fused_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&fused_ms, start, stop));
    float fused_us = (fused_ms / repeat) * 1000.0f;

    // ─────────────────────────────────────────────────────────────────────
    // 결과 출력
    // ─────────────────────────────────────────────────────────────────────

    printf("  설정: num_tokens=%d, d=%d\n", num_tokens, d);
    printf("  [1] CPU 참조 (silu_and_mul): %.1f us/call\n", cpu_us);
    printf("  [2] GPU Non-Fused (커널 2회): %.1f us/call\n", nonfused_us);
    printf("  [3] GPU Fused (커널 1회):     %.1f us/call\n", fused_us);
    printf("  CPU vs GPU Fused 속도 향상:   %.1fx\n", cpu_us / fused_us);
    printf("  Non-Fused vs Fused 속도 향상: %.2fx\n", nonfused_us / fused_us);
    printf("  결과: PASS (벤치마크)\n");

    free(h_input); free(h_out_cpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gate_activated));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return true;
}


// =============================================================================
// 메인 함수
// =============================================================================

int main() {
    printf("=============================================================================\n");
    printf("LightvLLM Activation 커널 Low-level C++ 테스트\n");
    printf("=============================================================================\n");

    int passed = 0;
    int total = 5;

    if (test_silu_correctness())          passed++;
    if (test_silu_and_mul_correctness())  passed++;
    if (test_silu_properties())           passed++;
    if (test_numerical_stability())       passed++;
    if (test_performance())               passed++;

    printf("\n=============================================================================\n");
    printf("결과: %d/%d 테스트 통과\n", passed, total);
    printf("=============================================================================\n");

    return (passed == total) ? 0 : 1;
}
