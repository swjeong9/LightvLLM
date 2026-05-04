/**
 * =============================================================================
 * RoPE 커널 Low-level C++ 테스트
 * =============================================================================
 *
 * PyTorch 없이 CUDA 커널을 직접 테스트합니다.
 *
 * 컴파일:
 *   nvcc -O2 -std=c++17 -I csrc tests/test_rope_kernel.cu -o tests/test_rope_kernel
 *
 * 실행:
 *   ./tests/bin/test_rope_kernel
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>

#include "pos_encoding_kernels.cuh"

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
 * CPU에서 RoPE를 직접 계산하여 GPU 결과와 비교하기 위한 참조 구현
 * GPT-NeoX 스타일만 구현 (LLaMA 기준)
 */
void rope_cpu_reference(
    float* query,       // [num_tokens, num_heads, head_size]
    float* key,         // [num_tokens, num_kv_heads, head_size]
    const int64_t* positions,
    const float* cos_sin_cache,  // [max_position, rot_dim]
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_size,
    int rot_dim) {

    int embed_dim = rot_dim / 2;
    int query_stride = num_heads * head_size;
    int key_stride = num_kv_heads * head_size;

    for (int t = 0; t < num_tokens; t++) {
        int64_t pos = positions[t];
        const float* cache_ptr = cos_sin_cache + pos * rot_dim;
        const float* cos_ptr = cache_ptr;
        const float* sin_ptr = cache_ptr + embed_dim;

        // Query 처리
        for (int h = 0; h < num_heads; h++) {
            float* head_ptr = query + t * query_stride + h * head_size;
            for (int i = 0; i < embed_dim; i++) {
                // NeoX 스타일: 전반부/후반부 쌍
                int x_idx = i;
                int y_idx = embed_dim + i;
                float x = head_ptr[x_idx];
                float y = head_ptr[y_idx];
                float c = cos_ptr[i];
                float s = sin_ptr[i];
                head_ptr[x_idx] = x * c - y * s;
                head_ptr[y_idx] = x * s + y * c;
            }
        }

        // Key 처리
        for (int h = 0; h < num_kv_heads; h++) {
            float* head_ptr = key + t * key_stride + h * head_size;
            for (int i = 0; i < embed_dim; i++) {
                int x_idx = i;
                int y_idx = embed_dim + i;
                float x = head_ptr[x_idx];
                float y = head_ptr[y_idx];
                float c = cos_ptr[i];
                float s = sin_ptr[i];
                head_ptr[x_idx] = x * c - y * s;
                head_ptr[y_idx] = x * s + y * c;
            }
        }
    }
}


// =============================================================================
// cos/sin 캐시 생성 (CPU)
// =============================================================================

/**
 * θ_i = position × base^(-2i/d) 공식으로 cos/sin 캐시를 생성합니다.
 *
 * 캐시 레이아웃: [max_position, rot_dim]
 * 각 position에 대해: [cos(θ_0), cos(θ_1), ..., sin(θ_0), sin(θ_1), ...]
 */
void generate_cos_sin_cache(
    float* cache,       // [max_position, rot_dim]
    int max_position,
    int rot_dim,
    float base = 10000.0f) {

    int embed_dim = rot_dim / 2;

    for (int pos = 0; pos < max_position; pos++) {
        for (int i = 0; i < embed_dim; i++) {
            float theta = (float)pos * powf(base, -2.0f * i / rot_dim);
            cache[pos * rot_dim + i] = cosf(theta);                  // cos 부분
            cache[pos * rot_dim + embed_dim + i] = sinf(theta);      // sin 부분
        }
    }
}


// =============================================================================
// 테스트 유틸리티
// =============================================================================

/**
 * 두 float 배열을 비교하여 오차가 허용 범위 내인지 확인
 */
bool compare_arrays(const float* a, const float* b, int size,
                    float tolerance, const char* name) {
    float max_diff = 0.0f;
    int max_diff_idx = 0;

    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    if (max_diff > tolerance) {
        printf("  FAILED: %s - max diff = %e at index %d (gpu=%f, cpu=%f)\n",
               name, max_diff, max_diff_idx, a[max_diff_idx], b[max_diff_idx]);
        return false;
    }

    printf("  PASSED: %s - max diff = %e\n", name, max_diff);
    return true;
}

/**
 * 벡터의 L2 norm을 계산
 */
float compute_norm(const float* arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i] * arr[i];
    }
    return sqrtf(sum);
}


// =============================================================================
// 테스트 케이스
// =============================================================================

/**
 * 테스트 1: 기본 동작 테스트
 *
 * 작은 크기로 커널을 실행하고 CPU 참조 구현과 비교
 */
bool test_basic_correctness() {
    printf("\n[Test 1] Basic Correctness (GPU vs CPU reference)\n");

    // 파라미터 설정
    const int num_tokens = 1<<10;  // 1024
    const int num_heads = 64;
    const int num_kv_heads = 8;
    const int head_size = 128;
    const int rot_dim = head_size;  // 전체 head를 회전
    const int max_position = 1<<14;  // 16384

    // 크기 계산
    const int query_size = num_tokens * num_heads * head_size;
    const int key_size = num_tokens * num_kv_heads * head_size;
    const int cache_size = max_position * rot_dim;
    const int64_t query_stride = num_heads * head_size;
    const int64_t key_stride = num_kv_heads * head_size;

    // CPU 메모리 할당
    float* h_query_gpu = (float*)malloc(query_size * sizeof(float));
    float* h_query_cpu = (float*)malloc(query_size * sizeof(float));
    float* h_key_gpu = (float*)malloc(key_size * sizeof(float));
    float* h_key_cpu = (float*)malloc(key_size * sizeof(float));
    int64_t* h_positions = (int64_t*)malloc(num_tokens * sizeof(int64_t));
    float* h_cache = (float*)malloc(cache_size * sizeof(float));

    // 테스트 데이터 초기화
    for (int i = 0; i < query_size; i++) {
        float val = (float)(i % 7 + 1) * 0.1f;  // 0.1 ~ 0.7 반복
        h_query_gpu[i] = val;
        h_query_cpu[i] = val;  // CPU 참조용 복사
    }
    for (int i = 0; i < key_size; i++) {
        float val = (float)(i % 5 + 1) * 0.2f;  // 0.2 ~ 1.0 반복
        h_key_gpu[i] = val;
        h_key_cpu[i] = val;
    }

    // positions: 각 토큰에 0 ~ max_position-1 범위의 position 할당
    for (int i = 0; i < num_tokens; i++) {
        h_positions[i] = i * (max_position / num_tokens);  // 균등 분포
    }

    // cos/sin 캐시 생성
    generate_cos_sin_cache(h_cache, max_position, rot_dim);

    // CPU 참조 계산 (시간 측정)
    clock_t cpu_start = clock();
    rope_cpu_reference(h_query_cpu, h_key_cpu, h_positions, h_cache,
                       num_tokens, num_heads, num_kv_heads, head_size, rot_dim);
    clock_t cpu_end = clock();
    double cpu_time_us = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1e6;

    // GPU 메모리 할당
    float *d_query, *d_key, *d_cache;
    int64_t *d_positions;
    CUDA_CHECK(cudaMalloc(&d_query, query_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_key, key_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_positions, num_tokens * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_cache, cache_size * sizeof(float)));

    // CPU → GPU 복사
    CUDA_CHECK(cudaMemcpy(d_query, h_query_gpu, query_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, h_key_gpu, key_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions, h_positions, num_tokens * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cache, h_cache, cache_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // GPU 커널 실행 (CUDA Event로 시간 측정)
    cudaEvent_t gpu_start, gpu_end;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_end));

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * rot_dim / 2, 512));

    CUDA_CHECK(cudaEventRecord(gpu_start));
    lightvllm::rotary_embedding_kernel<float, true><<<grid, block>>>(
        d_positions, d_query, d_key, d_cache,
        rot_dim, query_stride, key_stride,
        num_heads, num_kv_heads, head_size);
    CUDA_CHECK(cudaEventRecord(gpu_end));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(gpu_end));

    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, gpu_start, gpu_end));
    double gpu_time_us = gpu_time_ms * 1000.0;

    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_end));

    // GPU → CPU 복사
    CUDA_CHECK(cudaMemcpy(h_query_gpu, d_query, query_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_key_gpu, d_key, key_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 결과 비교
    bool pass = true;
    pass &= compare_arrays(h_query_gpu, h_query_cpu, query_size, 1e-5f, "Query");
    pass &= compare_arrays(h_key_gpu, h_key_cpu, key_size, 1e-5f, "Key");

    // 실행 시간 출력
    printf("  CPU time: %.1f us\n", cpu_time_us);
    printf("  GPU time: %.1f us (kernel only, excludes memcpy)\n", gpu_time_us);
    printf("  Speedup:  %.1fx\n", cpu_time_us / gpu_time_us);

    // 정리
    free(h_query_gpu); free(h_query_cpu);
    free(h_key_gpu); free(h_key_cpu);
    free(h_positions); free(h_cache);
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaFree(d_cache));

    return pass;
}


/**
 * 테스트 2: Norm 보존 테스트
 *
 * 회전 행렬의 핵심 성질: ||R(θ)x|| = ||x||
 * RoPE 적용 전후로 각 head 벡터의 크기가 보존되는지 확인
 */
bool test_norm_preservation() {
    printf("\n[Test 2] Norm Preservation\n");

    // 파라미터 설정
    const int num_tokens = 1<<10;  // 1024
    const int num_heads = 64;
    const int num_kv_heads = 8;
    const int head_size = 128;
    const int rot_dim = head_size;  // 전체 head를 회전
    const int max_position = 1<<14;  // 16384

    const int query_size = num_tokens * num_heads * head_size;
    // const int key_size = num_tokens * num_kv_heads * head_size; # key는 nullptr로 테스트할 예정
    const int cache_size = max_position * rot_dim;
    const int64_t query_stride = num_heads * head_size;
    const int64_t key_stride = num_kv_heads * head_size;

    // CPU 메모리
    float* h_query_before = (float*)malloc(query_size * sizeof(float));
    float* h_query_after = (float*)malloc(query_size * sizeof(float));
    int64_t* h_positions = (int64_t*)malloc(num_tokens * sizeof(int64_t));
    float* h_cache = (float*)malloc(cache_size * sizeof(float));

    // 랜덤한 값으로 초기화
    srand(42);
    for (int i = 0; i < query_size; i++) {
        h_query_before[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        h_query_after[i] = h_query_before[i];
    }
    for (int i = 0; i < num_tokens; i++) {
        h_positions[i] = i * (max_position / num_tokens);
    }

    generate_cos_sin_cache(h_cache, max_position, rot_dim);

    // GPU 실행
    float *d_query, *d_cache;
    int64_t *d_positions;
    CUDA_CHECK(cudaMalloc(&d_query, query_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_positions, num_tokens * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_cache, cache_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_query, h_query_after, query_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions, h_positions, num_tokens * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cache, h_cache, cache_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * rot_dim / 2, 512));

    // key를 nullptr로 전달 (query만 테스트)
    lightvllm::rotary_embedding_kernel<float, true><<<grid, block>>>(
        d_positions, d_query, (float*)nullptr, d_cache,
        rot_dim, query_stride, key_stride,
        num_heads, num_kv_heads, head_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_query_after, d_query, query_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 각 head 벡터의 norm 비교
    bool pass = true;
    float tolerance = 1e-4f;

    for (int t = 0; t < num_tokens; t++) {
        for (int h = 0; h < num_heads; h++) {
            int offset = t * query_stride + h * head_size;
            float norm_before = compute_norm(h_query_before + offset, head_size);
            float norm_after = compute_norm(h_query_after + offset, head_size);
            float diff = fabsf(norm_before - norm_after);

            if (diff > tolerance) {
                printf("  FAILED: token %d, head %d - norm before=%f, after=%f, diff=%e\n",
                       t, h, norm_before, norm_after, diff);
                pass = false;
            }
        }
    }

    if (pass) {
        printf("  PASSED: All head norms preserved (tolerance=%e)\n", tolerance);
    }

    // 정리
    free(h_query_before); free(h_query_after);
    free(h_positions); free(h_cache);
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaFree(d_cache));

    return pass;
}


/**
 * 테스트 3: Position 0 항등 테스트
 *
 * position = 0이면 θ = 0 × base^(...) = 0
 * cos(0) = 1, sin(0) = 0 → 회전 없음 (항등 변환)
 * 따라서 입력과 출력이 동일해야 함
 */
bool test_position_zero_identity() {
    printf("\n[Test 3] Position 0 Identity\n");

    // 파라미터 설정
    const int num_tokens = 1<<10;  // 1024
    const int num_heads = 64;
    const int num_kv_heads = 8;
    const int head_size = 128;
    const int rot_dim = head_size;  // 전체 head를 회전
    const int max_position = 1<<14;  // 16384

    const int query_size = num_tokens * num_heads * head_size;
    const int cache_size = max_position * rot_dim;
    const int64_t query_stride = num_heads * head_size;
    const int64_t key_stride = num_kv_heads * head_size;

    float* h_query_before = (float*)malloc(query_size * sizeof(float));
    float* h_query_after = (float*)malloc(query_size * sizeof(float));
    int64_t* h_positions = (int64_t*)malloc(num_tokens * sizeof(int64_t));
    float* h_cache = (float*)malloc(cache_size * sizeof(float));

    for (int i = 0; i < query_size; i++) {
        float val = (float)(i + 1) * 0.5f;
        h_query_before[i] = val;
        h_query_after[i] = val;
    }
    for (int i = 0; i < num_tokens; i++) {
        h_positions[i] = 0;  // 모든 토큰 position 0
    }

    generate_cos_sin_cache(h_cache, max_position, rot_dim);

    // GPU 실행
    float *d_query, *d_cache;
    int64_t *d_positions;
    CUDA_CHECK(cudaMalloc(&d_query, query_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_positions, num_tokens * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_cache, cache_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_query, h_query_after, query_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions, h_positions, num_tokens * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cache, h_cache, cache_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * rot_dim / 2, 512));

    lightvllm::rotary_embedding_kernel<float, true><<<grid, block>>>(
        d_positions, d_query, (float*)nullptr, d_cache,
        rot_dim, query_stride, key_stride,
        num_heads, num_kv_heads, head_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_query_after, d_query, query_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // position 0이면 입력과 출력이 동일해야 함
    bool pass = compare_arrays(h_query_after, h_query_before, query_size,
                               1e-6f, "Position 0 identity");

    free(h_query_before); free(h_query_after);
    free(h_positions); free(h_cache);
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaFree(d_cache));

    return pass;
}


// =============================================================================
// 메인
// =============================================================================

int main() {
    printf("=== RoPE Kernel Low-level Test ===\n");

    // GPU 정보 출력
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    int passed = 0;
    int total = 0;

    total++; if (test_basic_correctness()) passed++;
    total++; if (test_norm_preservation()) passed++;
    total++; if (test_position_zero_identity()) passed++;

    printf("\n=== Results: %d / %d passed ===\n", passed, total);

    return (passed == total) ? 0 : 1;
}
