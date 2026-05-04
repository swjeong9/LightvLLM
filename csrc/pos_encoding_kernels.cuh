/**
 * =============================================================================
 * LightvLLM Position Encoding CUDA 커널 (순수 CUDA, PyTorch 의존 없음)
 * =============================================================================
 *
 * 이 헤더는 RoPE의 순수 CUDA 커널 코드만 포함합니다.
 * PyTorch 의존성이 없으므로 테스트에서도 직접 include 가능합니다.
 *
 * 사용처:
 *   - csrc/pos_encoding_kernels.cu  (PyTorch wrapper에서 include)
 *   - tests/test_rope_kernel.cu     (low-level 테스트에서 include)
 */

#pragma once

#include <cstdint>
#include "cuda_compat.h"

namespace lightvllm {

/**
 * apply_token_rotary_embedding - 단일 2D 쌍에 회전 적용
 *
 * 하나의 (x, y) 쌍을 각도 θ만큼 회전시킵니다.
 *
 * @tparam scalar_t  데이터 타입 (float, half, bfloat16)
 * @tparam IS_NEOX   true: GPT-NeoX 스타일, false: GPT-J 스타일
 *
 * @param arr        회전을 적용할 벡터 (in-place 수정)
 * @param cos_ptr    cos 값 배열
 * @param sin_ptr    sin 값 배열
 * @param rot_offset 현재 처리 중인 회전 쌍의 인덱스
 * @param embed_dim  회전 차원의 절반 (rot_dim / 2)
 */
template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {

    int x_index, y_index;
    scalar_t cos, sin;

    if (IS_NEOX) {
        /**
         * GPT-NeoX 스타일 (LLaMA 사용)
         *
         * 벡터: [a0, a1, a2, a3, | b0, b1, b2, b3]
         *        ←  전반부  →    ←  후반부  →
         *
         * 쌍 구성: (a0, b0), (a1, b1), (a2, b2), (a3, b3)
         *
         * rot_offset = 0이면 (a0, b0) 처리
         * rot_offset = 1이면 (a1, b1) 처리
         */
        x_index = rot_offset;              // 전반부에서 가져옴
        y_index = embed_dim + rot_offset;  // 후반부에서 가져옴
        cos = VLLM_LDG(cos_ptr + x_index);
        sin = VLLM_LDG(sin_ptr + x_index);
    } else {
        /**
         * GPT-J 스타일
         *
         * 벡터: [a0, b0, a1, b1, a2, b2, a3, b3]
         *
         * 쌍 구성: (a0, b0), (a1, b1), (a2, b2), (a3, b3)
         * 인접한 원소끼리 쌍
         *
         * rot_offset = 0이면 인덱스 0, 1 처리
         * rot_offset = 1이면 인덱스 2, 3 처리
         */
        x_index = 2 * rot_offset;
        y_index = 2 * rot_offset + 1;
        cos = VLLM_LDG(cos_ptr + x_index / 2);
        sin = VLLM_LDG(sin_ptr + x_index / 2);
    }

    /**
     * 2D 회전 변환 적용
     *
     * [x']   [cos  -sin] [x]
     * [y'] = [sin   cos] [y]
     *
     * x' = x·cos - y·sin
     * y' = x·sin + y·cos
     */
    const scalar_t x = arr[x_index];
    const scalar_t y = arr[y_index];
    arr[x_index] = x * cos - y * sin;
    arr[y_index] = x * sin + y * cos;
}


/**
 * apply_rotary_embedding - 하나의 토큰에 대해 모든 head에 RoPE 적용
 *
 * 블록 내 스레드들이 협력하여 처리합니다.
 *
 * @param query         Query 텐서 포인터
 * @param key           Key 텐서 포인터 (nullptr 가능)
 * @param cache_ptr     해당 position의 cos/sin 캐시 포인터
 * @param head_size     각 head의 차원
 * @param num_heads     Query head 수
 * @param num_kv_heads  Key/Value head 수 (GQA에서 더 적을 수 있음)
 * @param rot_dim       회전을 적용할 차원 수
 * @param token_idx     현재 토큰 인덱스
 * @param query_stride  Query에서 토큰 간 stride
 * @param key_stride    Key에서 토큰 간 stride
 */
template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* cache_ptr,
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int64_t query_stride,
    const int64_t key_stride) {

    // embed_dim = rot_dim / 2 (회전 쌍의 수)
    const int embed_dim = rot_dim / 2;

    // 캐시 레이아웃: [cos_0, cos_1, ..., cos_{d/2-1}, sin_0, sin_1, ..., sin_{d/2-1}]
    const scalar_t* cos_ptr = cache_ptr;
    const scalar_t* sin_ptr = cache_ptr + embed_dim;

    /**
     * Query 처리
     *
     * 총 처리할 회전 쌍: num_heads × embed_dim
     * 각 스레드가 일부를 담당
     */
    const int nq = num_heads * embed_dim;
    for (int i = threadIdx.x; i < nq; i += blockDim.x) {
        // i번째 작업에서 처리할 head와 회전 위치 계산
        const int head_idx = i / embed_dim;
        const int rot_offset = i % embed_dim;

        // 해당 토큰, 해당 head의 시작 위치
        const int64_t token_head = token_idx * query_stride + head_idx * head_size;

        apply_token_rotary_embedding<scalar_t, IS_NEOX>(
            query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }

    /**
     * Key 처리 (key가 있는 경우에만)
     *
     * GQA에서 num_kv_heads < num_heads 가능
     */
    if (key != nullptr) {
        const int nk = num_kv_heads * embed_dim;
        for (int i = threadIdx.x; i < nk; i += blockDim.x) {
            const int head_idx = i / embed_dim;
            const int rot_offset = i % embed_dim;

            const int64_t token_head = token_idx * key_stride + head_idx * head_size;

            apply_token_rotary_embedding<scalar_t, IS_NEOX>(
                key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
        }
    }
}


/**
 * rotary_embedding_kernel - RoPE 메인 CUDA 커널
 *
 * 병렬화 전략:
 * - Grid: (num_tokens,) - 각 블록이 하나의 토큰 담당
 * - Block: (min(num_heads * rot_dim/2, 512),) - 스레드들이 협력
 *
 * @param positions      각 토큰의 위치 [num_tokens]
 * @param query          Query 텐서 [num_tokens, num_heads, head_size]
 * @param key            Key 텐서 (nullable) [num_tokens, num_kv_heads, head_size]
 * @param cos_sin_cache  미리 계산된 cos/sin 값 [max_position, rot_dim]
 * @param rot_dim        회전 차원 수
 * @param query_stride   Query 토큰 간 stride
 * @param key_stride     Key 토큰 간 stride
 * @param num_heads      Query head 수
 * @param num_kv_heads   Key/Value head 수
 * @param head_size      각 head의 차원
 */
template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos_sin_cache,
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {

    // 각 블록이 하나의 토큰 처리
    const int token_idx = blockIdx.x;

    // 해당 토큰의 position 가져오기
    const int64_t pos = positions[token_idx];

    // 해당 position의 cos/sin 캐시 포인터
    // 캐시 레이아웃: [max_position, rot_dim]
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    // 회전 적용
    apply_rotary_embedding<scalar_t, IS_NEOX>(
        query, key, cache_ptr, head_size, num_heads, num_kv_heads,
        rot_dim, token_idx, query_stride, key_stride);
}

}  // namespace lightvllm
