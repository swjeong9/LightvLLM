/**
 * =============================================================================
 * LightvLLM Position Encoding CUDA 커널
 * =============================================================================
 *
 * 이 파일은 Rotary Position Embedding (RoPE)을 구현합니다.
 * RoPE는 LLaMA, GPT-NeoX 등 최신 LLM에서 사용하는 위치 인코딩 방식입니다.
 *
 * 목차:
 * 1. RoPE가 필요한 이유
 * 2. RoPE의 수학적 원리
 * 3. 구현 세부사항
 * 4. CUDA 커널 코드
 */

/**
 * =============================================================================
 * 1. RoPE가 필요한 이유
 * =============================================================================
 *
 * [Transformer의 위치 정보 문제]
 *
 * Transformer의 Self-Attention은 순서를 무시합니다:
 *   Attention(Q, K, V) = softmax(QK^T / √d) V
 *
 * "나는 밥을 먹는다"와 "밥을 나는 먹는다"가 같은 결과를 낼 수 있음!
 * → 토큰의 위치 정보를 어떻게든 주입해야 함
 *
 *
 * [기존 방식들의 발전]
 *
 * 1. 원본 Transformer (Vaswani et al., 2017):
 *    - Sinusoidal Positional Encoding 사용 (고정, 학습하지 않음)
 *    - PE(pos, 2i)   = sin(pos / 10000^(2i/d))
 *    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
 *    - 장점: 외삽 가능, 상대적 위치를 선형 변환으로 표현 가능
 *    - 논문: "PE_{pos+k}는 PE_{pos}의 선형 함수로 표현 가능"
 *
 * 2. BERT, GPT 등 후속 모델:
 *    - Learnable Positional Embedding 사용 (학습)
 *    - 더 유연하지만, 학습 시 본 최대 길이를 넘으면 외삽 어려움
 *
 * 3. 공통적인 한계:
 *    - Embedding에 "더하는" 방식 → 위치 정보와 의미 정보가 섞임
 *    - Attention 계산 시 상대적 위치를 명시적으로 반영하지 않음
 *
 *
 * [RoPE의 해결책]
 *
 * RoPE (Su et al., 2021)는 근본적으로 다른 접근:
 * - Embedding에 더하지 않고, Q와 K를 **회전**시킴
 * - 회전 행렬의 수학적 성질로 상대적 위치가 자연스럽게 인코딩됨
 * - Sinusoidal의 외삽 장점 + 상대적 위치의 명시적 모델링
 *
 *
 * =============================================================================
 * 2. RoPE의 수학적 원리
 * =============================================================================
 *
 * [핵심 아이디어: 2D 회전]
 *
 * 2차원 벡터 [x, y]를 각도 θ만큼 회전:
 *
 *   [x']   [cos θ  -sin θ] [x]   [x·cos θ - y·sin θ]
 *   [y'] = [sin θ   cos θ] [y] = [x·sin θ + y·cos θ]
 *
 * RoPE는 이 회전을 고차원 벡터에 적용합니다.
 *
 *
 * [고차원으로 확장]
 *
 * head_size가 d일 때, 벡터를 d/2개의 2D 쌍으로 나눕니다:
 *
 *   [x0, x1, x2, x3, ..., x_{d-2}, x_{d-1}]
 *    ↓    ↓
 *   (x0, x1) → 첫 번째 2D 쌍, 각도 θ_0로 회전
 *   (x2, x3) → 두 번째 2D 쌍, 각도 θ_1로 회전
 *   ...
 *
 *
 * [각도 θ 계산]
 *
 * 각 차원 쌍 i에 대해:
 *   θ_i = position × base^(-2i/d)
 *
 * 여기서:
 * - position: 토큰의 위치 (0, 1, 2, ...)
 * - base: 기본값 10000 (하이퍼파라미터)
 * - d: head dimension
 * - i: 차원 쌍 인덱스 (0, 1, ..., d/2-1)
 *
 * 낮은 차원(작은 i)은 빠르게 회전 (고주파)
 * 높은 차원(큰 i)은 느리게 회전 (저주파)
 *
 *
 * [cos/sin 캐시]
 *
 * θ 계산은 position과 dimension에만 의존하므로,
 * 미리 계산해서 캐시로 저장해둡니다:
 *
 *   cos_sin_cache[pos, :] = [cos(θ_0), cos(θ_1), ..., sin(θ_0), sin(θ_1), ...]
 *
 *
 * [왜 Q와 K에만 적용하는가?]
 *
 * Attention score 계산: Q @ K^T
 *
 * Q_m = R(m) × q  (position m에서의 query)
 * K_n = R(n) × k  (position n에서의 key)
 *
 * Q_m @ K_n^T = q^T × R(m)^T × R(n) × k
 *             = q^T × R(n-m) × k
 *
 * 회전 행렬의 성질로 인해, 결과는 **상대적 위치 (n-m)**에만 의존!
 * → 상대적 위치 정보가 자연스럽게 인코딩됨
 *
 *
 * =============================================================================
 * 3. 구현 세부사항
 * =============================================================================
 *
 * [두 가지 구현 스타일]
 *
 * 1. GPT-NeoX 스타일 (IS_NEOX = true):
 *    벡터를 반으로 나눠서 회전
 *    [x0, x1, x2, x3] → (x0, x2), (x1, x3) 쌍으로 회전
 *
 * 2. GPT-J 스타일 (IS_NEOX = false):
 *    인접한 원소끼리 쌍
 *    [x0, x1, x2, x3] → (x0, x1), (x2, x3) 쌍으로 회전
 *
 * LLaMA는 GPT-NeoX 스타일을 사용합니다.
 *
 *
 * [메모리 레이아웃]
 *
 * Query: [num_tokens, num_heads, head_size]
 * Key:   [num_tokens, num_kv_heads, head_size]
 *
 * GQA (Grouped Query Attention)에서 num_kv_heads < num_heads 가능
 *
 *
 * [병렬화 전략]
 *
 * - 각 블록이 하나의 토큰 처리
 * - 블록 내 스레드들이 협력하여 모든 head의 회전 수행
 *
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace lightvllm {

/**
 * =============================================================================
 * 4. CUDA 커널 코드
 * =============================================================================
 */

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


/**
 * =============================================================================
 * PyTorch 연동 함수
 * =============================================================================
 */

/**
 * rotary_embedding - Python에서 호출되는 진입점
 *
 * @param positions      토큰 위치 텐서 [num_tokens]
 * @param query          Query 텐서 [num_tokens, num_heads, head_size]
 * @param key            Key 텐서 (optional) [num_tokens, num_kv_heads, head_size]
 * @param head_size      각 head의 차원
 * @param cos_sin_cache  cos/sin 캐시 [max_position, rot_dim]
 * @param is_neox        true: GPT-NeoX 스타일 (LLaMA), false: GPT-J 스타일
 */
void rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox) {

    // 토큰 수 계산
    int64_t num_tokens = positions.numel();

    // head 수 계산
    int64_t query_hidden_size = query.numel() / num_tokens;
    int64_t key_hidden_size = key.numel() / num_tokens;
    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key_hidden_size / head_size;

    // 회전 차원
    int rot_dim = cos_sin_cache.size(1);

    // stride 계산 (토큰 간 거리)
    int64_t query_stride = query.stride(0);
    int64_t key_stride = key.stride(0);

    // 커널 launch 설정
    dim3 grid(num_tokens);  // 각 블록 = 하나의 토큰
    dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));

    // CUDA 디바이스 가드 (멀티 GPU 지원)
    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // dtype에 따라 적절한 커널 호출
    VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
        if (is_neox) {
            lightvllm::rotary_embedding_kernel<scalar_t, true>
                <<<grid, block, 0, stream>>>(
                    positions.data_ptr<int64_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<scalar_t>(),
                    rot_dim,
                    query_stride,
                    key_stride,
                    num_heads,
                    num_kv_heads,
                    head_size);
        } else {
            lightvllm::rotary_embedding_kernel<scalar_t, false>
                <<<grid, block, 0, stream>>>(
                    positions.data_ptr<int64_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<scalar_t>(),
                    rot_dim,
                    query_stride,
                    key_stride,
                    num_heads,
                    num_kv_heads,
                    head_size);
        }
    });
}
