---
name: benchmark
description: 성능 벤치마크 작성 가이드. CUDA Event 타이밍, PyTorch 비교, Non-Fused vs Fused 비교 패턴.
---

# 성능 벤치마크 작성 가이드

## Python 벤치마크 패턴

### 기본 구조

```python
def test_performance_vs_pytorch(self):
    num_tokens = 4096
    d = 11008          # LLaMA-7B intermediate_size
    warmup = 10
    repeat = 100

    input = torch.randn(num_tokens, d, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(input)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # ----- [1] 방법 1 -----
    for _ in range(warmup):
        # 연산
    torch.cuda.synchronize()

    start.record()
    for _ in range(repeat):
        # 연산
    end.record()
    torch.cuda.synchronize()
    method1_us = start.elapsed_time(end) / repeat * 1000

    # ----- [2] 방법 2 -----
    # (동일 패턴 반복)

    # 결과 출력
    print(f"\n  설정: num_tokens={num_tokens}, d={d}, dtype=...")
    print(f"  [1] 방법1: {method1_us:.1f} us")
    print(f"  [2] 방법2: {method2_us:.1f} us")
    print(f"  속도 비율: {method1_us / method2_us:.2f}x")
```

### 핵심 원칙

1. **warmup 필수**: GPU 커널 첫 실행은 JIT 컴파일, 캐시 워밍 등으로 느림. 최소 10회 warmup.

2. **torch.cuda.synchronize()**: 비동기 실행이므로 타이밍 전후에 동기화 필수.

3. **CUDA Event 사용**: `time.time()`은 CPU 시간이라 부정확. `torch.cuda.Event`가 GPU 시간을 정확히 측정.

4. **elapsed_time()은 밀리초 반환**: `/ repeat * 1000` → 마이크로초(us) 변환.

## PyTorch 비교 원칙

벤치마크에서 PyTorch 측도 동일한 조건으로 비교해야 공정하다.
`torch.xxx(out=output)` 처럼 미리 할당된 출력 텐서를 사용하는 등,
PyTorch에서 가능한 최적화를 적용한 상태로 비교한다.

## Non-Fused vs Fused 비교 구조

커널 퓨전 효과를 보여주는 표준 비교 구조:

```
[1] PyTorch Non-Fused     — 기준선
[2] CUDA Non-Fused (커널 N회) — 커널 호출만 분리
[3] CUDA Fused (커널 1회)     — 최적화 목표
```

핵심: **[1]과 [2]는 거의 동일한 성능**을 보인다.
PyTorch의 연산도 내부적으로 CUDA 커널이므로, Non-Fused 방식은
어떻게 최적화해도 동일한 메모리 접근 패턴이다.
**유일한 해결책은 [3] Fused 커널로 중간 버퍼를 제거하는 것.**

## C++ 벤치마크 패턴

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// warmup
for (int i = 0; i < warmup; i++) {
    kernel<<<grid, block>>>(args...);
}
cudaDeviceSynchronize();

// 측정
cudaEventRecord(start);
for (int i = 0; i < repeat; i++) {
    kernel<<<grid, block>>>(args...);
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
float us_per_call = ms / repeat * 1000.0f;
```
