# Ray to Kubernetes 마이그레이션 가이드

이 문서는 vLLM의 Ray 기반 분산 실행을 Kubernetes 기반으로 전환하기 위한 분석 및 설계를 담고 있다.

---

## Ray 분산 실행 분석

### 핵심 파일
- `vLLM/vllm/v1/executor/ray_executor.py` (624줄) - 메인 오케스트레이션
- `vLLM/vllm/v1/executor/ray_utils.py` (469줄) - Worker Wrapper
- `vLLM/vllm/ray/ray_env.py` (80줄) - 환경변수 관리
- `vLLM/vllm/distributed/device_communicators/ray_communicator.py` (260줄) - PP 통신

---

## Ray 핵심 기능 분석

### 1. Placement Group (리소스 배치)

**현재 Ray 구현:**
```python
PlacementGroupSchedulingStrategy(
    placement_group=placement_group,
    placement_group_bundle_index=bundle_id,
)
```

**역할:**
- GPU 리소스 예약
- 동일 노드에 TP 워커 배치
- PP 스테이지별 노드 분리

### 2. Remote Actor (Worker 생성)

**현재 Ray 구현:**
```python
worker = ray.remote(
    num_cpus=0,
    num_gpus=num_gpus,
    scheduling_strategy=strategy,
)(RayWorkerWrapper).remote(rpc_rank=rank)
```

**역할:**
- Worker 프로세스 생성
- GPU 리소스 할당
- 원격 메서드 호출

### 3. Collective RPC (집합 통신)

**현재 Ray 구현:**
```python
def collective_rpc(self, method, args, kwargs, non_block=False):
    ray_worker_outputs = [
        worker.execute_method.remote(method, *args, **kwargs)
        for worker in self.workers
    ]
    if non_block:
        return FutureWrapper(ray_worker_outputs)
    return ray.get(ray_worker_outputs)
```

**역할:**
- 모든 Worker에 동시 메서드 호출
- 결과 수집 및 동기화

### 4. Compiled DAG (최적화 실행)

**현재 Ray 구현:**
```python
def _compiled_ray_dag(self, enable_asyncio):
    from ray.dag import InputNode, MultiOutputNode

    with InputNode() as input_data:
        outputs = [input_data for _ in self.pp_tp_workers[0]]

        for pp_rank, tp_group in enumerate(self.pp_tp_workers):
            outputs = [
                worker.execute_model_ray.bind(outputs[i])
                for i, worker in enumerate(tp_group)
            ]

            if pp_rank < last_pp_rank:
                outputs = [
                    output.with_tensor_transport(transport="nccl")
                    for output in outputs
                ]

        forward_dag = MultiOutputNode(outputs)

    return forward_dag.experimental_compile(
        _overlap_gpu_communication=True,
    )
```

**역할:**
- PP 스테이지 간 데이터 흐름 정의
- 중간 텐서 전송 최적화 (NCCL/SharedMemory)
- GPU 통신 오버랩

---

## Kubernetes 대체 설계

### 1. Placement Group → Pod Affinity/Anti-Affinity

**K8s 설계:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: llm-worker
spec:
  replicas: 8  # TP=4, PP=2 → 8 workers
  template:
    spec:
      affinity:
        # TP 워커는 같은 노드에 배치
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                tp-group: "0"
            topologyKey: kubernetes.io/hostname
        # PP 스테이지는 다른 노드에 분산
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  pp-stage: "1"
              topologyKey: kubernetes.io/hostname
      containers:
      - name: worker
        resources:
          limits:
            nvidia.com/gpu: 1
```

### 2. Remote Actor → StatefulSet + gRPC Service

**Worker Service 정의:**
```protobuf
syntax = "proto3";

service LLMWorker {
    // 워커 초기화
    rpc InitWorker(InitRequest) returns (InitResponse);

    // 모델 로딩
    rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);

    // Forward 실행
    rpc ExecuteModel(ExecuteModelRequest) returns (ExecuteModelResponse);

    // Forward 실행 (스트리밍)
    rpc ExecuteModelStream(ExecuteModelRequest) returns (stream IntermediateTensor);

    // 상태 확인
    rpc HealthCheck(HealthRequest) returns (HealthResponse);
}

message ExecuteModelRequest {
    bytes scheduler_output = 1;
    bytes grammar_output = 2;
    bytes intermediate_tensors = 3;  // PP 이전 스테이지 출력
}

message ExecuteModelResponse {
    bytes model_output = 1;
    bytes intermediate_tensors = 2;  // PP 다음 스테이지용
}
```

**K8s Service:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-worker-headless
spec:
  clusterIP: None
  selector:
    app: llm-worker
  ports:
  - port: 50051
    name: grpc
---
# 각 Worker는 DNS로 접근 가능
# llm-worker-0.llm-worker-headless.default.svc.cluster.local
```

### 3. Collective RPC → gRPC Fan-out/Fan-in

**Python 구현:**
```python
import grpc
import asyncio
from concurrent.futures import ThreadPoolExecutor

class K8sDistributedExecutor:
    def __init__(self, worker_addresses: list[str]):
        self.channels = [
            grpc.aio.insecure_channel(addr)
            for addr in worker_addresses
        ]
        self.stubs = [
            LLMWorkerStub(channel)
            for channel in self.channels
        ]

    async def collective_rpc(self, method: str, args: tuple):
        """모든 Worker에 동시 호출"""
        tasks = [
            getattr(stub, method)(*args)
            for stub in self.stubs
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def init_all_workers(self, config):
        return await self.collective_rpc(
            "InitWorker",
            (InitRequest(config=config),)
        )
```

### 4. Compiled DAG → Custom Controller + CRD

**Custom Resource Definition:**
```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: llminferences.lightvllm.io
spec:
  group: lightvllm.io
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              model:
                type: string
              tensorParallelSize:
                type: integer
              pipelineParallelSize:
                type: integer
              replicas:
                type: integer
          status:
            type: object
            properties:
              phase:
                type: string
              workers:
                type: array
                items:
                  type: object
                  properties:
                    name:
                      type: string
                    rank:
                      type: integer
                    ready:
                      type: boolean
  scope: Namespaced
  names:
    plural: llminferences
    singular: llminference
    kind: LLMInference
    shortNames:
    - llmi
```

**사용 예시:**
```yaml
apiVersion: lightvllm.io/v1
kind: LLMInference
metadata:
  name: llama-7b-inference
spec:
  model: meta-llama/Llama-2-7b-hf
  tensorParallelSize: 4
  pipelineParallelSize: 2
  replicas: 1
```

**Controller 로직:**
```python
class LLMInferenceController:
    async def reconcile(self, inference: LLMInference):
        """CRD 상태 동기화"""

        # 1. Worker StatefulSet 생성/업데이트
        await self.ensure_worker_statefulset(inference)

        # 2. Worker 준비 대기
        await self.wait_for_workers_ready(inference)

        # 3. Rank 할당
        await self.assign_ranks(inference)

        # 4. 분산 초기화
        await self.init_distributed(inference)

        # 5. 모델 로딩
        await self.load_model(inference)

        # 6. 상태 업데이트
        await self.update_status(inference, "Ready")
```

### 5. Worker Discovery → K8s API + DNS

**Worker 검색:**
```python
from kubernetes import client, config

class WorkerDiscovery:
    def __init__(self, namespace: str, statefulset_name: str):
        config.load_incluster_config()
        self.v1 = client.CoreV1Api()
        self.namespace = namespace
        self.statefulset_name = statefulset_name

    def get_worker_addresses(self) -> list[str]:
        """모든 Worker Pod의 주소 반환"""
        pods = self.v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"app={self.statefulset_name}"
        )

        addresses = []
        for pod in pods.items:
            # StatefulSet DNS 형식
            hostname = f"{pod.metadata.name}.{self.statefulset_name}-headless"
            addresses.append(f"{hostname}:50051")

        return sorted(addresses)  # 순서 보장

    def get_worker_nodes(self) -> dict[str, str]:
        """Worker -> Node 매핑"""
        pods = self.v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"app={self.statefulset_name}"
        )
        return {
            pod.metadata.name: pod.spec.node_name
            for pod in pods.items
        }
```

### 6. 환경변수 전파 → ConfigMap/Secret

**ConfigMap:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-inference-config
data:
  TENSOR_PARALLEL_SIZE: "4"
  PIPELINE_PARALLEL_SIZE: "2"
  MODEL_NAME: "meta-llama/Llama-2-7b-hf"
  BLOCK_SIZE: "16"
  MAX_NUM_SEQS: "256"
---
apiVersion: v1
kind: Secret
metadata:
  name: llm-inference-secrets
type: Opaque
data:
  HF_TOKEN: <base64-encoded-token>
```

**Pod에 마운트:**
```yaml
spec:
  containers:
  - name: worker
    envFrom:
    - configMapRef:
        name: llm-inference-config
    - secretRef:
        name: llm-inference-secrets
    env:
    - name: RANK
      valueFrom:
        fieldRef:
          fieldPath: metadata.labels['rank']
```

---

## 중간 텐서 전송 방안

### Option 1: gRPC Streaming

```python
async def execute_model_stream(
    self,
    request: ExecuteModelRequest
) -> AsyncIterator[IntermediateTensor]:
    """PP 스테이지 간 텐서 스트리밍"""

    # Forward 실행
    output = self.model_runner.execute_model(
        request.scheduler_output,
        request.intermediate_tensors
    )

    if isinstance(output, IntermediateTensors):
        # 다음 PP 스테이지로 전송
        for tensor_name, tensor in output.tensors.items():
            yield IntermediateTensor(
                name=tensor_name,
                data=tensor.cpu().numpy().tobytes(),
                shape=list(tensor.shape),
                dtype=str(tensor.dtype)
            )
```

### Option 2: RDMA/GPUDirect (고성능)

```python
# NCCL ProcessGroup을 직접 사용
import torch.distributed as dist

class NCCLCommunicator:
    def __init__(self, rank: int, world_size: int):
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )
        self.pp_group = dist.new_group(...)

    def send_to_next_stage(self, tensor: torch.Tensor, dst: int):
        dist.send(tensor, dst, group=self.pp_group)

    def recv_from_prev_stage(self, tensor: torch.Tensor, src: int):
        dist.recv(tensor, src, group=self.pp_group)
```

### Option 3: 공유 스토리지 (NFS/PVC)

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tensor-exchange
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: nfs
---
# Pod에 마운트
volumes:
- name: tensor-exchange
  persistentVolumeClaim:
    claimName: tensor-exchange
```

---

## 구현 단계

### Step 1: 단일 노드 멀티 GPU (TP only)

```
목표: torch.distributed만 사용, K8s 없이 로컬 테스트

구현:
1. NCCL ProcessGroup 초기화
2. TP-aware Linear Layer 구현
3. AllReduce/AllGather 통신
4. 단일 노드에서 검증
```

### Step 2: 멀티 노드 gRPC 통신 (PP)

```
목표: gRPC로 PP 스테이지 간 통신

구현:
1. Worker gRPC 서비스 구현
2. 중간 텐서 직렬화/역직렬화
3. PP 스테이지 동기화
4. 수동으로 여러 노드에서 실행
```

### Step 3: Kubernetes 통합

```
목표: CRD + Controller로 자동화

구현:
1. CRD 정의
2. Controller 구현
3. StatefulSet 자동 생성
4. Rank 자동 할당
5. 상태 모니터링
```

### Step 4: 최적화

```
목표: 성능 최적화

구현:
1. NCCL 직접 사용 (PP 통신)
2. 텐서 전송 오버랩
3. 동적 스케일링
4. 장애 복구
```

---

## 아키텍처 비교

| 항목 | Ray | Kubernetes |
|------|-----|------------|
| 리소스 관리 | Placement Group | StatefulSet + Affinity |
| Worker 생성 | ray.remote | Pod 생성 |
| 집합 통신 | collective_rpc | gRPC Fan-out |
| 텐서 전송 | Ray Channel | gRPC/NCCL |
| 상태 관리 | Ray Actor | CRD + Controller |
| 장애 복구 | Ray 자동 | K8s 자동 |
| 스케일링 | 수동 | HPA 가능 |

---

## 예상 프로젝트 구조

```
lightvllm/
├── distributed/
│   ├── k8s/
│   │   ├── controller.py        # K8s Controller
│   │   ├── crd.py               # CRD 정의
│   │   ├── discovery.py         # Worker 검색
│   │   └── executor.py          # K8s Executor
│   ├── grpc/
│   │   ├── proto/
│   │   │   └── worker.proto     # gRPC 정의
│   │   ├── server.py            # gRPC 서버
│   │   └── client.py            # gRPC 클라이언트
│   ├── tensor_parallel.py       # TP 구현
│   └── pipeline_parallel.py     # PP 구현
│
├── kubernetes/
│   ├── crd/
│   │   └── llminference.yaml
│   ├── controller/
│   │   └── deployment.yaml
│   └── worker/
│       ├── statefulset.yaml
│       ├── service.yaml
│       └── configmap.yaml
```

---

## 참고 자료

- Kubernetes Operator Pattern: https://kubernetes.io/docs/concepts/extend-kubernetes/operator/
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- gRPC Python: https://grpc.io/docs/languages/python/
- NCCL: https://developer.nvidia.com/nccl
