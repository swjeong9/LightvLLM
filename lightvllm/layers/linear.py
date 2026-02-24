"""
선형 레이어: Linear, MergedLinear

LLaMA 모델의 모든 프로젝션(QKV, Output, Gate, Up, Down)의 기본 빌딩블록을 제공합니다.

- ``Linear``: F.linear 기반 단순 선형 레이어. weight를 직접 관리하여
  가중치 로딩 방식을 통일합니다.
- ``MergedLinear``: 여러 프로젝션을 하나의 GEMM으로 융합합니다.
  gate_up_proj (gate + up)이나 qkv_proj (Q + K + V)에 사용됩니다.
  ``weight_loader``로 HuggingFace 체크포인트의 별도 가중치를 융합 파라미터에 로딩합니다.

왜 별도 CUDA 커널(.cuh/.cu)이 없는가?
    Linear 연산(y = Wx + b)은 내부적으로 cuBLAS GEMM을 호출합니다.
    cuBLAS는 NVIDIA가 Tensor Core까지 활용하여 하드웨어 수준으로 최적화한
    행렬곱 라이브러리이므로, 수작업 CUDA 커널로는 이길 수 없습니다.
    vLLM도 Linear 자체에 대해서는 커스텀 커널을 작성하지 않습니다.

    반면 RMSNorm, SiLU, RoPE 등은 메모리 바운드 연산으로, 여러 단계를
    하나의 커널에 융합(fuse)하면 글로벌 메모리 왕복을 줄여 성능이 향상됩니다.
    Linear(행렬곱)은 컴퓨트 바운드 연산이라 융합할 대상 자체가 없습니다.

vLLM 참조: vLLM/vllm/model_executor/layers/linear.py
(단순화: TP, 양자화, CustomOp 없음)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# gate_up_proj의 shard ID (MLP에서 gate와 up을 구분하기 위한 상수)
# vLLM은 0, 1을 하드코딩하지만, 가독성을 위해 명시적 상수로 정의합니다.
# 사용 예) merged.weight_loader(merged.weight, gate_weight, shard_id=SHARD_GATE)
SHARD_GATE = 0
SHARD_UP = 1


class Linear(nn.Module):
    """단순 선형 레이어 (TP 없음, 양자화 없음).

    F.linear을 감싸는 얇은 래퍼. weight 파라미터를 직접 관리합니다.
    LLaMA는 bias를 사용하지 않으므로 기본값은 False입니다.

    Args:
        input_size: 입력 feature 차원.
        output_size: 출력 feature 차원.
        bias: bias 항 포함 여부 (기본값: False).
        dtype: 파라미터 dtype (기본값: 현재 default dtype).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        weight_dtype = dtype or torch.get_default_dtype()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=weight_dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(output_size, dtype=weight_dtype)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedLinear(nn.Module):
    """여러 프로젝션을 하나의 GEMM으로 융합하는 선형 레이어.

    gate_up_proj (gate_proj + up_proj)이나 qkv_proj (Q + K + V)에 사용됩니다.
    출력은 마지막 차원을 따라 split하여 각 프로젝션 결과를 분리할 수 있습니다.

    HuggingFace는 이들을 별도 가중치(예: q_proj.weight, k_proj.weight,
    v_proj.weight)로 저장합니다. ``weight_loader`` 메서드가 각 shard를
    융합 weight 행렬의 올바른 오프셋에 로딩합니다.

    왜 융합하는가? 별도의 F.linear 호출 2~3회는 CUDA 커널 launch도 2~3회입니다.
    하나의 큰 F.linear로 합치면 GEMM 1회로 처리되어, 커널 launch 오버헤드가
    줄고 GPU utilization이 향상됩니다.

    Args:
        input_size: 입력 feature 차원.
        output_sizes: 각 서브 프로젝션의 출력 크기 리스트.
            예: gate+up은 [intermediate, intermediate],
            Q+K+V는 [q_size, kv_size, kv_size].
        bias: bias 항 포함 여부 (기본값: False).
        dtype: 파라미터 dtype.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_sizes = list(output_sizes)
        total_output_size = sum(output_sizes)
        weight_dtype = dtype or torch.get_default_dtype()

        self.weight = nn.Parameter(
            torch.empty(total_output_size, input_size, dtype=weight_dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(total_output_size, dtype=weight_dtype)
            )
        else:
            self.bias = None

        # 각 shard의 시작 행(row) 오프셋을 미리 계산 (weight_loader에서 사용)
        #
        # "shard"란 융합된 weight를 구성하는 각 조각을 의미합니다.
        # HuggingFace는 q_proj, k_proj, v_proj를 별도 파일로 저장하는데,
        # 우리의 융합 weight에 각 조각을 올바른 행(row) 범위에 넣어야 합니다.
        #
        # 예) qkv_proj, output_sizes=[256, 128, 128]:
        #   offsets = [0, 256, 384, 512]
        #
        #   융합 weight 행렬:
        #   ┌──────────────────┐ row 0
        #   │  q_proj (shard 0)│         offsets[0]:offsets[1] = 0:256
        #   ├──────────────────┤ row 256
        #   │  k_proj (shard 1)│         offsets[1]:offsets[2] = 256:384
        #   ├──────────────────┤ row 384
        #   │  v_proj (shard 2)│         offsets[2]:offsets[3] = 384:512
        #   └──────────────────┘ row 512
        offsets: list[int] = [0]
        for size in output_sizes:
            offsets.append(offsets[-1] + size)
        self._offsets = offsets

        # weight_loader를 파라미터 텐서에 직접 연결.
        # load_weights()가 getattr(param, "weight_loader")로 접근할 수 있도록 합니다.
        # vLLM도 set_weight_attrs()로 동일한 패턴을 사용합니다.
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int | str,
    ) -> None:
        """HuggingFace 체크포인트에서 융합 weight의 한 shard를 로딩합니다.

        gate_up_proj: shard_id=0 (gate), shard_id=1 (up).
        qkv_proj: shard_id="q", "k", "v".

        Args:
            param: 로딩 대상 융합 파라미터 (self.weight 또는 self.bias).
            loaded_weight: 체크포인트에서 읽은 weight 텐서.
            shard_id: 이 weight가 어떤 shard에 해당하는지.
                int면 직접 인덱스 (0=gate, 1=up),
                str이면 {"q": 0, "k": 1, "v": 2}로 변환.
        """
        # shard_id를 정수 인덱스로 통일
        if isinstance(shard_id, str):
            shard_map = {"q": 0, "k": 1, "v": 2}
            shard_idx = shard_map[shard_id]
        else:
            shard_idx = shard_id

        # _offsets에서 이 shard가 차지하는 행(row) 범위를 구함
        # 예) shard_idx=1이고 offsets=[0, 256, 384, 512]이면
        #     start=offsets[1]=256, end=offsets[2]=384 → param.data[256:384]
        start = self._offsets[shard_idx]
        end = self._offsets[shard_idx + 1]
        param.data[start:end].copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
