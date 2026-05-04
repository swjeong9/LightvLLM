"""
HuggingFace safetensors 가중치 로더.

HuggingFace 체크포인트(.safetensors)에서 가중치를 읽어 모델 파라미터에 로딩합니다.

로딩 흐름::

    safetensors 파일들
        ↓  safetensors_weights_iterator()
    (이름, 텐서) 생성기
        ↓  load_weights()
    stacked_params_mapping 으로 이름 변환
      예) "model.layers.0.mlp.gate_proj.weight"
          → "model.layers.0.mlp.gate_up_proj.weight" (shard_id=SHARD_GATE)
        ↓
    param.weight_loader(param, tensor, shard_id)  ← MergedLinear용
    또는 default_weight_loader(param, tensor)      ← 1:1 매핑용

stacked_params_mapping이란?
    HuggingFace는 gate_proj, up_proj를 별도 텐서로 저장하지만,
    우리 모델은 gate_up_proj로 융합합니다. 이 매핑 테이블이
    HF 이름 → 융합 파라미터 이름 + shard_id 변환을 정의합니다.

    매핑은 모델이 정의합니다 (모델별로 융합 구조가 다를 수 있으므로).
    loader는 모델에 무관한 범용 함수입니다.

vLLM 참조:
    - vLLM/vllm/model_executor/model_loader/weight_utils.py
    - vLLM/vllm/model_executor/models/llama.py (load_weights)
"""

import glob
from collections.abc import Generator

import torch
import torch.nn as nn
from safetensors.torch import safe_open


def safetensors_weights_iterator(
    weights_dir: str,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """safetensors 파일에서 (이름, 텐서) 쌍을 순차 생성.

    디렉토리 내의 모든 .safetensors 파일을 정렬 순서로 순회하며,
    각 파일의 텐서를 lazy하게 로딩합니다 (메모리 효율적).

    Args:
        weights_dir: safetensors 파일이 있는 디렉토리 경로.

    Yields:
        (파라미터 이름, 텐서) 튜플.
    """
    files = sorted(glob.glob(f"{weights_dir}/*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"safetensors 파일을 찾을 수 없습니다: {weights_dir}"
        )
    for path in files:
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def default_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    """1:1 shape 매칭 시 단순 copy.

    융합되지 않은 파라미터(down_proj, layernorm, embedding 등)에 사용됩니다.

    Args:
        param: 모델의 파라미터 텐서 (in-place 수정).
        loaded_weight: 체크포인트에서 읽은 텐서.

    Raises:
        AssertionError: shape이 일치하지 않을 때.
    """
    assert param.size() == loaded_weight.size(), (
        f"Shape 불일치: param={param.size()}, weight={loaded_weight.size()}"
    )
    param.data.copy_(loaded_weight)


def load_weights(
    model: nn.Module,
    weights_dir: str,
    stacked_params_mapping: list[tuple[str, str, int | str]] | None = None,
) -> set[str]:
    """HF safetensors 체크포인트를 모델에 로딩.

    stacked_params_mapping을 사용하여 HF의 별도 가중치를
    모델의 융합 파라미터에 올바르게 로딩합니다.

    용어 정리::

        param         : 우리 모델의 파라미터 (torch.empty로 생성된 빈 텐서).
                        여기에 값을 채워넣는 것이 이 함수의 목적.
        loaded_weight : HF 체크포인트에서 읽어온 실제 가중치 값.
                        param에 copy된 후, generator가 다음으로 넘어가면
                        참조가 끊겨 GC 대상이 됨.

    Args:
        model: 가중치를 로딩할 nn.Module.
        weights_dir: safetensors 파일 디렉토리.
        stacked_params_mapping: (param_suffix, weight_suffix, shard_id) 리스트.
            - param_suffix  : 우리 모델의 융합 파라미터 접미사
            - weight_suffix : HF 체크포인트의 원래 접미사
            - shard_id      : 융합 weight에서 이 조각의 위치 (int 또는 str)

            예) MLP gate+up 융합::

                stacked_params_mapping = [
                    (".gate_up_proj", ".gate_proj", SHARD_GATE),  # 0
                    (".gate_up_proj", ".up_proj",   SHARD_UP),    # 1
                ]

                HF "...gate_proj.weight" → 우리 "...gate_up_proj.weight" shard 0
                HF "...up_proj.weight"   → 우리 "...gate_up_proj.weight" shard 1

            예) Attention QKV 융합::

                stacked_params_mapping = [
                    (".qkv_proj", ".q_proj", "q"),
                    (".qkv_proj", ".k_proj", "k"),
                    (".qkv_proj", ".v_proj", "v"),
                ]

    Returns:
        로딩된 파라미터 이름 집합.
    """
    if stacked_params_mapping is None:
        stacked_params_mapping = []

    # 모델의 모든 파라미터를 {이름: nn.Parameter} 딕셔너리로 수집.
    # 예) {"model.layers.0.mlp.gate_up_proj.weight": Parameter(...),
    #      "model.layers.0.mlp.down_proj.weight": Parameter(...), ...}
    params_dict = dict(model.named_parameters())
    loaded_params: set[str] = set()

    for name, loaded_weight in safetensors_weights_iterator(weights_dir):
        # RoPE 캐시 텐서는 건너뜀 (모델이 자체 생성)
        if "rotary_emb.inv_freq" in name:
            continue

        # --- stacked_params_mapping으로 융합 파라미터 처리 ---
        #
        # 매핑 테이블을 순회하며, 현재 HF 가중치 이름(name)이
        # 어떤 융합 파라미터에 해당하는지 찾는다.
        matched = False
        for param_suffix, weight_suffix, shard_id in stacked_params_mapping:
            # 이 매핑 규칙이 현재 가중치와 관련 없으면 다음 규칙으로.
            # 예) name="...up_proj.weight"인데 weight_suffix=".gate_proj"이면 skip
            if weight_suffix not in name:
                continue

            # HF 이름 → 융합 파라미터 이름으로 변환
            # 예) "model.layers.0.mlp.gate_proj.weight"
            #   → "model.layers.0.mlp.gate_up_proj.weight"
            fused_name = name.replace(weight_suffix, param_suffix)

            # 변환된 이름이 우리 모델에 없으면 skip.
            # weight_suffix 매칭은 부분 문자열 검사라 오탐이 가능하다.
            # 예) weight_suffix=".gate_proj"이면 MLP뿐 아니라
            #     "self_attn.gate_proj"도 매칭됨 → 변환 결과가 params_dict에
            #     없으면 이 매핑 규칙이 적용 대상이 아닌 것이므로 skip.
            if fused_name not in params_dict:
                continue

            param = params_dict[fused_name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                # MergedLinear: shard_id에 해당하는 offset 범위에 부분 copy
                weight_loader(param, loaded_weight, shard_id)
            else:
                # weight_loader가 없으면 1:1 전체 copy (fallback)
                default_weight_loader(param.data, loaded_weight)
            loaded_params.add(fused_name)
            matched = True
            break

        if matched:
            continue

        # --- 1:1 매핑 (융합 아닌 일반 파라미터) ---
        # down_proj, layernorm, embedding 등 이름과 shape이 그대로 대응되는 경우.
        # 우리 모델에 없는 파라미터(다른 레이어, lm_head 등)는 skip.
        if name not in params_dict:
            continue
        param = params_dict[name]
        default_weight_loader(param.data, loaded_weight)
        loaded_params.add(name)

    return loaded_params
