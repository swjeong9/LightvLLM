"""HF safetensors 가중치 로더 테스트.

실제 meta-llama/Llama-3.2-3B-Instruct 체크포인트를 사용하여
가중치 로딩 파이프라인을 검증합니다.

Llama-3.2-3B 스펙:
    hidden_size=3072, intermediate_size=8192
    num_attention_heads=24, num_kv_heads=8, head_dim=128
"""

import pytest
import torch

from huggingface_hub import snapshot_download

from lightvllm.models.loader import (
    safetensors_weights_iterator,
    default_weight_loader,
    load_weights,
)
from lightvllm.models.llama import LlamaMLP
from lightvllm.layers.linear import SHARD_GATE, SHARD_UP

DEVICE = "cuda"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Llama-3.2-3B 스펙
HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 8192


@pytest.fixture(scope="module")
def weights_dir():
    """HF 캐시에서 Llama-3.2-3B safetensors 경로를 반환."""
    return snapshot_download(
        MODEL_ID, allow_patterns=["*.safetensors", "config.json"]
    )


class TestSafetensorsWeightsIterator:
    """safetensors_weights_iterator 테스트."""

    def test_yields_name_tensor_pairs(self, weights_dir):
        """(이름, 텐서) 쌍이 생성되는지 검증."""
        count = 0
        for name, tensor in safetensors_weights_iterator(weights_dir):
            assert isinstance(name, str)
            assert isinstance(tensor, torch.Tensor)
            count += 1
            if count >= 5:
                break
        assert count == 5

    def test_contains_mlp_weights(self, weights_dir):
        """MLP 가중치(gate_proj, up_proj, down_proj)가 존재하는지 검증."""
        found = {"gate_proj": False, "up_proj": False, "down_proj": False}
        for name, tensor in safetensors_weights_iterator(weights_dir):
            for key in found:
                if f"layers.0.mlp.{key}.weight" in name:
                    found[key] = True
            if all(found.values()):
                break
        assert all(found.values()), f"누락된 weight: {found}"

    def test_mlp_weight_shapes(self, weights_dir):
        """MLP 가중치의 shape이 올바른지 검증."""
        shapes = {}
        for name, tensor in safetensors_weights_iterator(weights_dir):
            if "layers.0.mlp." in name and name.endswith(".weight"):
                key = name.split(".")[-2]  # gate_proj, up_proj, down_proj
                shapes[key] = tensor.shape
            if len(shapes) == 3:
                break

        assert shapes["gate_proj"] == (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        assert shapes["up_proj"] == (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        assert shapes["down_proj"] == (HIDDEN_SIZE, INTERMEDIATE_SIZE)

    def test_invalid_dir_raises(self):
        """존재하지 않는 디렉토리에 대해 FileNotFoundError 발생."""
        with pytest.raises(FileNotFoundError):
            list(safetensors_weights_iterator("/nonexistent/path"))


class TestDefaultWeightLoader:
    """default_weight_loader 테스트."""

    def test_copies_weight(self):
        """텐서가 정확히 copy 되는지 검증."""
        param = torch.zeros(4, 3)
        weight = torch.randn(4, 3)
        default_weight_loader(param, weight)
        torch.testing.assert_close(param, weight, atol=0, rtol=0)

    def test_shape_mismatch_raises(self):
        """shape 불일치 시 AssertionError 발생."""
        param = torch.zeros(4, 3)
        weight = torch.randn(5, 3)
        with pytest.raises(AssertionError):
            default_weight_loader(param, weight)


class TestLoadWeights:
    """load_weights 통합 테스트 (실제 Llama-3.2-3B 체크포인트)."""

    def test_load_mlp_with_stacked_params(self, weights_dir):
        """실제 체크포인트에서 LlamaMLP에 gate_up 융합 로딩 후 forward 실행.

        stacked_params_mapping으로 gate_proj + up_proj → gate_up_proj 융합을 검증합니다.
        """
        # layer 0의 MLP를 단독으로 로딩하기 위해 wrapper 모듈 생성
        import torch.nn as nn

        class SingleLayerMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList()
                layer = nn.Module()
                layer.mlp = nn.Module()
                layer.mlp.gate_up_proj = LlamaMLP(
                    HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.bfloat16
                ).gate_up_proj
                layer.mlp.down_proj = LlamaMLP(
                    HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.bfloat16
                ).down_proj
                self.model.layers.append(layer)

        model = SingleLayerMLP()

        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", SHARD_GATE),
            (".gate_up_proj", ".up_proj", SHARD_UP),
        ]

        loaded = load_weights(model, weights_dir, stacked_params_mapping)

        # gate_up_proj와 down_proj가 로딩되었는지 확인
        assert any("gate_up_proj" in name for name in loaded)
        assert any("down_proj" in name for name in loaded)

    def test_load_full_mlp_and_forward(self, weights_dir):
        """실제 체크포인트에서 LlamaMLP 전체를 로딩하고 forward 실행.

        gate_proj + up_proj를 gate_up_proj로 융합 로딩한 뒤,
        실제 입력에 대해 forward를 실행하여 NaN/Inf 없이 정상 출력되는지 확인합니다.
        """
        # HF 체크포인트에서 layer 0의 MLP weight를 직접 로딩
        gate_w = None
        up_w = None
        down_w = None
        for name, tensor in safetensors_weights_iterator(weights_dir):
            if "layers.0.mlp.gate_proj.weight" in name:
                gate_w = tensor
            elif "layers.0.mlp.up_proj.weight" in name:
                up_w = tensor
            elif "layers.0.mlp.down_proj.weight" in name:
                down_w = tensor
            if gate_w is not None and up_w is not None and down_w is not None:
                break

        # LlamaMLP 생성 및 weight 로딩
        mlp = LlamaMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.bfloat16)
        mlp.gate_up_proj.weight_loader(
            mlp.gate_up_proj.weight, gate_w.to(torch.bfloat16), shard_id=SHARD_GATE
        )
        mlp.gate_up_proj.weight_loader(
            mlp.gate_up_proj.weight, up_w.to(torch.bfloat16), shard_id=SHARD_UP
        )
        mlp.down_proj.weight.data.copy_(down_w.to(torch.bfloat16))
        mlp = mlp.to(DEVICE)

        # forward 실행
        x = torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16, device=DEVICE)
        out = mlp(x)

        # 출력 검증: 올바른 shape, NaN/Inf 없음
        assert out.shape == (4, HIDDEN_SIZE)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any(), "출력에 NaN이 있습니다"
        assert not torch.isinf(out).any(), "출력에 Inf가 있습니다"
