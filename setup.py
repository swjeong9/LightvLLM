"""
LightvLLM CUDA Extension 빌드 스크립트

사용법:
    uv run python setup.py build_ext --inplace

빌드 결과:
    lightvllm/_C.cpython-*.so 가 생성되어
    Python에서 import lightvllm._C 로 사용 가능
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="lightvllm",
    ext_modules=[
        CUDAExtension(
            name="lightvllm._C",
            sources=[
                "csrc/torch_bindings.cpp",
                "csrc/pos_encoding_kernels.cu",
                "csrc/layernorm_kernels.cu",
            ],
            include_dirs=["csrc"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
