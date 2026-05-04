# Phase 1: Q&A

개발 과정에서 나온 질문과 답변을 정리한 문서입니다.

---

## CUDA / GPU

### Q: GPU 드라이버가 깨지는 이유는?

AWS EC2 Deep Learning AMI에서 빈번하게 발생하는 문제입니다.

**원인**: 커널이 자동 업데이트(`apt upgrade` 또는 `unattended-upgrades`)되면서 NVIDIA DKMS 모듈이 새 커널에 대해 재빌드에 실패하는 것이 원인입니다.

**진단 방법**:
```bash
uname -r       # 현재 실행 중인 커널 버전
dkms status    # NVIDIA 드라이버가 어떤 커널용으로 빌드되어 있는지
```
이 두 값이 다르면 "커널은 업데이트됐는데 드라이버가 안 따라간" 상태입니다.

**복구 방법**:
```bash
sudo apt install linux-headers-$(uname -r)
sudo dkms install nvidia/<version> -k $(uname -r)
sudo modprobe nvidia
nvidia-smi
```

**예방**:
```bash
sudo apt-mark hold linux-image-$(uname -r) linux-headers-$(uname -r)
```

### Q: DKMS가 뭐야?

**DKMS (Dynamic Kernel Module Support)** — 커널 업데이트 시 외부 커널 모듈(예: NVIDIA 드라이버)을 자동으로 재빌드해주는 프레임워크입니다.

NVIDIA 드라이버는 `.ko` (kernel object) 파일로 커널에 로드되는데, 이 파일은 빌드된 커널 버전에 종속적입니다. DKMS는 `/usr/src/nvidia-xxx/`에 소스를 보관해두고, 새 커널이 설치될 때 자동으로 재컴파일합니다. 이 재빌드가 실패하면 드라이버가 깨집니다.

---

## 빌드 시스템

### Q: `setup.py`가 무슨 역할을 하는 거지?

C++/CUDA 소스 코드를 Python이 import할 수 있는 `.so` (shared object) 파일로 컴파일해주는 빌드 스크립트입니다.

```python
CUDAExtension(
    name="lightvllm._C",
    sources=["csrc/torch_bindings.cpp", "csrc/pos_encoding_kernels.cu"],
    include_dirs=["csrc"],
)
```

내부적으로 하는 일:
1. `nvcc`로 `.cu` 파일 컴파일 → `.o`
2. `g++`로 `.cpp` 파일 컴파일 → `.o`
3. `g++`로 `.o` 파일들을 링크 → `_C.cpython-312-x86_64-linux-gnu.so`
4. `lightvllm/` 폴더에 `.so` 파일 배치

순수 Python 패키지는 `setup.py`가 필요 없지만, C/CUDA extension은 컴파일이 필요하므로 `setup.py`가 필요합니다.

### Q: `lightvllm.egg-info`는 왜 생기는 거지?

`setup.py build_ext --inplace`를 실행하면 setuptools가 자동 생성하는 패키지 메타데이터 디렉토리입니다.

| 파일 | 역할 |
|------|------|
| `PKG-INFO` | 패키지 이름, 버전, 설명 등 (pyproject.toml에서 추출) |
| `requires.txt` | 의존성 목록 (`torch>=2.8.0` 등) |
| `SOURCES.txt` | 패키지에 포함된 모든 소스 파일 목록 |

개발 모드에서 pip/setuptools에게 "이 디렉토리가 Python 패키지다"라고 알려주는 역할입니다. `.gitignore`에 추가해도 무방합니다.

### Q: `uv.lock`은 필수적인가?

uv를 사용한다면 필수적입니다. pip의 `requirements.txt`에 해당합니다.

`pyproject.toml`에 `torch>=2.8.0`이라고 적으면 "2.8 이상 아무거나"를 의미하지만, `uv.lock`은 실제로 설치된 정확한 버전(예: `torch==2.10.0+cu128`)을 기록합니다. git에 커밋해두면 누가 어디서 `uv sync`해도 동일한 환경이 재현됩니다.

### Q: `uv` 없이 `python`으로 직접 실행할 수 있나?

가능합니다. 가상환경을 활성화하면 됩니다:

```bash
source .venv/bin/activate               # 가상환경 활성화
python setup.py build_ext --inplace     # 빌드
pytest tests/kernels/test_rope.py -v    # 테스트
```

`uv run`은 이 `source activate`를 매번 자동으로 해주는 편의 기능일 뿐입니다.

### Q: 다른 터미널에서 `uv sync` 해도 해당 셸에 가상환경이 활성화가 안 되는 이유는?

`uv sync`는 `.venv`를 생성/업데이트할 뿐, 현재 셸의 환경변수를 바꾸지 않습니다. 셸에서 가상환경을 "활성화"한다는 건 `PATH` 환경변수 앞에 `.venv/bin/`을 추가하는 것인데, 이건 각 셸 프로세스마다 독립적입니다.

### Q: `uv pip install -e .`와 `python setup.py build_ext --inplace`의 차이는?

| | `setup.py build_ext --inplace` | `uv pip install -e .` |
|---|---|---|
| **CUDA 컴파일** | O | O |
| **패키지 등록** | X | O (editable 모드) |
| **Python 수정 즉시 반영** | — | O |

- **`python setup.py build_ext --inplace`**: CUDA extension(.so)만 컴파일합니다. 패키지 등록은 하지 않습니다.
- **`uv pip install -e .`**: CUDA 컴파일 + Python 패키지를 가상환경에 editable 모드로 등록합니다.

editable 모드(`-e`)는 소스 코드를 `.venv/site-packages/`로 복사하는 대신, 현재 디렉토리를 가리키는 링크를 만듭니다. 그래서 Python 파일을 수정하면 재설치 없이 즉시 반영됩니다. 단, CUDA 코드를 수정하면 재컴파일이 필요합니다.

### Q: `build/` 폴더는 왜 생기는 거지?

`setup.py build_ext --inplace`를 실행하면 setuptools가 중간 컴파일 결과물을 저장하는 디렉토리입니다.

```
build/
└── temp.linux-x86_64-cpython-312/
    └── csrc/
        ├── pos_encoding_kernels.o    ← nvcc가 .cu를 컴파일한 오브젝트 파일
        └── torch_bindings.o          ← g++가 .cpp를 컴파일한 오브젝트 파일
```

캐시 역할을 해서, 소스를 수정하지 않으면 다음 빌드 시 재컴파일을 건너뜁니다. 삭제해도 다시 생성되므로 `.gitignore`에 넣으면 됩니다.

### Q: `nvcc`로 직접 컴파일하는 것과 `setup.py` 빌드의 차이는?

`test_rope_kernel.cu`를 `nvcc`로 직접 컴파일할 때와, `setup.py`로 Python extension을 빌드할 때의 차이:

| | nvcc 직접 컴파일 | setup.py 빌드 |
|---|---|---|
| **결과물** | 독립 실행파일 (`test_rope_kernel`) | 공유 라이브러리 (`_C.so`) |
| **실행 방법** | `./test_rope_kernel` | `import lightvllm._C` |
| **진입점** | `main()` | `PYBIND11_MODULE` |
| **데이터 전달** | `cudaMalloc`/`cudaMemcpy` 직접 관리 | PyTorch 텐서 (`torch::Tensor`) |
| **링크 대상** | CUDA runtime만 | CUDA + PyTorch + Python |
| **용도** | 커널 로직 자체를 격리 검증 | 실제 서비스에서 사용할 인터페이스 검증 |

---

## PyTorch 바인딩

### Q: `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`이 뭐야?

pybind11은 C++ 코드를 Python 모듈로 만들어주는 라이브러리입니다.

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_embedding", &rotary_embedding, "Rotary Position Embedding");
}
```

- **`TORCH_EXTENSION_NAME`**: 매크로. 빌드 시 `setup.py`의 `name="lightvllm._C"`에서 `_C` 부분이 자동 대입됩니다. 컴파일 명령에 `-DTORCH_EXTENSION_NAME=_C` 플래그로 전달됩니다.
- **`m`**: 모듈 객체. Python 모듈에 해당하며, `m.def()`로 함수를 등록하면 Python에서 `_C.rotary_embedding()`처럼 호출할 수 있습니다.

### Q: `lightvllm._C`라는 모듈 이름은 어떻게 만들어진 거지?

`setup.py`에서 정의됩니다:

```python
CUDAExtension(name="lightvllm._C", ...)
```

이 이름이 두 곳에 영향을 줍니다:
1. **파일 배치**: `lightvllm/_C.cpython-312-x86_64-linux-gnu.so` 생성
2. **컴파일 매크로**: `-DTORCH_EXTENSION_NAME=_C`가 전달되어 `PYBIND11_MODULE`에서 사용

`_C`는 PyTorch 관례입니다. PyTorch 자체도 내부 C++ 바인딩을 `torch._C`로 제공합니다.

### Q: `torch_bindings.cpp`에서 IntelliSense 오류가 나는 이유는?

```
cannot open source file "Python.h" (dependency of "torch/extension.h")
```

빌드 오류가 아니라 **VSCode의 IntelliSense(코드 편집기) 오류**입니다. 실제 컴파일은 정상입니다.

VSCode C++ 확장이 Python/PyTorch/CUDA include 경로를 모르기 때문입니다. `.vscode/c_cpp_properties.json`에 아래 경로를 추가하면 해결됩니다:
- `/usr/include/python3.12` (Python.h)
- `.venv/lib/.../torch/include` (torch 헤더)
- `/usr/local/cuda-12.8/include` (CUDA)

`setup.py`는 이 경로를 자동으로 찾아서 컴파일러에 넘기므로 빌드에는 영향 없습니다.

---

## 테스트

### Q: C++ 테스트는 `main()` 함수가 필요한데, Python 테스트는 왜 필요 없지?

**pytest가 main 역할을 대신하기 때문입니다.**

C++ 프로그램은 OS가 실행할 진입점(`main`)이 반드시 필요합니다:
```
OS → main() → test_basic_correctness() → test_norm_preservation() → ...
```

pytest는 테스트 러너 프레임워크로, 파일을 스캔하여 `test_`로 시작하는 함수를 자동으로 찾아 실행합니다:
```
OS → pytest(main) → test_rope.py 스캔 → Test* 클래스 → test_* 메서드 실행
```

pytest 자체가 main이고, 테스트 파일은 "실행할 함수 목록"만 제공하면 됩니다.

### Q: bf16 테스트에서 CPU 참조 구현과 GPU 커널 결과가 달랐던 이유는?

CPU에서 bf16 연산을 수행하면 내부적으로 **float32로 변환 후 계산**하고 결과를 bf16으로 다시 변환합니다. GPU는 bf16 하드웨어 유닛으로 직접 연산합니다. 이 때문에 중간 연산 결과의 반올림이 달라져서 최대 4.4의 오차가 발생했습니다.

해결: 참조 구현도 GPU에서 동일한 dtype으로 실행하면 bit-exact 일치합니다 (`atol=0, rtol=0`).

### Q: CUDA extension을 빌드하는 방법은?

Python에서 CUDA 커널을 사용하려면 먼저 빌드가 필요합니다:

```bash
uv pip install -e .
```

이 명령은 `csrc/` 디렉토리의 `.cu`와 `.cpp` 파일을 컴파일하고, 패키지를 editable 모드로 가상환경에 등록합니다. `.so` 파일은 `.venv/lib/.../site-packages/lightvllm/` 안에 생성됩니다. Python 코드는 재빌드 없이 즉시 반영되지만, CUDA 소스 코드를 수정하면 다시 빌드해야 합니다.

참고로 CUDA 컴파일만 빠르게 하고 싶다면 `--inplace` 옵션도 있습니다:
```bash
uv run python setup.py build_ext --inplace
```
이 경우 `.so`가 프로젝트 폴더(`lightvllm/`) 안에 직접 생성되며, 패키지 등록은 하지 않습니다.

### Q: pytest에서 특정 테스트만 실행하는 방법은?

**전체 테스트 실행**:
```bash
uv run pytest tests/kernels/test_rope.py -v
```

**특정 테스트 클래스/메서드 지정** (`::` 구분자 사용):
```bash
# 특정 메서드
uv run pytest tests/kernels/test_rope.py::TestRoPE::test_basic_correctness -v

# 특정 클래스 전체
uv run pytest tests/kernels/test_rope.py::TestRoPE -v
```

**키워드 매칭** (`-k` 옵션):
```bash
# "norm"이 이름에 포함된 테스트만
uv run pytest tests/kernels/test_rope.py -k "norm" -v

# 여러 키워드 조합
uv run pytest tests/kernels/test_rope.py -k "norm or identity" -v
```

**전체 테스트 디렉토리 실행** (새 커널 테스트가 추가된 경우):
```bash
uv run pytest tests/kernels/ -v
```
