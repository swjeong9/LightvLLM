"""
모델 인터페이스 (향후 구현 예정).

현재 Phase 1에서는 모든 모델이 단순히 nn.Module을 상속합니다.
향후 LoRA, Tensor Parallelism, Pipeline Parallelism 등을 지원할 때
이 파일에 Protocol 인터페이스를 추가할 예정입니다.

vLLM의 접근 방식:
    vLLM은 ABC(추상 기반 클래스)가 아닌 @runtime_checkable Protocol을 사용합니다.
    Protocol은 구조적 duck typing으로, 특정 메서드/속성이 존재하는지만 체크합니다.
    모델 클래스가 명시적으로 상속하지 않아도 구조만 맞으면 호환됩니다.

    예시:
        class SupportsLoRA(Protocol):
            supports_lora: ClassVar[Literal[True]]
            packed_modules_mapping: ClassVar[dict[str, list[str]]]

        class SupportsPP(Protocol):
            supports_pp: ClassVar[Literal[True]]
            def make_empty_intermediate_tensors(...) -> ...: ...

    모델 선언:
        class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
            supports_lora = True
            supports_pp = True
            ...

    런타임 체크:
        if isinstance(model, SupportsLoRA):
            # LoRA 적용 로직

향후 추가 예정 Protocol:
    - SupportsLoRA: LoRA adapter 적용 가능 여부
    - SupportsTP: Tensor Parallelism (weight 분산)
    - SupportsPP: Pipeline Parallelism (레이어 분산)

vLLM 참조:
    - vLLM/vllm/model_executor/models/interfaces.py
    - vLLM/vllm/model_executor/models/interfaces_base.py
"""
