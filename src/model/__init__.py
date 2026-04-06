from .configuration import DecoderConfig

__all__ = ["DecoderConfig"]

try:
    from .modeling import (
        CausalLMOutput,
        DecoderBackbone,
        DecoderForCausalLM,
        DecoderModelOutput,
    )

    __all__ += [
        "DecoderBackbone",
        "DecoderForCausalLM",
        "DecoderModelOutput",
        "CausalLMOutput",
    ]
except ModuleNotFoundError:
    # 在未安装 torch 时，仍允许先进行配置层工作流。
    pass
