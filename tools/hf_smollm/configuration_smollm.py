# coding=utf-8
"""SmolLM model configuration — thin wrapper around LlamaConfig."""

from transformers import LlamaConfig


class SmolLMConfig(LlamaConfig):
    """
    Configuration class for SmolLM models.

    SmolLM uses the standard LLaMA architecture (RMSNorm, SwiGLU, RoPE, GQA).
    This config is identical to LlamaConfig except for the model_type identifier
    and optional INT8 forward quantization control.

    Args:
        int8_forward (`bool`, *optional*, defaults to `False`):
            Whether to apply INT8 group-wise quantization to MLP linear layers
            during forward pass, matching INT8 mixed-precision training behavior.
        int8_group_size (`int`, *optional*, defaults to `64`):
            Group size for INT8 quantization along the K dimension.
    """

    model_type = "smollm"

    def __init__(self, int8_forward=False, int8_group_size=64, **kwargs):
        super().__init__(**kwargs)
        self.int8_forward = int8_forward
        self.int8_group_size = int8_group_size


__all__ = ["SmolLMConfig"]
