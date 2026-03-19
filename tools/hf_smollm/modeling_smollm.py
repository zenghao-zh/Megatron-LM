# coding=utf-8
"""SmolLM model — thin wrapper around LlamaForCausalLM.

SmolLM is architecturally identical to LLaMA, so we simply inherit from
the transformers LLaMA implementation and register a new model_type.
"""

from transformers import LlamaForCausalLM, LlamaModel

from .configuration_smollm import SmolLMConfig


class SmolLMModel(LlamaModel):
    """SmolLM base model (identical to LlamaModel)."""
    config_class = SmolLMConfig


class SmolLMForCausalLM(LlamaForCausalLM):
    """SmolLM for causal language modeling (identical to LlamaForCausalLM)."""
    config_class = SmolLMConfig


__all__ = ["SmolLMConfig", "SmolLMModel", "SmolLMForCausalLM"]
