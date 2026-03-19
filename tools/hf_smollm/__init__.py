from .configuration_smollm import SmolLMConfig
from .modeling_smollm import SmolLMForCausalLM, SmolLMModel

# Register with transformers auto-classes
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("smollm", SmolLMConfig)
AutoModelForCausalLM.register(SmolLMConfig, SmolLMForCausalLM)

__all__ = ["SmolLMConfig", "SmolLMForCausalLM", "SmolLMModel"]
