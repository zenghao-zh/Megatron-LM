# coding=utf-8
"""SmolLM model configuration — thin wrapper around LlamaConfig."""

from transformers import LlamaConfig


class SmolLMConfig(LlamaConfig):
    """
    Configuration class for SmolLM models.

    SmolLM uses the standard LLaMA architecture (RMSNorm, SwiGLU, RoPE, GQA).
    This config is identical to LlamaConfig except for the model_type identifier
    and optional extensions for INT8 forward quantization and activation sparsity.

    Args:
        int8_forward (`bool`, *optional*, defaults to `False`):
            Whether to apply INT8 group-wise quantization to MLP linear layers
            during forward pass, matching INT8 mixed-precision training behavior.
        int8_group_size (`int`, *optional*, defaults to `64`):
            Group size for INT8 quantization along the K dimension.
        act_sparse_training (`bool`, *optional*, defaults to `False`):
            Whether the model was trained with activation sparsity (BalancedTopk).
            When True, each MLP block includes a Predictor and BalancedTopkModule.
        predictor_hidden_size (`int`, *optional*, defaults to `64`):
            Hidden dimension of the 2-layer Predictor MLP.
        predictor_bank_size (`int`, *optional*, defaults to `64`):
            Bank size for balanced top-k selection.
        predictor_topk (`int`, *optional*, defaults to `16`):
            Number of top-k activations to keep per bank.
        swiglu_without_silu (`bool`, *optional*, defaults to `False`):
            If True, skip the SiLU activation in the gated MLP (use identity),
            matching Megatron ``--act-sparse-swiglu-without-silu``.
    """

    model_type = "smollm"

    def __init__(
        self,
        int8_forward=False,
        int8_group_size=64,
        act_sparse_training=False,
        predictor_hidden_size=64,
        predictor_bank_size=64,
        predictor_topk=16,
        swiglu_without_silu=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.int8_forward = int8_forward
        self.int8_group_size = int8_group_size
        self.act_sparse_training = act_sparse_training
        self.predictor_hidden_size = predictor_hidden_size
        self.predictor_bank_size = predictor_bank_size
        self.predictor_topk = predictor_topk
        self.swiglu_without_silu = swiglu_without_silu


__all__ = ["SmolLMConfig"]
