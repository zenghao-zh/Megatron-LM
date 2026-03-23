# coding=utf-8
"""SmolLM model — thin wrapper around LlamaForCausalLM.

SmolLM is architecturally identical to LLaMA, so we simply inherit from
the transformers LLaMA implementation and register a new model_type.

When ``act_sparse_training`` is enabled in the config, the standard LlamaMLP
in each decoder layer is replaced with a BalancedTopkMLP that includes a
Predictor and top-k sparsity mask.
"""

import torch
from torch import nn

from transformers import LlamaForCausalLM, LlamaModel
from transformers.activations import ACT2FN

from .configuration_smollm import SmolLMConfig


# ---------------------------------------------------------------------------
# Activation-sparsity helper modules
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """Two-layer linear predictor: hidden_size -> mid -> intermediate_size."""

    def __init__(self, in_features: int, mid_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.fc_1 = nn.Linear(in_features, mid_features, bias=bias)
        self.fc_2 = nn.Linear(mid_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_2(self.fc_1(x))


class BalancedTopkFunction(torch.autograd.Function):
    """Per-bank top-k masking with bias for load balancing."""

    @staticmethod
    def forward(ctx, input, k, bias):
        _, topk_indices = (input.abs() + bias).topk(k, dim=-1)
        mask = torch.zeros_like(input, dtype=input.dtype)
        mask.scatter_(-1, topk_indices, 1)
        output = input * mask
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None, None


class BalancedTopkModule(nn.Module):
    """Balanced top-k selection with per-bank bias buffers."""

    def __init__(self, hidden_size: int, topk: int, bank_size: int):
        super().__init__()
        self.register_buffer("balanced_bias", torch.zeros(hidden_size, dtype=torch.float32))
        self.register_buffer("num_assigned_tokens", torch.zeros(hidden_size))
        self.topk = topk
        self.bank_size = bank_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor):
        mask_ = BalancedTopkFunction.apply(
            x.view(-1, self.hidden_size // self.bank_size, self.bank_size),
            self.topk,
            self.balanced_bias.view(-1, self.bank_size),
        )
        mask = mask_.view_as(x)
        if self.training:
            with torch.no_grad():
                self.num_assigned_tokens += (mask != 0).sum(dim=tuple(range(x.ndim - 1)))
        return mask, self.num_assigned_tokens


class BalancedTopkMLP(nn.Module):
    """MLP with activation sparsity via Predictor + BalancedTopk masking."""

    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=getattr(config, "mlp_bias", False))
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=getattr(config, "mlp_bias", False))
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=getattr(config, "mlp_bias", False))

        self.swiglu_without_silu = getattr(config, "swiglu_without_silu", False)
        if not self.swiglu_without_silu:
            self.act_fn = ACT2FN[config.hidden_act]

        self.predictor = Predictor(
            self.hidden_size,
            config.predictor_hidden_size,
            self.intermediate_size,
            bias=getattr(config, "mlp_bias", False),
        )
        self.topk_model = BalancedTopkModule(
            self.intermediate_size,
            config.predictor_topk,
            config.predictor_bank_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_x = torch.sigmoid(self.predictor(x))
        mask, *_ = self.topk_model(pred_x)

        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)

        if self.swiglu_without_silu:
            intermediate = mask * x_gate * x_up
        else:
            intermediate = mask * self.act_fn(x_gate) * x_up

        return self.down_proj(intermediate)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class SmolLMModel(LlamaModel):
    """SmolLM base model (identical to LlamaModel)."""
    config_class = SmolLMConfig


class SmolLMForCausalLM(LlamaForCausalLM):
    """SmolLM for causal language modeling.

    When ``config.act_sparse_training`` is True, replaces each layer's
    standard LlamaMLP with a BalancedTopkMLP.
    """
    config_class = SmolLMConfig

    def __init__(self, config: SmolLMConfig):
        super().__init__(config)
        if getattr(config, "act_sparse_training", False):
            for layer in self.model.layers:
                layer.mlp = BalancedTopkMLP(config)


__all__ = ["SmolLMConfig", "SmolLMModel", "SmolLMForCausalLM"]
