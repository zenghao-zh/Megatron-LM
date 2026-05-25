# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse

import torch

from megatron.core.transformer.mlp import (
    _linear_warmup_progress,
    _rms_normalize_last_dim,
    _scheduled_topk,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import add_megatron_arguments


def test_sparse_gate_rms_normalize_sets_unit_rms():
    torch.manual_seed(123)
    x = torch.randn(2, 3, 8) * 3.0 + 0.5

    y = _rms_normalize_last_dim(x, eps=1e-8)

    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)


def test_sparse_gate_rms_normalize_preserves_zero_mask_positions():
    sparse_gate = torch.tensor([[[0.5, 0.0, 0.25, 0.0, 0.75, 0.0, 0.125, 0.0]]])

    normalized = _rms_normalize_last_dim(sparse_gate, eps=1e-8)

    assert torch.equal(normalized == 0, sparse_gate == 0)
    assert torch.allclose(
        normalized.pow(2).mean(dim=-1).sqrt(),
        torch.ones(1, 1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_transformer_config_accepts_energy_preserving_flag():
    config = TransformerConfig(
        num_layers=1,
        num_attention_heads=1,
        act_sparse_energy_preserving_swiglu=True,
        act_sparse_swiglu_gate_warmup_steps=2000,
        act_sparse_topk_warmup_steps=2000,
    )

    assert config.act_sparse_energy_preserving_swiglu is True
    assert config.act_sparse_swiglu_gate_warmup_steps == 2000
    assert config.act_sparse_topk_warmup_steps == 2000


def test_parser_accepts_energy_preserving_flag():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser = add_megatron_arguments(parser)

    args = parser.parse_args(
        [
            "--act-sparse-energy-preserving-swiglu",
            "--act-sparse-swiglu-gate-warmup-steps",
            "2000",
            "--act-sparse-topk-warmup-steps",
            "2000",
        ]
    )

    assert args.act_sparse_energy_preserving_swiglu is True
    assert args.act_sparse_swiglu_gate_warmup_steps == 2000
    assert args.act_sparse_topk_warmup_steps == 2000


def test_linear_warmup_progress():
    assert _linear_warmup_progress(current_iteration=0, warmup_steps=2000) == 0.0
    assert _linear_warmup_progress(current_iteration=1000, warmup_steps=2000) == 0.5
    assert _linear_warmup_progress(current_iteration=3000, warmup_steps=2000) == 1.0
    assert _linear_warmup_progress(current_iteration=0, warmup_steps=0) == 1.0


def test_scheduled_topk_ramps_from_bank_size_to_target():
    assert _scheduled_topk(target_topk=16, bank_size=64, progress=0.0) == 64
    assert _scheduled_topk(target_topk=16, bank_size=64, progress=0.5) == 40
    assert _scheduled_topk(target_topk=16, bank_size=64, progress=1.0) == 16
