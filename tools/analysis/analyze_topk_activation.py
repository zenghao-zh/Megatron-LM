#!/usr/bin/env python3
"""Analyze per-channel activation frequency of BalancedTopkMLP masks.

Loads a sparse-trained SmolLM checkpoint, runs one batch of training data,
and records how often each intermediate channel is activated (selected by
the top-k mask) across all tokens and layers.

Outputs:
  - Per-layer statistics to stdout
  - activation_counts.pt   -- raw per-channel counts
  - activation_heatmap.png -- heatmap (layers x channels)
"""

import argparse
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MEGATRON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TOOLS_DIR = os.path.join(MEGATRON_ROOT, "tools")
sys.path.insert(0, MEGATRON_ROOT)
sys.path.insert(0, TOOLS_DIR)

from hf_smollm import SmolLMConfig, SmolLMForCausalLM  # noqa: E402
from megatron.core.datasets.indexed_dataset import IndexedDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pretrained",
        type=str,
        default="/root/data/megatron-models/checkpoints/smollm-130m-btopk-005/hf_smollm_iter_0150000",
    )
    p.add_argument(
        "--data-prefix",
        type=str,
        default="/root/data/smollm_corpus/merged_smollm_corpus",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output_btopk_005"),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_batch_from_indexed_dataset(data_prefix, batch_size, seq_len):
    """Read the first documents from a Megatron IndexedDataset and pack them
    into ``batch_size`` sequences of ``seq_len`` tokens."""
    dataset = IndexedDataset(data_prefix)
    tokens = []
    total_needed = batch_size * seq_len
    collected = 0
    doc_idx = 0
    while collected < total_needed and doc_idx < len(dataset):
        doc = dataset[doc_idx]
        tokens.append(doc)
        collected += len(doc)
        doc_idx += 1

    flat = np.concatenate(tokens)[:total_needed]
    return torch.from_numpy(flat.astype(np.int64)).reshape(batch_size, seq_len)


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

def register_mask_hooks(model):
    """Register forward hooks on every BalancedTopkModule to capture masks.

    Returns (activation_counts dict, hook_handles list).
    """
    num_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size
    device = next(model.parameters()).device

    activation_counts = {
        i: torch.zeros(intermediate_size, device=device) for i in range(num_layers)
    }

    def make_hook(layer_idx):
        def hook(_module, _input, output):
            mask, _ = output
            activation_counts[layer_idx] += (
                (mask != 0).float().sum(dim=tuple(range(mask.ndim - 1)))
            )
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        h = layer.mlp.topk_model.register_forward_hook(make_hook(i))
        handles.append(h)

    return activation_counts, handles


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_heatmap(activation_counts, total_tokens, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_layers = len(activation_counts)
    intermediate_size = activation_counts[0].shape[0]

    freq = torch.stack(
        [activation_counts[i] for i in range(num_layers)]
    ).cpu().float() / total_tokens

    fig, axes = plt.subplots(
        2, 1, figsize=(20, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- Heatmap ---
    ax = axes[0]
    im = ax.imshow(freq.numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Layer index")
    ax.set_title(f"Per-channel activation frequency (total_tokens={total_tokens})")
    fig.colorbar(im, ax=ax, label="activation freq (count / total_tokens)")

    # --- Per-layer mean / std bar chart ---
    ax2 = axes[1]
    means = freq.mean(dim=1).numpy()
    stds = freq.std(dim=1).numpy()
    layer_ids = np.arange(num_layers)
    ax2.bar(layer_ids, means, yerr=stds, capsize=2, alpha=0.8)
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Mean activation freq")
    ax2.set_title("Per-layer mean activation frequency (+/- std)")
    ax2.set_xticks(layer_ids)

    plt.tight_layout()
    path = os.path.join(output_dir, "activation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {path}")


def print_layer_stats(activation_counts, total_tokens):
    num_layers = len(activation_counts)
    print(f"\n{'Layer':>6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Entropy':>10}")
    print("-" * 62)
    for i in range(num_layers):
        freq = activation_counts[i].cpu().float() / total_tokens
        mean = freq.mean().item()
        std = freq.std().item()
        mn = freq.min().item()
        mx = freq.max().item()
        p = freq.clamp(min=1e-10)
        entropy = -(p * p.log()).sum().item()
        print(f"{i:>6d} {mean:>10.4f} {std:>10.4f} {mn:>10.4f} {mx:>10.4f} {entropy:>10.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model
    print(f"Loading model from {args.pretrained} ...")
    model = SmolLMForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16
    )
    model.to(args.device).eval()
    num_layers = len(model.model.layers)
    print(f"  {num_layers} layers, intermediate_size={model.config.intermediate_size}")

    # 2. Load data
    print(f"Loading batch ({args.batch_size} x {args.seq_len}) from {args.data_prefix} ...")
    input_ids = load_batch_from_indexed_dataset(
        args.data_prefix, args.batch_size, args.seq_len
    ).to(args.device)
    total_tokens = args.batch_size * args.seq_len
    print(f"  input_ids shape: {input_ids.shape}, total_tokens={total_tokens}")

    # 3. Register hooks
    activation_counts, handles = register_mask_hooks(model)

    # 4. Forward pass
    print("Running forward pass ...")
    with torch.no_grad():
        model(input_ids=input_ids)
    print("Done.")

    # Remove hooks
    for h in handles:
        h.remove()

    # 5. Output results
    print_layer_stats(activation_counts, total_tokens)

    counts_path = os.path.join(args.output_dir, "activation_counts.pt")
    torch.save({i: activation_counts[i].cpu() for i in range(num_layers)}, counts_path)
    print(f"\nSaved raw counts to {counts_path}")

    save_heatmap(activation_counts, total_tokens, args.output_dir)


if __name__ == "__main__":
    main()
