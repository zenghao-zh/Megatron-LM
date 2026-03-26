#!/usr/bin/env python3
"""Analyze intra-group channel co-activation patterns in BalancedTopkMLP masks.

Since BalancedTopk selects top-k independently within each bank (group of
``bank_size`` channels), this script computes co-activation statistics
*within* each group to reveal which channels inside a group tend to fire
together.

Outputs:
  - Per-layer / per-group statistics to stdout
  - coactivation_data.pt                -- raw per-group co-activation matrices
  - coactivation_group_heatmaps.png     -- Jaccard heatmap grid (groups x groups)
  - coactivation_layer_group_summary.png -- (layers x groups) mean/max Jaccard
  - coactivation_top_pairs.txt          -- top co-activated pairs per layer per group
  - coactivation_group_clusters.png     -- dendrograms for selected layer/groups
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
        default="/root/data/megatron-models/checkpoints/smollm-130m-btopk/hf_smollm_iter_0150000_balanced",
    )
    p.add_argument(
        "--data-prefix",
        type=str,
        default="/root/data/smollm_corpus/merged_smollm_corpus",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--top-k-pairs", type=int, default=20,
                    help="Number of top co-activated pairs to report per group")
    p.add_argument("--cluster-layers", type=str, default=None,
                    help="Comma-separated layer indices for detailed plots "
                         "(default: first, mid, last)")
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output_coactivation_group"),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_batch_from_indexed_dataset(data_prefix, batch_size, seq_len):
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
# Hook helpers -- per-group AND global co-activation
# ---------------------------------------------------------------------------

def register_coactivation_hooks(model):
    """Accumulate both per-group and global co-activation matrices.

    Returns (coact, counts, global_coact, global_counts, handles,
             num_groups, bank_size).
    coact[layer]        : Tensor (G, B, B)   -- intra-group
    counts[layer]       : Tensor (G, B)
    global_coact[layer] : Tensor (C, C)      -- full cross-channel
    global_counts[layer]: Tensor (C,)
    """
    num_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size
    bank_size = model.config.predictor_bank_size
    num_groups = intermediate_size // bank_size
    device = next(model.parameters()).device

    coact = {
        i: torch.zeros(num_groups, bank_size, bank_size, device=device)
        for i in range(num_layers)
    }
    counts = {
        i: torch.zeros(num_groups, bank_size, device=device)
        for i in range(num_layers)
    }
    global_coact = {
        i: torch.zeros(intermediate_size, intermediate_size, device=device)
        for i in range(num_layers)
    }
    global_counts = {
        i: torch.zeros(intermediate_size, device=device)
        for i in range(num_layers)
    }

    def make_hook(layer_idx):
        def hook(_module, _input, output):
            mask, _ = output
            # per-group
            binary_g = (mask != 0).float().reshape(-1, num_groups, bank_size)
            counts[layer_idx] += binary_g.sum(dim=0)
            coact[layer_idx] += torch.einsum("ngb,ngc->gbc", binary_g, binary_g)
            # global
            binary = (mask != 0).float().reshape(-1, intermediate_size)  # (N, C)
            global_counts[layer_idx] += binary.sum(dim=0)
            global_coact[layer_idx] += binary.T @ binary                 # (C, C)
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        h = layer.mlp.topk_model.register_forward_hook(make_hook(i))
        handles.append(h)

    return (coact, counts, global_coact, global_counts,
            handles, num_groups, bank_size)


# ---------------------------------------------------------------------------
# Metrics (operate on single-group tensors)
# ---------------------------------------------------------------------------

def compute_jaccard(coact_g, counts_g):
    """Jaccard for one group.  coact_g: (B,B), counts_g: (B,)."""
    ci = counts_g.unsqueeze(1)
    cj = counts_g.unsqueeze(0)
    union = ci + cj - coact_g
    jac = coact_g / union.clamp(min=1)
    jac.fill_diagonal_(0)
    return jac


def compute_conditional_prob(coact_g, counts_g):
    """P(j|i) = coact(i,j) / count(i).  Asymmetric."""
    return coact_g / counts_g.unsqueeze(1).clamp(min=1)


def top_pairs(matrix, k):
    B = matrix.shape[0]
    triu_idx = torch.triu_indices(B, B, offset=1)
    vals = matrix[triu_idx[0], triu_idx[1]]
    topk_vals, topk_pos = vals.topk(min(k, vals.numel()))
    pairs = []
    for v, p in zip(topk_vals.tolist(), topk_pos.tolist()):
        pairs.append((triu_idx[0][p].item(), triu_idx[1][p].item(), v))
    return pairs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def cluster_order(jac_np):
    """Return leaf ordering from hierarchical clustering on a Jaccard matrix."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    dist = np.clip(1.0 - jac_np, 0, None)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average", optimal_ordering=True)
    return leaves_list(Z)


def save_group_heatmaps(coact, counts, layer_indices, num_groups, bank_size,
                        output_dir):
    """For each selected layer, draw a grid of Jaccard heatmaps with rows/cols
    reordered by hierarchical clustering so co-activated channels are adjacent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 6
    nrows = (num_groups + ncols - 1) // ncols

    for li in layer_indices:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows))
        axes_flat = axes.flatten()

        for g in range(num_groups):
            ax = axes_flat[g]
            jac = compute_jaccard(
                coact[li][g].cpu().float(), counts[li][g].cpu().float()
            ).numpy()

            order = cluster_order(jac)
            jac_sorted = jac[np.ix_(order, order)]

            im = ax.imshow(jac_sorted, aspect="equal", cmap="hot",
                           interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(f"G{g}", fontsize=9)
            n_ticks = 5
            tick_pos = np.linspace(0, bank_size - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([str(order[t]) for t in tick_pos], fontsize=6)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels([str(order[t]) for t in tick_pos], fontsize=6)

        for g in range(num_groups, len(axes_flat)):
            axes_flat[g].axis("off")

        fig.suptitle(f"Layer {li} — Intra-group Jaccard (clustered order, "
                     f"{bank_size} ch/group)", fontsize=13)
        fig.colorbar(im, ax=axes_flat[:num_groups].tolist(), shrink=0.6,
                     label="Jaccard")
        plt.tight_layout(rect=[0, 0, 0.93, 0.95])
        path = os.path.join(output_dir, f"coactivation_group_heatmaps_layer{li}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved group heatmaps to {path}")


def save_layer_group_summary(coact, counts, num_layers, num_groups, output_dir):
    """2-D heatmap: (layers x groups) for mean-Jaccard and max-Jaccard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_jac = np.zeros((num_layers, num_groups))
    max_jac = np.zeros((num_layers, num_groups))
    strong_cnt = np.zeros((num_layers, num_groups))
    B = counts[0].shape[1]
    triu = torch.triu_indices(B, B, offset=1)

    for li in range(num_layers):
        for g in range(num_groups):
            jac = compute_jaccard(
                coact[li][g].cpu().float(), counts[li][g].cpu().float()
            )
            vals = jac[triu[0], triu[1]]
            mean_jac[li, g] = vals.mean().item()
            max_jac[li, g] = vals.max().item()
            strong_cnt[li, g] = (vals > 0.5).sum().item()

    fig, axes = plt.subplots(3, 1, figsize=(max(10, num_groups * 0.6), 12))

    im0 = axes[0].imshow(mean_jac, aspect="auto", cmap="YlOrRd",
                          interpolation="nearest")
    axes[0].set_title("Mean intra-group Jaccard per (layer, group)")
    axes[0].set_ylabel("Layer")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(max_jac, aspect="auto", cmap="YlOrRd",
                          interpolation="nearest")
    axes[1].set_title("Max intra-group Jaccard per (layer, group)")
    axes[1].set_ylabel("Layer")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(strong_cnt, aspect="auto", cmap="YlGnBu",
                          interpolation="nearest")
    axes[2].set_title("# channel pairs with Jaccard > 0.5 per (layer, group)")
    axes[2].set_ylabel("Layer")
    axes[2].set_xlabel("Group index")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, "coactivation_layer_group_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved layer-group summary to {path}")


def save_group_clusters(coact, counts, layer_indices, num_groups, bank_size,
                        output_dir):
    """Dendrograms for each group in selected layers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    ncols = 6
    nrows = (num_groups + ncols - 1) // ncols

    for li in layer_indices:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
        axes_flat = axes.flatten()

        for g in range(num_groups):
            ax = axes_flat[g]
            jac = compute_jaccard(
                coact[li][g].cpu().float(), counts[li][g].cpu().float()
            )
            dist = (1.0 - jac).clamp(min=0).numpy()
            np.fill_diagonal(dist, 0)
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method="average")
            dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7)
            ax.set_title(f"G{g}", fontsize=9)
            ax.tick_params(labelsize=6)

        for g in range(num_groups, len(axes_flat)):
            axes_flat[g].axis("off")

        fig.suptitle(f"Layer {li} — Intra-group channel clustering", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(output_dir, f"coactivation_group_clusters_layer{li}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved group clusters to {path}")


def save_top_pairs_report(coact, counts, total_tokens, num_layers, num_groups,
                          bank_size, top_k, output_dir):
    path = os.path.join(output_dir, "coactivation_top_pairs.txt")
    with open(path, "w") as f:
        for li in range(num_layers):
            f.write(f"\n{'=' * 78}\n")
            f.write(f"Layer {li}\n")
            f.write(f"{'=' * 78}\n")

            for g in range(num_groups):
                jac = compute_jaccard(
                    coact[li][g].cpu().float(), counts[li][g].cpu().float()
                )
                pairs = top_pairs(jac, top_k)
                if not pairs or pairs[0][2] < 0.3:
                    continue

                global_offset = g * bank_size
                f.write(f"\n  Group {g}  (global channels {global_offset}–"
                        f"{global_offset + bank_size - 1}):\n")
                f.write(f"  {'Rank':>4}  {'Local_i':>7}  {'Local_j':>7}  "
                        f"{'Global_i':>8}  {'Global_j':>8}  {'Jaccard':>9}  "
                        f"{'CoAct':>8}  {'Cnt_i':>8}  {'Cnt_j':>8}\n")
                coact_g = coact[li][g].cpu()
                counts_g = counts[li][g].cpu()
                for rank, (ci, cj, jval) in enumerate(pairs, 1):
                    f.write(f"  {rank:>4}  {ci:>7}  {cj:>7}  "
                            f"{ci + global_offset:>8}  {cj + global_offset:>8}  "
                            f"{jval:>9.4f}  "
                            f"{coact_g[ci, cj].item():>8.0f}  "
                            f"{counts_g[ci].item():>8.0f}  "
                            f"{counts_g[cj].item():>8.0f}\n")

    print(f"Saved top pairs report to {path}")


def compute_global_layer_stats(global_coact, global_counts, num_layers, C):
    """Compute per-layer summary from the full (C, C) co-activation matrix.

    Returns dict with arrays of shape (num_layers,).
    """
    triu = torch.triu_indices(C, C, offset=1)
    n_triu = triu.shape[1]

    mean_jac = np.zeros(num_layers)
    median_jac = np.zeros(num_layers)
    max_jac = np.zeros(num_layers)
    strong_pairs = np.zeros(num_layers)
    locked_pairs = np.zeros(num_layers)

    for li in range(num_layers):
        jac = compute_jaccard(
            global_coact[li].cpu().float(), global_counts[li].cpu().float()
        )
        vals = jac[triu[0], triu[1]]
        mean_jac[li] = vals.mean().item()
        median_jac[li] = vals.median().item()
        max_jac[li] = vals.max().item()
        strong_pairs[li] = (vals > 0.5).sum().item()
        locked_pairs[li] = (vals > 0.9).sum().item()

    return {
        "mean_jac": mean_jac,
        "median_jac": median_jac,
        "max_jac": max_jac,
        "strong_pairs": strong_pairs,
        "locked_pairs": locked_pairs,
        "total_pairs_per_layer": n_triu,
    }


def save_global_layer_summary(stats, num_layers, output_dir):
    """Bar charts showing per-layer full-C×C co-activation strength."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = np.arange(num_layers)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    axes[0].bar(layers, stats["mean_jac"], alpha=0.8, color="steelblue",
                label="mean")
    axes[0].bar(layers, stats["median_jac"], alpha=0.4, color="orange",
                label="median")
    axes[0].set_ylabel("Jaccard")
    axes[0].set_title("Global mean / median Jaccard per layer (all C×C pairs)")
    axes[0].legend()
    axes[0].set_xticks(layers)

    axes[1].bar(layers, stats["max_jac"], alpha=0.8, color="coral")
    axes[1].set_ylabel("Max Jaccard")
    axes[1].set_title("Global max Jaccard per layer")
    axes[1].set_xticks(layers)
    axes[1].axhline(0.9, color="red", ls="--", lw=0.8, label="J=0.9")
    axes[1].legend()

    axes[2].bar(layers, stats["strong_pairs"], alpha=0.8, color="seagreen")
    axes[2].set_ylabel("# pairs")
    axes[2].set_title("Channel pairs with Jaccard > 0.5 per layer"
                       f"  (total pairs = {stats['total_pairs_per_layer']})")
    axes[2].set_xticks(layers)

    axes[3].bar(layers, stats["locked_pairs"], alpha=0.8, color="firebrick")
    axes[3].set_ylabel("# pairs")
    axes[3].set_title("Locked channel pairs (Jaccard > 0.9) per layer")
    axes[3].set_xlabel("Layer index")
    axes[3].set_xticks(layers)

    plt.tight_layout()
    path = os.path.join(output_dir, "coactivation_global_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved global layer summary to {path}")


def save_global_heatmaps(global_coact, global_counts, num_layers, C,
                         output_dir):
    """Full C×C Jaccard heatmap for ALL layers, reordered by clustering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    ncols = 6
    nrows = (num_layers + ncols - 1) // ncols

    fig = plt.figure(figsize=(4 * ncols + 1, 4 * nrows + 0.8))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols),
                     axes_pad=0.35, cbar_location="right",
                     cbar_mode="single", cbar_size="3%", cbar_pad=0.15)

    for li in range(num_layers):
        ax = grid[li]
        jac = compute_jaccard(
            global_coact[li].cpu().float(), global_counts[li].cpu().float()
        ).numpy()
        order = cluster_order(jac)
        jac_sorted = jac[np.ix_(order, order)]

        im = ax.imshow(jac_sorted, aspect="equal", cmap="hot",
                        interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"Layer {li}", fontsize=10, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(num_layers, nrows * ncols):
        grid[i].axis("off")

    grid.cbar_axes[0].colorbar(im, label="Jaccard")
    fig.suptitle(f"Global C×C Jaccard similarity per layer "
                 f"(clustered, C={C})", fontsize=14, y=0.99)
    path = os.path.join(output_dir, "coactivation_global_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global heatmaps to {path}")


def save_global_top_pairs(global_coact, global_counts, num_layers, top_k,
                          output_dir):
    """Top co-activated channel pairs across all channels per layer."""
    path = os.path.join(output_dir, "coactivation_global_top_pairs.txt")
    with open(path, "w") as f:
        for li in range(num_layers):
            jac = compute_jaccard(
                global_coact[li].cpu().float(),
                global_counts[li].cpu().float(),
            )
            pairs = top_pairs(jac, top_k)

            f.write(f"{'=' * 70}\n")
            f.write(f"Layer {li}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"{'Rank':>5}  {'Ch_i':>6}  {'Ch_j':>6}  {'Jaccard':>9}"
                    f"  {'CoAct':>10}  {'Cnt_i':>10}  {'Cnt_j':>10}\n")
            coact_cpu = global_coact[li].cpu()
            counts_cpu = global_counts[li].cpu()
            for rank, (ci, cj, jval) in enumerate(pairs, 1):
                f.write(f"{rank:>5}  {ci:>6}  {cj:>6}  {jval:>9.4f}"
                        f"  {coact_cpu[ci, cj].item():>10.0f}"
                        f"  {counts_cpu[ci].item():>10.0f}"
                        f"  {counts_cpu[cj].item():>10.0f}\n")
            f.write("\n")
    print(f"Saved global top pairs to {path}")


def print_global_layer_summary(stats, num_layers):
    print(f"\n{'Layer':>6} {'MeanJac':>9} {'MedJac':>9} {'MaxJac':>9}"
          f" {'#J>0.5':>8} {'#J>0.9':>8}")
    print("-" * 55)
    for li in range(num_layers):
        print(f"{li:>6d} {stats['mean_jac'][li]:>9.4f}"
              f" {stats['median_jac'][li]:>9.4f}"
              f" {stats['max_jac'][li]:>9.4f}"
              f" {stats['strong_pairs'][li]:>8.0f}"
              f" {stats['locked_pairs'][li]:>8.0f}")


def print_per_group_summary(coact, counts, total_tokens, num_layers,
                            num_groups, bank_size):
    B = bank_size
    triu = torch.triu_indices(B, B, offset=1)

    print(f"\n{'Layer':>6} | ", end="")
    print(" | ".join(f"{'G' + str(g) + ' max':>8}" for g in range(num_groups)))
    print("\n" + "-" * (10 + num_groups * 11))

    for li in range(num_layers):
        row = f"{li:>6} | "
        parts = []
        for g in range(num_groups):
            jac = compute_jaccard(
                coact[li][g].cpu().float(), counts[li][g].cpu().float()
            )
            mx = jac[triu[0], triu[1]].max().item()
            parts.append(f"{mx:>8.3f}")
        row += " | ".join(parts)
        print(row)

    print()
    print(f"{'Layer':>6} | ", end="")
    print(" | ".join(f"{'G' + str(g) + ' #>.5':>8}" for g in range(num_groups)))
    print("\n" + "-" * (10 + num_groups * 11))

    for li in range(num_layers):
        row = f"{li:>6} | "
        parts = []
        for g in range(num_groups):
            jac = compute_jaccard(
                coact[li][g].cpu().float(), counts[li][g].cpu().float()
            )
            n_strong = (jac[triu[0], triu[1]] > 0.5).sum().item()
            parts.append(f"{n_strong:>8d}")
        row += " | ".join(parts)
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.pretrained} ...")
    model = SmolLMForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16
    )
    model.to(args.device).eval()
    num_layers = len(model.model.layers)
    C = model.config.intermediate_size
    B = model.config.predictor_bank_size
    G = C // B
    print(f"  {num_layers} layers, intermediate_size={C}, "
          f"bank_size={B}, num_groups={G}")

    print(f"Loading batch ({args.batch_size} x {args.seq_len}) "
          f"from {args.data_prefix} ...")
    input_ids = load_batch_from_indexed_dataset(
        args.data_prefix, args.batch_size, args.seq_len
    ).to(args.device)
    total_tokens = args.batch_size * args.seq_len
    print(f"  input_ids shape: {input_ids.shape}, total_tokens={total_tokens}")

    (coact, counts, global_coact, global_counts,
     handles, num_groups, bank_size) = register_coactivation_hooks(model)

    print("Running forward pass ...")
    with torch.no_grad():
        model(input_ids=input_ids)
    print("Done.")

    for h in handles:
        h.remove()

    if args.cluster_layers is not None:
        layer_indices = [int(x) for x in args.cluster_layers.split(",")]
    else:
        layer_indices = sorted(set([0, num_layers // 2, num_layers - 1]))

    # ---- Global (full C×C) per-layer summary ----
    global_stats = compute_global_layer_stats(global_coact, global_counts,
                                              num_layers, C)
    print_global_layer_summary(global_stats, num_layers)

    # ---- Per-group detail ----
    print_per_group_summary(coact, counts, total_tokens, num_layers,
                            num_groups, bank_size)

    # ---- Save data ----
    data_path = os.path.join(args.output_dir, "coactivation_data.pt")
    torch.save({
        "coact": {i: coact[i].cpu() for i in range(num_layers)},
        "counts": {i: counts[i].cpu() for i in range(num_layers)},
        "global_coact": {i: global_coact[i].cpu() for i in range(num_layers)},
        "global_counts": {i: global_counts[i].cpu() for i in range(num_layers)},
        "total_tokens": total_tokens,
        "num_layers": num_layers,
        "num_groups": num_groups,
        "bank_size": bank_size,
        "intermediate_size": C,
        "global_stats": global_stats,
    }, data_path)
    print(f"\nSaved raw data to {data_path}")

    # ---- Global visualizations ----
    save_global_layer_summary(global_stats, num_layers, args.output_dir)
    save_global_heatmaps(global_coact, global_counts, num_layers, C,
                         args.output_dir)
    save_global_top_pairs(global_coact, global_counts, num_layers,
                          args.top_k_pairs, args.output_dir)

    # ---- Per-group visualizations ----
    save_top_pairs_report(coact, counts, total_tokens, num_layers, num_groups,
                          bank_size, args.top_k_pairs, args.output_dir)
    save_group_heatmaps(coact, counts, layer_indices, num_groups, bank_size,
                        args.output_dir)
    save_layer_group_summary(coact, counts, num_layers, num_groups,
                             args.output_dir)
    save_group_clusters(coact, counts, layer_indices, num_groups, bank_size,
                        args.output_dir)


if __name__ == "__main__":
    main()
