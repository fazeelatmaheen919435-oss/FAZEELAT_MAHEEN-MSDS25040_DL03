"""
visualize.py — t-SNE plots + retrieval grids
Generates:
  - 3 t-SNE scatter plots (one per experiment)
  - Retrieval grid: query + top-5 neighbors for 10 query images

Usage:
    python visualize.py \
        --dataset_path /path/to/caltech-101 \
        --exp1_emb embeddings/exp1/test \
        --exp2_emb embeddings/exp2/test \
        --exp3_emb embeddings/exp3/test \
        --model_path weights/best_model_exp3.pt
"""

import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

from dataset import split_dataset, EVAL_TRANSFORM
from retrieval import get_top_k_neighbors
from save_embeddings import load_embeddings


# ── 1.8 t-SNE Visualization ───────────────────────────────────────────────────

def plot_tsne(embeddings, labels, title='t-SNE', save_path='tsne.png',
              max_classes=30, perplexity=30):
    """
    Reduce embeddings to 2-D with t-SNE and save a scatter plot coloured
    by class label. Shows at most max_classes classes for readability.
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Sub-sample classes if there are too many
    if n_classes > max_classes:
        chosen = np.random.choice(unique_labels, size=max_classes, replace=False)
        mask = np.isin(labels, chosen)
        embeddings = embeddings[mask]
        labels     = labels[mask]
        unique_labels = chosen

    print(f"Running t-SNE on {len(embeddings)} samples, {len(unique_labels)} classes …")
    reduced = TSNE(n_components=2, perplexity=perplexity,
                   random_state=42, n_iter=1000).fit_transform(embeddings)

    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {lbl: cmap(i) for i, lbl in enumerate(sorted(unique_labels))}

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in sorted(unique_labels):
        idx = labels == lbl
        ax.scatter(reduced[idx, 0], reduced[idx, 1],
                   c=[label_to_color[lbl]], s=12, alpha=0.7, label=str(lbl))

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    if len(unique_labels) <= 20:
        ax.legend(loc='best', fontsize=6, ncol=2, markerscale=1.5)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved: {save_path}")


# ── 1.9 Retrieval Grid Visualization ─────────────────────────────────────────

def show_retrieval(query_idx, embeddings, image_paths, labels,
                   idx_to_class, save_path, k=5):
    """
    For a single query image, display the query + top-k nearest neighbors.
    Green border = correct class, Red border = wrong class.
    """
    neighbor_idxs = get_top_k_neighbors(query_idx, embeddings, k=k)

    n_cols = k + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))

    def _show_img(ax, path, title, border_color):
        try:
            img = Image.open(path).convert('RGB').resize((150, 150))
        except Exception:
            img = Image.new('RGB', (150, 150), (180, 180, 180))
        ax.imshow(img)
        ax.set_title(title, fontsize=8, wrap=True)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
        ax.set_xticks([]); ax.set_yticks([])

    query_class = labels[query_idx]
    query_name  = idx_to_class[query_class]
    _show_img(axes[0], image_paths[query_idx],
              f'QUERY\n{query_name}', 'blue')

    for col, nb_idx in enumerate(neighbor_idxs, start=1):
        nb_class = labels[nb_idx]
        nb_name  = idx_to_class[nb_class]
        color    = 'green' if nb_class == query_class else 'red'
        _show_img(axes[col], image_paths[nb_idx],
                  f'Top-{col}\n{nb_name}', color)

    plt.suptitle(f'Query: {query_name}  |  Green=correct, Red=wrong',
                 fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def visualize_retrievals(embeddings, image_paths, labels, idx_to_class,
                         out_dir, n_queries=10, k=5, seed=42):
    """
    Generate retrieval grid images for n_queries distinct query images.
    """
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    query_idxs = random.sample(range(len(labels)), n_queries)

    for i, q_idx in enumerate(query_idxs):
        save_path = os.path.join(out_dir, f'retrieval_query{i+1:02d}.png')
        show_retrieval(q_idx, embeddings, image_paths, labels,
                       idx_to_class, save_path, k=k)
        print(f"Saved retrieval grid {i+1}/{n_queries}: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualization — t-SNE + retrieval grids')
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--exp1_emb', default='embeddings/exp1/test',
                        help='Prefix path for exp1 test embeddings (no extension)')
    parser.add_argument('--exp2_emb', default='embeddings/exp2/test')
    parser.add_argument('--exp3_emb', default='embeddings/exp3/test')
    parser.add_argument('--n_queries', type=int, default=10)
    parser.add_argument('--k',         type=int, default=5)
    args = parser.parse_args()

    os.makedirs('graphs', exist_ok=True)

    # Load test split (need image paths & class map)
    _, _, test_ds = split_dataset(args.dataset_path)
    image_paths   = test_ds.get_paths()
    idx_to_class  = test_ds.idx_to_class

    # ── t-SNE for all three experiments ──────────────────────────────────────
    emb_paths = [
        (args.exp1_emb, 'Exp-1: Contrastive Loss',           'graphs/tsne_exp1.png'),
        (args.exp2_emb, 'Exp-2: Triplet Loss (Random)',      'graphs/tsne_exp2.png'),
        (args.exp3_emb, 'Exp-3: Triplet Loss (Hard Mining)', 'graphs/tsne_exp3.png'),
    ]

    for prefix, title, out in emb_paths:
        try:
            emb, lbl = load_embeddings(prefix)
            plot_tsne(emb, lbl, title=title, save_path=out)
        except FileNotFoundError:
            print(f"Embeddings not found for {title} at {prefix} — skipping")

    # ── Retrieval grids for exp3 (best model) ─────────────────────────────────
    try:
        emb3, lbl3 = load_embeddings(args.exp3_emb)
        visualize_retrievals(emb3, image_paths, lbl3, idx_to_class,
                             out_dir='graphs/retrieval_exp3',
                             n_queries=args.n_queries, k=args.k)
    except FileNotFoundError:
        print("Exp-3 embeddings not found — skipping retrieval grids")

    print("\nAll visualizations saved to graphs/")


if __name__ == '__main__':
    main()
