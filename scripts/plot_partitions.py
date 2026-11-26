"""Show how CIFAR-10 classes are split across clients: one heatmap per setting.

Each panel is clients × classes, and darker cells mean more images. IID looks even;
lower Dirichlet alpha makes each client hold mostly a few classes. Uses exactly the
same seeded split as the experiments (fedlearn.task.federated_dataset). No training,
but it loads CIFAR-10 (downloaded on first use).

Examples:
    python scripts/plot_partitions.py                     # IID, α=1.0, 0.5, 0.1, 10 clients
    python scripts/plot_partitions.py --style poster -o figures/poster/task3-label-skew.png
    python scripts/plot_partitions.py --style report -o figures/report/task3-label-skew.png
    python scripts/plot_partitions.py --alphas 0.5 -n 5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `fedlearn` imports without an install

# Sequential: one hue, light (few images) to dark (many). White = none at all.
BLUES = LinearSegmentedColormap.from_list(
    "blues",
    ["#ffffff", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"],
)
TEXT, TEXT_MUTED = "#1f1f1f", "#5c5c5c"
STYLES = {
    "screen": {"panel": 3.2, "font": 9, "cols": 4},
    "poster": {"panel": 3.6, "font": 15, "cols": 2},
    # Full page width of a 2-column IEEE paper (figure* in LaTeX)
    "report": {"panel": 1.55, "font": 6.5, "cols": 4},
}


def label_counts(labels_per_client: list[list[int]], num_classes: int) -> np.ndarray:
    """clients × classes matrix of image counts."""
    counts = np.zeros((len(labels_per_client), num_classes), dtype=int)
    for client, labels in enumerate(labels_per_client):
        counts[client] = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
    return counts


def load_counts(partitioner: str, num_clients: int, alpha: float, seed: int):
    from fedlearn.task import federated_dataset

    fds = federated_dataset(partitioner, num_clients, alpha, seed)
    parts = [fds.load_partition(i).select_columns(["label"]) for i in range(num_clients)]
    names = parts[0].features["label"].names
    return label_counts([p["label"] for p in parts], len(names)), names


def plot(panels, class_names, out: Path, style: str) -> Path:
    s = STYLES[style]
    vmax = max(counts.max() for _, counts in panels)
    with plt.rc_context(
        {
            "font.size": s["font"],
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
        }
    ):
        cols = min(s["cols"], len(panels))
        rows = -(-len(panels) // cols)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(s["panel"] * cols + 1.2, s["panel"] * 1.35 * rows),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes.ravel()
        for ax in axes[len(panels) :]:
            ax.set_visible(False)
        for ax, (title, counts) in zip(axes, panels, strict=False):
            image = ax.imshow(counts, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(title, fontweight="bold")
            ax.set_xticks(range(len(class_names)), class_names, rotation=90)
            ax.set_yticks(range(counts.shape[0]), [str(i) for i in range(counts.shape[0])])
            ax.tick_params(colors=TEXT_MUTED, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Thin white gaps between cells
            ax.set_xticks(np.arange(-0.5, counts.shape[1]), minor=True)
            ax.set_yticks(np.arange(-0.5, counts.shape[0]), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="minor", length=0)
        for row in range(rows):
            axes[row * cols].set_ylabel("Client")
        bar = fig.colorbar(image, ax=list(axes[: len(panels)]), shrink=0.8, pad=0.02)
        bar.set_label("Training images")
        bar.outline.set_visible(False)

        out = out.with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")
        plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-n", "--clients", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5, 0.1])
    parser.add_argument("--no-iid", action="store_true", help="leave out the IID panel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--style", choices=list(STYLES), default="screen")
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "figures" / "task3-label-skew.png")
    args = parser.parse_args(argv)

    setups = [] if args.no_iid else [("IID", "iid", 0.5)]
    setups += [(f"α = {a:.1f}", "dirichlet", a) for a in args.alphas]
    panels, names = [], None
    for title, partitioner, alpha in setups:
        counts, names = load_counts(partitioner, args.clients, alpha, args.seed)
        panels.append((title, counts))
        sizes = counts.sum(axis=1)
        print(f"{title:<10} images per client: min {sizes.min()}, max {sizes.max()}")
    plot(panels, names, args.out, args.style)


if __name__ == "__main__":
    main()
