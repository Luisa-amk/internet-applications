"""Plot test accuracy (or loss) per round from results JSON files, and print a summary.

Examples:
    python scripts/plot_results.py                                  # every file in results/
    python scripts/plot_results.py results/task2/*.json -o figures/task2.png --x time
    python scripts/plot_results.py results/task3/*.json --metric loss

Styles: "screen" (default), "poster" (large text for A0) and "report" (sized for one
column of a 2-column IEEE paper, so LaTeX can include it at 100% width). All PNG.

Repeated seeds: with --seeds, runs whose names differ only in a trailing -s<seed> are
averaged into one line, with a shaded band showing the min–max across seeds:
    python scripts/plot_results.py --seeds results/seeds/seeds-c*-iid-s*.json

Readable figures: --ref NAME draws that run as the grey dashed baseline, --label
NAME=TEXT sets display names. Runs get colours in the order the files are given, so
list them in a meaningful order (e.g. least to most skewed). See
scripts/build_figures.sh for complete examples.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

ROOT = Path(__file__).resolve().parent.parent

# Colourblind-checked categorical colours, used in this fixed order (never cycled).
# Orange and aqua are under 3:1 contrast on white, so every line also gets its own
# marker shape and a label (direct label at the line end, or the legend).
SERIES_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600"]
SERIES_MARKERS = ["s", "^", "D", "v"]
REF_COLOR, REF_MARKER = "#6b6b6b", "o"
TEXT, TEXT_MUTED, GRID = "#1f1f1f", "#5c5c5c", "#e4e4e4"

STYLES = {
    # size in inches; font in pt; direct = labels at line ends; legend_y = gap below
    "screen": {"size": (7, 4.5), "font": 10, "line": 2.0, "marker": 5, "direct": True,
               "legend_y": -0.17},
    "poster": {"size": (7.6, 5.0), "font": 20, "line": 2.6, "marker": 7, "direct": True,
               "legend_y": -0.18},
    "report": {"size": (3.5, 2.6), "font": 8, "line": 1.3, "marker": 3, "direct": False,
               "legend_y": -0.25},
}  # fmt: skip
DPI = 300
SEED_SUFFIX = re.compile(r"-s\d+$")


def load(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        run = json.loads(path.read_text())
        run["_file"] = path.name
        runs.append(run)
    return runs


def summary_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        rounds = run["rounds"]
        accs = [r["accuracy"] for r in rounds]
        round_times = [r["round_time_s"] for r in rounds if r.get("round_time_s") is not None]
        rows.append(
            {
                "name": run["name"],
                "clients": run["num_clients"],
                "rounds": len(rounds) - 1,  # round 0 is the untrained model
                "final_acc": accs[-1] if accs else None,
                "best_acc": max(accs) if accs else None,
                "mean_round_s": sum(round_times) / len(round_times) if round_times else None,
                "total_s": run.get("total_time_s"),
                "done": run.get("finished", False),
            }
        )
    return rows


def print_summary(rows: list[dict]) -> None:
    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    print(f"{'name':<34} {'cli':>4} {'rnds':>5} {'final':>7} {'best':>7} {'s/rnd':>7} {'total':>8}")
    for r in rows:
        print(
            f"{r['name']:<34} {r['clients']:>4} {r['rounds']:>5} "
            f"{fmt(r['final_acc'], '.2%'):>7} {fmt(r['best_acc'], '.2%'):>7} "
            f"{fmt(r['mean_round_s'], '.1f'):>7} {fmt(r['total_s'], '.0f'):>8}"
            + ("" if r["done"] else "  (unfinished)")
        )


def x_values(run: dict, x: str) -> list[float]:
    if x == "round":
        return [r["round"] for r in run["rounds"]]
    # Cumulative training time; round 0 starts at 0
    xs, t = [], 0.0
    for r in run["rounds"]:
        t += r.get("round_time_s") or 0.0
        xs.append(t)
    return xs


def make_series(runs: list[dict], metric: str, x: str, seeds: bool) -> list[dict]:
    """One line per run, or with seeds=True one averaged line per group of seeds.

    Each series has name, xs, ys (mean), lo/hi (min/max across seeds, or None) and n.
    """
    groups: dict[str, list[dict]] = {}
    for run in runs:
        key = SEED_SUFFIX.sub("", run["name"]) if seeds else run["name"]
        groups.setdefault(key, []).append(run)

    series = []
    for name, members in groups.items():
        length = min(len(r["rounds"]) for r in members)
        per_run_x = [x_values(r, x)[:length] for r in members]
        per_run_y = [[rd[metric] for rd in r["rounds"][:length]] for r in members]
        many = len(members) > 1
        series.append(
            {
                "name": name,
                "xs": [statistics.fmean(col) for col in zip(*per_run_x, strict=True)],
                "ys": [statistics.fmean(col) for col in zip(*per_run_y, strict=True)],
                "lo": [min(col) for col in zip(*per_run_y, strict=True)] if many else None,
                "hi": [max(col) for col in zip(*per_run_y, strict=True)] if many else None,
                "n": len(members),
            }
        )
    return series


def print_seed_summary(runs: list[dict]) -> None:
    """Mean ± standard deviation across seeds, per group."""
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(SEED_SUFFIX.sub("", run["name"]), []).append(run)
    print(f"\n{'group':<28} {'seeds':>5} {'final acc (mean ± sd)':>22} {'s/rnd, rounds 2+':>17}")
    for name, members in groups.items():
        finals = [r["rounds"][-1]["accuracy"] for r in members]
        per_round = [
            statistics.fmean(rd["round_time_s"] for rd in r["rounds"][2:]) for r in members
        ]
        sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
        print(
            f"{name:<28} {len(members):>5} "
            f"{statistics.fmean(finals):>13.2%} ± {sd:.2%} "
            f"{statistics.fmean(per_round):>10.1f}"
        )


def is_ref(name: str, refs: list[str]) -> bool:
    return any(name == ref or name.startswith(ref) for ref in refs)


def spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Nudge label heights apart so end labels don't overlap.

    Only labels that would collide move: each group of touching labels is spaced
    min_gap apart and centred on the group's real values. Isolated labels stay put.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    groups: list[list[int]] = []
    for i in order:
        groups.append([i])
        # Merge with the group below while the two overlap, re-centring each time
        while len(groups) > 1:
            below, top = groups[-2], groups[-1]
            below_top = _centre(below, ys, min_gap) + (len(below) - 1) * min_gap
            if _centre(top, ys, min_gap) - below_top >= min_gap:
                break
            groups[-2:] = [below + top]
    placed = list(ys)
    for group in groups:
        start = _centre(group, ys, min_gap)
        for k, i in enumerate(group):
            placed[i] = start + k * min_gap
    return placed


def _centre(group: list[int], ys: list[float], min_gap: float) -> float:
    """Lowest label position for a group spaced min_gap apart around its mean value."""
    mean = sum(ys[i] for i in group) / len(group)
    return mean - (len(group) - 1) * min_gap / 2


def plot(
    runs: list[dict],
    metric: str,
    x: str,
    out: Path,
    *,
    style: str = "screen",
    refs: list[str] | None = None,
    labels: dict[str, str] | None = None,
    title: str | None = None,
    target: float | None = None,
    seeds: bool = False,
) -> Path:
    refs, labels, s = refs or [], labels or {}, STYLES[style]
    with plt.rc_context(
        {
            "font.size": s["font"],
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT_MUTED,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT,
        }
    ):
        fig, ax = plt.subplots(figsize=s["size"])
        ends = []  # (x, y, label) for direct labels
        slot = 0
        for line in make_series(runs, metric, x, seeds):
            ref = is_ref(line["name"], refs)
            if ref:
                color, marker = REF_COLOR, REF_MARKER
            else:
                color = SERIES_COLORS[slot % len(SERIES_COLORS)]
                marker = SERIES_MARKERS[slot % len(SERIES_MARKERS)]
                slot += 1
            label = labels.get(line["name"], line["name"])
            xs, ys = line["xs"], line["ys"]
            if line["lo"] is not None:
                # Min–max across seeds
                ax.fill_between(
                    xs, line["lo"], line["hi"], color=color, alpha=0.18, linewidth=0, zorder=1
                )
            ax.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                markersize=s["marker"],
                markevery=max(1, len(xs) // 10),
                linewidth=s["line"] * (0.8 if ref else 1.0),
                linestyle="--" if ref else "-",
                label=label,
                zorder=2 if ref else 3,
                clip_on=False,  # markers sit on the axis edges at round 0 and the last round
            )
            ends.append((xs[-1], ys[-1], label))

        ax.set_xlabel("Training time (s)" if x == "time" else "Round")
        ax.set_ylabel("Test accuracy" if metric == "accuracy" else "Test loss")
        if metric == "accuracy":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        if x == "round":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, max(e[0] for e in ends))
        ax.margins(x=0)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if title:
            ax.set_title(title, loc="left", fontweight="bold")
        if target is not None:
            ax.axhline(target, color=TEXT_MUTED, linewidth=s["line"] * 0.5, linestyle=":")
            ax.annotate(
                f"{target:.0%} target" if metric == "accuracy" else f"target {target:g}",
                xy=(0, target),
                xytext=(s["font"] * 0.3, s["font"] * 0.3),
                textcoords="offset points",
                xycoords=("axes fraction", "data"),
                color=TEXT_MUTED,
                fontsize=s["font"] * 0.75,
            )

        # Direct labels at the line ends (at most 4 series), plus the legend
        if s["direct"] and len(ends) <= 4:
            lo, hi = ax.get_ylim()
            axes_height_pt = ax.get_position().height * fig.get_figheight() * 72
            gap = 1.25 * s["font"] / axes_height_pt * (hi - lo)
            for (ex, _, text), ly in zip(
                ends, spread_labels([e[1] for e in ends], gap), strict=True
            ):
                ax.annotate(
                    text,
                    xy=(ex, ly),
                    xytext=(s["font"] * 0.6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=s["font"] * 0.85,
                    annotation_clip=False,
                )
        # Legend below the plot, so it never covers a line
        ax.legend(
            frameon=False,
            fontsize=s["font"] * 0.85,
            loc="upper center",
            bbox_to_anchor=(0.5, s["legend_y"]),
            ncol=2,
        )

        out = out.with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")
        plt.close(fig)
    return out


def parse_labels(pairs: list[str]) -> dict[str, str]:
    labels = {}
    for pair in pairs:
        name, sep, text = pair.partition("=")
        if not sep:
            raise SystemExit(f"--label expects NAME=TEXT, got {pair!r}")
        labels[name] = text
    return labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", type=Path, help="default: results/*.json")
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "figures" / "accuracy.png")
    parser.add_argument("--metric", choices=["accuracy", "loss"], default="accuracy")
    parser.add_argument("--x", choices=["round", "time"], default="round")
    parser.add_argument("--style", choices=list(STYLES), default="screen")
    parser.add_argument("--seeds", action="store_true", help="average runs that differ by -sN")
    parser.add_argument(
        "--ref", action="append", default=[], metavar="NAME", help="grey reference run"
    )
    parser.add_argument(
        "--label", action="append", default=[], metavar="NAME=TEXT", help="display name"
    )
    parser.add_argument("--title")
    parser.add_argument("--target", type=float, help="dotted reference line, e.g. 0.5")
    args = parser.parse_args(argv)

    files = args.files or sorted((ROOT / "results").glob("*.json"))
    if not files:
        raise SystemExit("no results files found; run scripts/run.py first")
    runs = load(files)
    print_summary(summary_rows(runs))
    if args.seeds:
        print_seed_summary(runs)
    print()
    plot(
        runs,
        args.metric,
        args.x,
        args.out,
        style=args.style,
        refs=args.ref,
        labels=parse_labels(args.label),
        title=args.title,
        target=args.target,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
