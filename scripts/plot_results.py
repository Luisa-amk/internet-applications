"""Plot test accuracy (or loss) per round from results JSON files, and print a summary.

Examples:
    python scripts/plot_results.py                                  # every file in results/
    python scripts/plot_results.py results/iid-c*.json -o figures/task2.png
    python scripts/plot_results.py results/*alpha* --metric loss --x time
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def load(paths: list[Path]) -> list[dict]:
    runs = []
    for path in sorted(paths):
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


def plot(runs: list[dict], metric: str, x: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        rounds = run["rounds"]
        if x == "time":
            # Cumulative training time; round 0 starts at 0
            xs, t = [], 0.0
            for r in rounds:
                t += r.get("round_time_s") or 0.0
                xs.append(t)
        else:
            xs = [r["round"] for r in rounds]
        ax.plot(xs, [r[metric] for r in rounds], marker=".", label=run["name"])

    ax.set_xlabel("Training time (s)" if x == "time" else "Round")
    ax.set_ylabel("Test accuracy" if metric == "accuracy" else "Test loss")
    if metric == "accuracy":
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\nSaved {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", type=Path, help="default: results/*.json")
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "figures" / "accuracy.png")
    parser.add_argument("--metric", choices=["accuracy", "loss"], default="accuracy")
    parser.add_argument("--x", choices=["round", "time"], default="round")
    args = parser.parse_args(argv)

    files = args.files or list((ROOT / "results").glob("*.json"))
    if not files:
        raise SystemExit("no results files found; run scripts/run.py first")
    runs = load(files)
    print_summary(summary_rows(runs))
    plot(runs, args.metric, args.x, args.out)


if __name__ == "__main__":
    main()
