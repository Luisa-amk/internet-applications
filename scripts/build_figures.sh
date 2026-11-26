#!/bin/sh
# Rebuild every figure for the poster and the report (300 dpi PNGs):
#   figures/poster/  large text, for the A0 poster
#   figures/report/  sized for one column of the 2-column IEEE report (100% width in LaTeX);
#                    task3-label-skew spans both columns (figure*)
# Run from anywhere with the venv active:  sh scripts/build_figures.sh
#
# Tasks 2 and 3 use results/seeds/: every setting run with seeds 42, 43 and 44 on one
# machine (Apple M4). Lines are the mean across seeds; shaded bands are the min–max.
# The grey dashed line is always the baseline settings (10 clients, IID).
set -e
cd "$(dirname "$0")/.."
S=results/seeds

for STYLE in poster report; do
    OUT=figures/$STYLE
    PLOT="python scripts/plot_results.py --style $STYLE"

    # Task 1: learning rate / local epochs sweep (one seed), chosen baseline in grey
    $PLOT -o $OUT/task1-sweep.png --target 0.5 \
        results/task1-baseline_*.json results/task1-sweep/sweep-lr0.01-e2_*.json \
        results/task1-sweep/sweep-lr0.05-e1_*.json results/task1-sweep/sweep-lr0.1-e1_*.json \
        --ref task1-baseline \
        --label task1-baseline="lr 0.01, 1 epoch (baseline)" \
        --label sweep-lr0.01-e2="lr 0.01, 2 epochs" \
        --label sweep-lr0.05-e1="lr 0.05, 1 epoch" \
        --label sweep-lr0.1-e1="lr 0.1, 1 epoch"

    # Task 2: number of clients, by round and by training time
    for X in round time; do
        $PLOT -o $OUT/task2-$X.png --x $X --seeds \
            $S/seeds-c10-iid-s*.json $S/seeds-c5-iid-s*.json $S/seeds-c2-iid-s*.json \
            --ref seeds-c10-iid \
            --label seeds-c10-iid="10 clients (baseline)" \
            --label seeds-c5-iid="5 clients" \
            --label seeds-c2-iid="2 clients"
    done

    # Task 3: IID vs non-IID, least to most skewed
    $PLOT -o $OUT/task3-accuracy.png --seeds \
        $S/seeds-c10-iid-s*.json $S/seeds-c10-dir1.0-s*.json \
        $S/seeds-c10-dir0.5-s*.json $S/seeds-c10-dir0.1-s*.json \
        --ref seeds-c10-iid \
        --label seeds-c10-iid="IID (baseline)" \
        --label seeds-c10-dir1.0="α = 1.0" \
        --label seeds-c10-dir0.5="α = 0.5" \
        --label seeds-c10-dir0.1="α = 0.1"

    # Task 3: how skewed the clients' data is (seed 42; loads CIFAR-10, no training)
    python scripts/plot_partitions.py --style $STYLE -o $OUT/task3-label-skew.png
done
