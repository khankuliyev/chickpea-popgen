"""Summarize regional ALT frequencies and sensitivity to sample size."""
import csv
import gzip
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from stage2 import read_panel

from project_paths import DATA, FIGURES, TABLES

REPLICATES = 100
SEED = 2026
BINS = ("0", "(0, 5%]", "(5%, 25%]", "(25%, 50%]",
        "(50%, 75%]", "(75%, 100%)", "100%")


def summarize(data, groups):
    alt = np.stack([np.maximum(data[:, i], 0).sum(axis=1) for i in groups])
    called = 2 * np.stack([(data[:, i] >= 0).sum(axis=1) for i in groups])
    valid = (called > 0).all(axis=0)
    freq = np.divide(alt, called, out=np.full(alt.shape, np.nan), where=called > 0)
    occupied = (alt > 0).sum(axis=0)
    return freq, occupied, valid


def regional_stats(freq, occupied, valid, rare):
    sites = freq[:, valid]
    private = (sites > 0) & (occupied[valid] == 1)
    return np.stack([(sites == 0).mean(axis=1), (sites == 1).mean(axis=1),
                     private.mean(axis=1), (freq[:, rare & valid] > 0).mean(axis=1)])


def save_csv(name, header, rows):
    with (TABLES / name).open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    data, regions, labels = read_panel()
    groups = [np.flatnonzero(labels == i) for i in range(len(regions))]
    n = min(map(len, groups))
    rng = np.random.default_rng(SEED)
    draws = [[rng.choice(group, n, replace=False) for group in groups]
             for _ in range(REPLICATES)]
    classes = []
    with gzip.open(DATA / "stage1_frequencies.csv.gz", "rt") as source:
        for row in csv.DictReader(source):
            if row["region"] == regions[0]:
                classes.append(f'{row["order"]} / {row["te_class"]}')
    classes = np.array(classes)
    if len(classes) != len(data):
        raise ValueError("Annotation and genotype row counts differ")
    regional_rows, occupancy_rows, variance_rows, panel_rows = [], [], [], []
    metrics = ("alt_absent", "alt_fixed", "private_alt", "rare_alt_detected")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for j, (name, threshold) in enumerate((("call95", 0.95), ("complete", 1.0))):
        mask = (data >= 0).mean(axis=1) >= threshold
        panel, panel_classes = data[mask], classes[mask]
        freq, occupied, valid = summarize(panel, groups)
        pooled = np.maximum(panel, 0).sum(axis=1) / (2 * (panel >= 0).sum(axis=1))
        rare = (pooled > 0) & (pooled <= 0.05)
        variance = freq.var(axis=0)
        bins = np.searchsorted([0, .05, .25, .50, .75], pooled, side="left")
        bins[pooled == 1] = 6
        for te_class in np.unique(panel_classes):
            for i, label in enumerate(BINS):
                values = variance[valid & (panel_classes == te_class) & (bins == i)]
                if len(values):
                    variance_rows.append((name, te_class, label, len(values),
                                          values.mean(), *np.quantile(values, [.25, .5, .75])))
        observed = regional_stats(freq, occupied, valid, rare)
        balanced, balanced_occupancy, balanced_valid = [], [], []
        for draw in draws:
            f, o, v = summarize(panel, draw)
            balanced.append(regional_stats(f, o, v, rare))
            balanced_occupancy.append(np.bincount(o[rare & v], minlength=7) /
                                      (rare & v).sum())
            balanced_valid.append(v.sum())
        balanced = np.array(balanced)
        low, high = np.quantile(balanced, [.025, .975], axis=0)
        mean = balanced.mean(axis=0)
        for i, metric in enumerate(metrics):
            for k, region in enumerate(regions):
                regional_rows.append((name, region, len(groups[k]), n, metric,
                                      observed[i, k], mean[i, k], low[i, k], high[i, k]))
        rare_counts = np.bincount(occupied[rare & valid], minlength=7)
        obs_occ = rare_counts / rare_counts.sum()
        bal_occ = np.array(balanced_occupancy)
        occ_low, occ_high = np.quantile(bal_occ, [.025, .975], axis=0)
        for i in range(7):
            occupancy_rows.append((name, i, rare_counts[i], obs_occ[i],
                                   bal_occ[:, i].mean(), occ_low[i], occ_high[i]))
        panel_rows.append((name, len(panel), valid.sum(), rare_counts.sum(),
                           (valid & (occupied == 1)).sum() / valid.sum(),
                           np.nanmean(variance), min(balanced_valid), max(balanced_valid)))
        x = np.arange(len(regions))
        ax = axes[0, j]
        ax.bar(x - .18, 100 * observed[2], .36, label="Все образцы", color="#416c9a")
        ax.bar(x + .18, 100 * mean[2], .36, label=f"По {n} образцов", color="#d78543")
        ax.vlines(x + .18, 100 * low[2], 100 * high[2], color="#282828")
        ax.set_xticks(x, regions)
        ax.set_ylabel("Приватный ALT, % локусов")
        ax.set_title("Определено ≥95% генотипов" if j == 0 else "Панель без пропусков")
        if j == 0:
            ax.legend(frameon=False)
        ax = axes[1, j]
        k = np.arange(7)
        ax.bar(k - .18, 100 * obs_occ, .36, color="#416c9a")
        ax.bar(k + .18, 100 * bal_occ.mean(axis=0), .36, color="#d78543")
        ax.vlines(k + .18, 100 * occ_low, 100 * occ_high, color="#282828")
        ax.set(xlabel="Число регионов, где обнаружен ALT",
               ylabel="Доля редких локусов, %", xticks=k)
    for row in axes:
        upper = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(0, upper)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage3_drift.{ext}", dpi=180)
    save_csv("stage3_regional.csv",
             ["panel", "region", "samples", "balanced_samples", "metric", "observed",
              "balanced_mean", "balanced_q025", "balanced_q975"], regional_rows)
    save_csv("stage3_rare_occupancy.csv",
             ["panel", "regions_detected", "observed_loci", "observed_fraction",
              "balanced_mean", "balanced_q025", "balanced_q975"], occupancy_rows)
    save_csv("stage3_variance.csv",
             ["panel", "te_group", "pooled_alt_bin", "loci", "mean_variance",
              "q25", "median", "q75"], variance_rows)
    save_csv("stage3_summary.csv",
             ["panel", "loci", "loci_all_regions", "rare_loci", "private_alt_fraction",
              "mean_frequency_variance", "balanced_loci_min", "balanced_loci_max"], panel_rows)


if __name__ == "__main__":
    main()
