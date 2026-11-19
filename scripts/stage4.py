"""Forward Wright–Fisher grid for regional chickpea TE allele frequencies."""
import csv
from itertools import permutations, product

import matplotlib.pyplot as plt
import numpy as np

from stage1 import spreadsheet
from stage2 import fst, read_panel

from project_paths import FIGURES, TABLES

SEED = 2026
REPLICATES = 20
LOCI = 2048
SIZES = (100, 500)
TIMES = (10, 50, 150)
RATES = (0.005, 0.03)
PRIORS = {"beta_0.1_3": (0.1, 3), "beta_0.3_2": (0.3, 2)}
METRICS = ("fst", "private_alt_fraction", "frequency_variance")
FILTERS = ("all", "polymorphic")


def write_csv(name, columns, rows):
    with (TABLES / name).open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        writer.writerows(rows)


def neighbor_orders(regions):
    """Two shortest open geographic paths, identifying a path with its reverse."""
    coordinates = {region: set() for region in regions}
    for _, _, region, _, longitude, latitude in spreadsheet("chickpea_data.xlsx"):
        if longitude is not None and latitude is not None:
            coordinates[region].add((longitude, latitude))
    centers = np.array([np.mean(sorted(coordinates[r]), axis=0) for r in regions])
    lon, lat = np.radians(centers).T
    xyz = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    distance = 6371 * np.arccos(np.clip(xyz.T @ xyz, -1, 1))
    paths = [(sum(distance[a, b] for a, b in zip(path, path[1:])), path)
             for path in permutations(range(len(regions))) if path[0] < path[-1]]
    return sorted(paths)[:2]


def settings(regions, paths):
    k = len(regions)
    yield "isolation", 0.0, "", np.eye(k)
    for rate in RATES:
        yield "island", rate, "", np.eye(k) * (1 - rate) + (1 - np.eye(k)) * rate / (k - 1)
        for _, order in paths:
            # Each edge receives m/2 in both directions. Endpoints have only one edge.
            weights = np.eye(k)
            for a, b in zip(order, order[1:]):
                weights[a, b] = weights[b, a] = rate / 2
                weights[a, a] -= rate / 2
                weights[b, b] -= rate / 2
            yield "stepping_stone", rate, "-".join(regions[i] for i in order), weights


def statistics(data, labels, groups):
    alt = np.stack([np.maximum(data[:, labels == g], 0).sum(axis=1) for g in groups])
    called = 2 * np.stack([(data[:, labels == g] >= 0).sum(axis=1) for g in groups])
    freq = alt / called
    private = (alt > 0).sum(axis=0) == 1
    variance = freq.var(axis=0)
    pooled_alt = alt.sum(axis=0)
    polymorphic = (pooled_alt > 0) & (pooled_alt < called.sum(axis=0))
    result = {}
    for name, mask in (("all", slice(None)), ("polymorphic", polymorphic)):
        result[name] = np.array([fst(data[mask], labels, groups)[0],
                                 private[mask].mean(), variance[mask].mean()])
    return result


def main():
    raw, regions, labels = read_panel()
    panel = raw[(raw >= 0).mean(axis=1) >= 0.95]
    groups = range(len(regions))
    empirical = statistics(panel, labels, groups)
    paths = neighbor_orders(regions)
    write_csv("stage4_geography.csv", ["rank", "chain", "total_distance_km"],
              [(i, "-".join(regions[g] for g in path), km)
               for i, (km, path) in enumerate(paths, 1)])
    write_csv("stage4_empirical.csv", ["ascertainment", *METRICS], empirical_rows(empirical))

    rng = np.random.default_rng(SEED)
    # Use the same sampled masks across settings to control this source of variation.
    masks = panel[rng.integers(len(panel), size=(REPLICATES, LOCI))] < 0
    replicate_rows, loss_rows, grid_rows = [], [], []
    for prior, size, scenario in product(PRIORS, SIZES, settings(regions, paths)):
        model, rate, chain, weights = scenario
        n_copies = 2 * size
        config = (model, size, rate, chain, prior)
        summaries = {(time, filt): [] for time, filt in product(TIMES, FILTERS)}
        for r, mask in enumerate(masks, 1):
            ancestral = rng.beta(*PRIORS[prior], size=LOCI)
            founder_count = np.clip(np.rint(n_copies * ancestral), 1, n_copies - 1)
            rare = founder_count / n_copies <= 0.05
            counts = np.repeat(founder_count[:, None], len(regions), axis=1)
            for generation in range(1, TIMES[-1] + 1):
                probs = (counts @ weights.T) / n_copies
                counts = rng.binomial(n_copies, np.clip(probs, 0, 1))
                if generation not in TIMES:
                    continue
                lost = counts[rare] == 0
                loss_rows.append((*config, generation, r, rare.sum(),
                                  lost.all(axis=1).mean(), lost.mean()))
                sampled = rng.binomial(2, counts[:, labels] / n_copies).astype(np.int8)
                sampled[mask] = -1
                for name, values in statistics(sampled, labels, groups).items():
                    summaries[(generation, name)].append(values)
                    replicate_rows.append((*config, generation, name, r, *values))
        for (time, name), values in summaries.items():
            low, median, high = np.quantile(values, [.025, .5, .975], axis=0)
            target = empirical[name]
            inside = (low <= target) & (target <= high)
            # Equal weighting after scaling each deviation by its observed value.
            error = np.sqrt(np.mean(((median - target) / target) ** 2))
            grid_rows.append((*config, time, name, error, int(inside.sum()),
                              *np.column_stack((median, low, high)).ravel()))
        print(model, size, rate, chain, prior, flush=True)

    keys = ["model", "ne", "m", "chain", "prior", "generations"]
    write_csv("stage4_replicates.csv", [*keys, "ascertainment", "replicate", *METRICS],
              replicate_rows)
    write_csv("stage4_loss.csv", [*keys, "replicate", "initially_rare_loci",
              "globally_lost_fraction", "mean_regional_loss_fraction"], loss_rows)
    grid_rows.sort(key=lambda row: (row[6], row[7]))
    write_csv("stage4_grid.csv", [*keys, "ascertainment", "relative_rmse", "metrics_inside",
              *(f"{metric}_{stat}" for metric in METRICS for stat in ("median", "q025", "q975"))],
              grid_rows)
    plot_grid(grid_rows, empirical)


def empirical_rows(empirical):
    return [(name, *values) for name, values in empirical.items()]


def plot_grid(grid_rows, empirical):
    models = ("isolation", "island", "stepping_stone")
    labels = ("Изоляция", "Островная", "Ступенчатая")
    titles = ("Общий FST", "Доля приватных ALT", "Дисперсия частот")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    for i, name in enumerate(FILTERS):
        best = [next(row for row in grid_rows if row[6] == name and row[0] == model)
                for model in models]
        for j, title in enumerate(titles):
            ax = axes[i, j]
            ax.axvline(empirical[name][j], color="#b3502e", label="Данные")
            for y, row in enumerate(best):
                median, low, high = row[9 + j * 3:12 + j * 3]
                ax.hlines(y, low, high, color="#386a8c", linewidth=2)
                ax.scatter(median, y, color="#386a8c", zorder=3)
            ax.set_yticks(range(len(models)), labels if j == 0 else [])
            ax.invert_yaxis()
            ax.set_title(title)
            ax.grid(axis="x", alpha=.2)
            if j == 0:
                ax.set_ylabel("Все локусы" if i == 0 else "Полиморфные локусы")
    axes[0, 2].legend(frameon=False, loc="best")
    fig.suptitle("Ближайшие настройки трёх моделей Райта–Фишера\n"
                 "Медианы и центральные 95% симуляций")
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage4_models.{ext}", dpi=180)


if __name__ == "__main__":
    main()
