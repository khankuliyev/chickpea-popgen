"""Compare geography and genetic ordination at the sampling-location level."""
import csv
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform

from stage1 import spreadsheet
from stage2 import plt, read_panel

from project_paths import DATA, FIGURES, TABLES

PERMUTATIONS = 9999
SEED = 2026


def metadata_for_sites(sample_ids):
    metadata = {
        str(int(sample.removeprefix("vir_"))): (location, region, lon, lat)
        for sample, location, region, _, lon, lat in spreadsheet("chickpea_data.xlsx")
    }
    sites = defaultdict(list)
    for index, sample in enumerate(sample_ids):
        location, region, lon, lat = metadata[sample]
        if lon is None or lat is None:
            continue
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"Invalid longitude or latitude at {location}")
        sites[(location, region, lon, lat)].append(index)
    keys = sorted(sites, key=lambda key: (key[1], key[0]))
    if len({key[0] for key in keys}) != len(keys):
        raise ValueError("A location identifier has conflicting metadata")
    if len({key[2:] for key in keys}) != len(keys):
        raise ValueError("Different locations share the same coordinates")
    return keys, [sites[key] for key in keys]


def geographic_ordination(sites):
    lon, lat = np.radians([key[2:] for key in sites]).T
    xyz = np.column_stack((np.cos(lat) * np.cos(lon),
                           np.cos(lat) * np.sin(lon), np.sin(lat)))
    distance = 6371 * np.arccos(np.clip(xyz @ xyz.T, -1, 1))
    np.fill_diagonal(distance, 0)
    squared = distance ** 2
    centered = -0.5 * (squared - squared.mean(axis=0) -
                       squared.mean(axis=1, keepdims=True) + squared.mean())
    values, vectors = np.linalg.eigh(centered)
    coordinates = vectors[:, -2:] * np.sqrt(values[-2:])
    fraction = values[-2:].sum() / values[values > 0].sum()
    return coordinates, fraction


def site_frequencies(panel, groups):
    called = (panel >= 0).sum(axis=1)
    pooled = np.maximum(panel, 0).sum(axis=1) / (2 * called)
    frequencies, imputed = [], 0
    for indices in groups:
        subset = panel[:, indices]
        n = 2 * (subset >= 0).sum(axis=1)
        alt = np.maximum(subset, 0).sum(axis=1)
        frequencies.append(np.divide(alt, n, out=pooled.copy(), where=n > 0))
        imputed += np.count_nonzero(n == 0)
    return np.array(frequencies), imputed / (len(groups) * len(panel))


def genetic_ordination(freq, metric):
    features = freq if metric == "euclidean" else np.hstack(
        (np.sqrt(freq), np.sqrt(1 - freq)))
    squared = squareform(pdist(features, "sqeuclidean"))
    gram = -0.5 * (squared - squared.mean(axis=0) -
                   squared.mean(axis=1, keepdims=True) + squared.mean())
    values, vectors = np.linalg.eigh(gram)
    return vectors[:, -2:] * np.sqrt(values[-2:]), values[-2:].sum() / values.sum()


def unit_configuration(coords):
    centered = coords - coords.mean(axis=0)
    return centered / np.linalg.norm(centered)


def protest(geographic, genetic, regions):
    x, y = map(unit_configuration, (geographic, genetic))
    u, s, vt = np.linalg.svd(y.T @ x)
    observed = s.sum()
    fitted = observed * (y @ (u @ vt))
    rng = np.random.default_rng(SEED)
    blocks = [np.flatnonzero(regions == region) for region in np.unique(regions)]
    indices = np.arange(len(x))
    global_exceed = regional_exceed = 0
    for _ in range(PERMUTATIONS):
        global_draw = rng.permutation(indices)
        regional_draw = indices.copy()
        for block in blocks:
            regional_draw[block] = rng.permutation(block)
        for draw, restricted in ((global_draw, False), (regional_draw, True)):
            value = np.linalg.svd(y[draw].T @ x, compute_uv=False).sum()
            if value >= observed - 1e-12:
                if restricted:
                    regional_exceed += 1
                else:
                    global_exceed += 1
    return (observed, (global_exceed + 1) / (PERMUTATIONS + 1),
            (regional_exceed + 1) / (PERMUTATIONS + 1), x, fitted)


def main():
    data, _, _, sample_ids = read_panel(include_sample_ids=True)
    sites, groups = metadata_for_sites(sample_ids)
    regions = np.array([key[1] for key in sites])
    geography, geo_fraction = geographic_ordination(sites)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), layout="constrained")
    with (TABLES / "stage6_procrustes.csv").open("w", newline="") as summary, (
        DATA / "stage6_sites.csv"
    ).open("w", newline="") as residuals:
        stats, points = csv.writer(summary), csv.writer(residuals)
        stats.writerow(["panel", "metric", "samples", "sites", "loci",
                        "genetic_axis_variance", "geographic_positive_axis_variance",
                        "imputed_site_locus_fraction", "procrustes_r", "p_global",
                        "p_within_region", "permutations"])
        points.writerow(["panel", "metric", "location", "region", "longitude", "latitude",
                         "samples", "geo_axis_1", "geo_axis_2", "fit_axis_1",
                         "fit_axis_2", "residual"])
        for panel_name, threshold in (("call95", .95), ("complete", 1.0)):
            panel = data[(data >= 0).mean(axis=1) >= threshold]
            freq, imputed = site_frequencies(panel, groups)
            for ax, metric in zip(axes, ("euclidean", "chord")):
                genetic, gen_fraction = genetic_ordination(freq, metric)
                r, p_global, p_region, x, fitted = protest(geography, genetic, regions)
                result = [panel_name, metric, sum(map(len, groups)), len(sites), len(panel),
                          gen_fraction, geo_fraction, imputed, r, p_global, p_region,
                          PERMUTATIONS]
                stats.writerow(result)
                print(result, flush=True)
                errors = np.linalg.norm(x - fitted, axis=1)
                for key, group, geo, fit, error in zip(sites, groups, x, fitted, errors):
                    points.writerow([panel_name, metric, *key, len(group), *geo, *fit, error])
                if panel_name != "call95":
                    continue
                ax.quiver(x[:, 0], x[:, 1], (fitted - x)[:, 0], (fitted - x)[:, 1],
                          angles="xy", scale_units="xy", scale=1, color="#666666",
                          alpha=.24, width=.002, zorder=1)
                for region in np.unique(regions):
                    coords = x[regions == region]
                    ax.scatter(*coords.T, s=24, label=region, zorder=2)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("Географическая ось 1")
                ax.set_ylabel("Географическая ось 2")
                label = "Евклидово" if metric == "euclidean" else "Хордовое"
                ax.set_title(f"{label} расстояние · r = {r:.3f}")
        handles, names = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, names, loc="best", frameon=True, ncol=2)
    fig.suptitle("Анализ Прокруста по местам сбора\n"
                 "Точка — географическая позиция, стрелка — к совмещённой генетической позиции")
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage6_procrustes.{ext}", dpi=180)


if __name__ == "__main__":
    main()
