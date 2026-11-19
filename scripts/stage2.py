"""Regional AMOVA of allele dosages and Weir–Cockerham F_ST."""
import csv
import gzip
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stage1 import GENOTYPES, spreadsheet

from project_paths import DATA, FIGURES, TABLES

PERMUTATIONS = 9999
SEED = 2026


def read_panel(*, include_sample_ids=False):
    metadata = {
        str(int(sample.removeprefix("vir_"))): region
        for sample, _, region, *_ in spreadsheet("chickpea_data.xlsx")
    }
    regions = sorted(set(metadata.values()))
    genotypes = {gt.encode(): alt if called else -1
                 for gt, (alt, called) in GENOTYPES.items()}
    rows = []
    with gzip.open(DATA / "chickpea_TE.norm.vcf.gz", "rb") as source:
        for line in source:
            if line.startswith(b"#CHROM"):
                samples = line.rstrip().split(b"\t")[9:]
                break
        indices = [i for i, s in enumerate(samples) if s.decode() in metadata]
        labels = np.array([regions.index(metadata[samples[i].decode()]) for i in indices])
        for line in source:
            fields = line.rstrip().split(b"\t")[9:]
            rows.append(bytes(genotypes[fields[i]] + 1 for i in indices))
    dosage = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(-1, len(indices))
    panel = dosage.astype(np.int8) - 1, regions, labels
    if include_sample_ids:
        return *panel, [samples[i].decode() for i in indices]
    return panel


def amova(data, labels):
    """One-level AMOVA with squared Euclidean distance on mean-imputed dosages."""
    called = data >= 0
    means = np.where(called, data, 0).sum(axis=1) / called.sum(axis=1)
    centered = np.where(called, data - means[:, None], 0.0)
    gram = centered.T @ centered / len(data)
    n = len(labels)
    sizes = np.bincount(labels)
    k = len(sizes)
    total = np.trace(gram)
    effective_size = (n - sizes @ sizes / n) / (k - 1)

    def components(group_labels):
        between = sum(gram[np.ix_(idx, idx)].sum() / len(idx)
                      for idx in (np.flatnonzero(group_labels == g) for g in range(k)))
        within = (total - between) / (n - k)
        among = (between / (k - 1) - within) / effective_size
        return among, within, among / (among + within)

    among, within, phi = components(labels)
    rng = np.random.default_rng(SEED)
    null = np.array([components(rng.permutation(labels))[2] for _ in range(PERMUTATIONS)])
    p = (1 + np.count_nonzero(null >= phi)) / (PERMUTATIONS + 1)
    return among, within, phi, p, *np.quantile(null, [0.025, 0.975])


def fst(data, labels, groups):
    """Ratio of summed variance components across loci (Weir & Cockerham, 1984)."""
    subsets = [data[:, labels == group] for group in groups]
    sizes = np.stack([(group >= 0).sum(axis=1) for group in subsets])
    alleles = np.stack([np.maximum(group, 0).sum(axis=1) for group in subsets])
    heterozygotes = np.stack([(group == 1).sum(axis=1) for group in subsets])
    usable = (sizes >= 2).all(axis=0)
    sizes, alleles, heterozygotes = (array[:, usable] for array in
                                    (sizes, alleles, heterozygotes))
    r = len(groups)
    total = sizes.sum(axis=0)
    n_bar = total / r
    n_c = (total - (sizes.astype(float) ** 2).sum(axis=0) / total) / (r - 1)
    p = alleles / (2 * sizes)
    p_bar = alleles.sum(axis=0) / (2 * total)
    h_bar = heterozygotes.sum(axis=0) / total
    s2 = (sizes * (p - p_bar) ** 2).sum(axis=0) / ((r - 1) * n_bar)
    residual = p_bar * (1 - p_bar) - (r - 1) / r * s2
    a = n_bar / n_c * (s2 - (residual - h_bar / 4) / (n_bar - 1))
    b = n_bar / (n_bar - 1) * (residual - (2 * n_bar - 1) * h_bar / (4 * n_bar))
    c = h_bar / 2
    return a.sum() / (a + b + c).sum(), int(usable.sum())


def main():
    data, regions, labels = read_panel()
    groups = range(len(regions))
    panels = {"call95": data[(data >= 0).mean(axis=1) >= 0.95],
              "complete": data[(data >= 0).all(axis=1)]}
    with (TABLES / "stage2_missingness.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["region", "samples", "missing_rate_all_loci"])
        for group, region in enumerate(regions):
            print(region, (data[:, labels == group] < 0).mean(), flush=True)
            writer.writerow([region, (labels == group).sum(),
                             (data[:, labels == group] < 0).mean()])

    with (TABLES / "stage2_amova.csv").open("w", newline="") as out, (
        TABLES / "stage2_fst.csv"
    ).open("w", newline="") as pairs:
        summary, pair_writer = csv.writer(out), csv.writer(pairs)
        summary.writerow(["panel", "loci", "variance_among", "variance_within",
                          "phi_st", "permutation_p", "null_2.5%", "null_97.5%",
                          "permutations", "global_wc_fst", "fst_loci"])
        pair_writer.writerow(["panel", "region_a", "region_b", "wc_fst", "loci"])
        matrices = []
        for name, panel in panels.items():
            result = [name, len(panel), *amova(panel, labels), PERMUTATIONS,
                      *fst(panel, labels, groups)]
            summary.writerow(result)
            print(result, flush=True)
            matrix = np.full((len(regions), len(regions)), np.nan)
            for i, j in combinations(groups, 2):
                value, loci = fst(panel, labels, (i, j))
                matrix[i, j] = matrix[j, i] = value
                pair_writer.writerow([name, regions[i], regions[j], value, loci])
            matrices.append(matrix)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    minimum = min(0, *(np.nanmin(matrix) for matrix in matrices))
    maximum = max(np.nanmax(matrix) for matrix in matrices)
    for ax, title, matrix in zip(axes, ("≥95% генотипов", "Без пропусков"), matrices):
        image = ax.imshow(matrix, cmap="Blues", vmin=minimum, vmax=maximum)
        ax.set_title(title)
        ax.set_xticks(list(groups), regions)
        ax.set_yticks(list(groups), regions)
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i, j]
            if i != j:
                color = "white" if image.norm(value) > 0.65 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color)
    fig.suptitle("Попарный FST по Вейру–Кокерхэму")
    fig.colorbar(image, ax=axes, shrink=0.8, label="FST")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage2_fst.{suffix}", dpi=180)


if __name__ == "__main__":
    main()
