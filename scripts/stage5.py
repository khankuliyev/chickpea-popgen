"""Compare regional trees by distance, linkage method, and locus bootstrap."""
import csv
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from stage2 import read_panel

from project_paths import FIGURES, TABLES, TREES

BOOTSTRAPS = 1000
SEED = 2026
METHODS = ("UPGMA", "NJ")
METRICS = ("euclidean", "chord")


def locus_contributions(data, labels, pairs):
    frequencies = []
    for i in range(int(labels.max()) + 1):
        subset = data[:, labels == i]
        frequencies.append(np.maximum(subset, 0).sum(axis=1) /
                           (2 * (subset >= 0).sum(axis=1)))
    p = np.array(frequencies)
    a, b = p[pairs[:, 0]], p[pairs[:, 1]]
    euclidean = (a - b) ** 2
    chord = (np.sqrt(a) - np.sqrt(b)) ** 2 + (np.sqrt(1 - a) - np.sqrt(1 - b)) ** 2
    return np.concatenate((euclidean, chord)).T.copy()


def make_tree(distances, names, method):
    if method == "NJ":
        matrix = squareform(distances)
        lower = [matrix[i, :i + 1].tolist() for i in range(len(names))]
        return DistanceTreeConstructor().nj(DistanceMatrix(names, lower))

    def clade(node, parent_height):
        height = node.dist / 2
        branch = parent_height - height
        if node.is_leaf():
            return Clade(branch_length=branch, name=names[node.id])
        return Clade(branch_length=branch, clades=[
            clade(node.left, height), clade(node.right, height)
        ])

    root = to_tree(linkage(distances, method="average"))
    return Tree(root=clade(root, root.dist / 2), rooted=True)


def tree_splits(tree, names, method):
    all_names = set(names)
    for clade in tree.get_nonterminals():
        inside = tuple(sorted(leaf.name for leaf in clade.get_terminals()))
        if len(inside) == len(names):
            continue
        if method == "NJ":
            outside = tuple(sorted(all_names - set(inside)))
            if len(outside) < 2:
                continue
            inside = min((inside, outside), key=lambda side: (len(side), side))
        yield clade, inside


def main():
    data, names, labels = read_panel()
    pairs = np.array(list(combinations(range(len(names)), 2)))
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout="constrained")
    with (TABLES / "stage5_support.csv").open("w", newline="") as support, (
        TABLES / "stage5_distances.csv"
    ).open("w", newline="") as distances:
        branches = csv.writer(support)
        distances = csv.writer(distances)
        branches.writerow(["panel", "metric", "method", "clade_or_split_side",
                           "bootstrap_count", "replicates"])
        distances.writerow(["panel", "metric", "region_a", "region_b", "distance"])
        for panel, threshold in (("call95", .95), ("complete", 1.0)):
            selected = data[(data >= 0).mean(axis=1) >= threshold]
            values = locus_contributions(selected, labels, pairs)
            base = np.sqrt(values.mean(axis=0)).reshape(2, -1)
            counts = {(metric, method): Counter() for metric in METRICS for method in METHODS}
            for _ in range(BOOTSTRAPS):
                draw = rng.integers(len(selected), size=len(selected))
                boot = np.sqrt(values[draw].mean(axis=0)).reshape(2, -1)
                for metric, d in zip(METRICS, boot):
                    for method in METHODS:
                        tree = make_tree(d, names, method)
                        counts[metric, method].update({
                            split for _, split in tree_splits(tree, names, method)
                        })

            for i, (metric, d) in enumerate(zip(METRICS, base)):
                for (a, b), distance in zip(pairs, d):
                    distances.writerow([panel, metric, names[a], names[b], distance])
                for j, method in enumerate(METHODS):
                    tree = make_tree(d, names, method)
                    for node, split in tree_splits(tree, names, method):
                        count = counts[metric, method][split]
                        node.confidence = 100 * count / BOOTSTRAPS
                        branches.writerow([panel, metric, method, "+".join(split),
                                           count, BOOTSTRAPS])
                    Phylo.write(tree, TREES / f"stage5_{panel}_{metric}_{method.lower()}.nwk",
                                "newick", format_branch_length="%1.9f")
                    if panel == "call95":
                        ax = axes[i, j]
                        Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False,
                                   branch_labels=lambda node: f"{node.confidence:.0f}%"
                                   if node.confidence is not None else None)
                        ax.set_title(f"{method} · " + ("евклидово" if i == 0 else "хордовое"))
                        ax.set_xlabel("Генетическое расстояние")
                        ax.set_ylabel("")
                        ax.set_yticks([])
            print(panel, len(selected), flush=True)
    fig.suptitle("Деревья регионов нута · локусы с ≥95% определённых генотипов\n"
                 "Поддержка ветвей по 1 000 повторным выборкам локусов")
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage5_trees.{ext}", dpi=180)


if __name__ == "__main__":
    main()
