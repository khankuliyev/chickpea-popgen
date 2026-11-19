"""Plot the ALT allele-frequency spectrum from the stage 1 regional table."""
import csv
import gzip
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from project_paths import DATA, FIGURES, TABLES

REGIONS = ("ETHI", "IND", "LEB", "MOR", "TUR", "UZB")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#8F559B", "#A76B00", "#50555D")
LABELS = ["0"] + [f"{i/10:g}–{(i+1)/10:g}".replace(".", ",") for i in range(10)] + ["1"]


def main():
    spectra = defaultdict(Counter)
    with gzip.open(DATA / "stage1_frequencies.csv.gz", "rt", newline="") as source:
        for row in csv.DictReader(source):
            called = int(row["called_copies"])
            if called == 0:
                continue
            alt = int(row["alt_copies"])
            if alt == 0:
                index = 0
            elif alt == called:
                index = 11
            else:
                index = 1 + 10 * alt // called
            key = (row["order"], int(row["te_class"]), row["region"])
            spectra[key][index] += 1

    with (TABLES / "stage1_summary.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["order"], int(row["te_class"]), row["region"])
            counts = spectra[key]
            expected = (int(row["called_loci"]), int(row["alt_absent_loci"]),
                        int(row["alt_fixed_loci"]))
            if (sum(counts.values()), counts[0], counts[11]) != expected:
                raise ValueError(f"Spectrum disagrees with stage 1 summary for {key}")

    panels = {(order, te_class) for order, te_class, _ in spectra}
    panels = sorted(panels, key=lambda pair: -max(
        sum(spectra[(*pair, region)].values()) for region in REGIONS
    ))
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True)
    fig.suptitle("Спектр частот аллеля ALT по регионам", fontsize=20, y=0.98)
    fig.supylabel("Доля локусов с определёнными генотипами", x=0.018)
    width = 0.14
    for ax, (order, te_class) in zip(axes.flat, panels):
        totals = []
        for j, (region, color) in enumerate(zip(REGIONS, COLORS)):
            counts = spectra[(order, te_class, region)]
            total = sum(counts.values())
            totals.append(total)
            positions = [i + width * (j - 2.5) for i in range(len(LABELS))]
            ax.bar(positions, [counts[i] / total for i in range(len(LABELS))],
                   width=width, color=color, label=region, zorder=3)
        lowest, highest = min(totals), max(totals)
        size = str(highest) if lowest == highest else f"{lowest}–{highest}"
        ax.set_title(f"{order} · класс {te_class} · {size} локусов", loc="left")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=6, frameon=False)
    fig.supxlabel("Частота ALT · 0 и 1 вынесены в отдельные столбцы", y=0.02)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.885, bottom=0.10,
                        wspace=0.14, hspace=0.53)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage1_spectrum.{ext}", dpi=180)


if __name__ == "__main__":
    main()
