"""Join chickpea TE genotypes and annotations and summarize regional frequencies."""
import csv
import gzip
from collections import Counter, defaultdict

from openpyxl import load_workbook

from project_paths import DATA, TABLES

GENOTYPES = {"0/0": (0, 2), "0/1": (1, 2), "1/0": (1, 2),
             "1/1": (2, 2), "./.": (0, 0)}


def spreadsheet(name):
    workbook = load_workbook(DATA / name, read_only=True, data_only=True)
    try:
        yield from workbook.active.iter_rows(min_row=2, values_only=True)
    finally:
        workbook.close()


def main():
    groups = defaultdict(list)
    metadata = {}
    for sample, _, region, *_ in spreadsheet("chickpea_data.xlsx"):
        key = str(int(sample.removeprefix("vir_")))
        if key in metadata:
            raise ValueError(f"Duplicate metadata sample: {sample}")
        metadata[key] = region

    annotations = defaultdict(set)
    for chrom, pos, family, order, te_class, count, _ in spreadsheet("Full_joint_SHORT.xlsx"):
        annotations[(chrom, pos)].add((family, order, te_class, count))

    audit = Counter()
    summary = defaultdict(Counter)
    seen = set()
    with gzip.open(DATA / "chickpea_TE.norm.vcf.gz", "rt") as vcf, (
        DATA / "stage1_frequencies.csv.gz"
    ).open("wb") as raw:
        with gzip.open(raw, "wt", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["chrom", "position", "ref", "alt", "family", "order",
                             "te_class", "region", "alt_copies", "called_copies",
                             "alt_frequency"])
            for line in vcf:
                if line.startswith("#CHROM"):
                    samples = line.rstrip().split("\t")[9:]
                    break
            for index, sample in enumerate(samples):
                if sample in metadata:
                    groups[metadata[sample]].append(index)
            print("Regional sample counts", {r: len(i) for r, i in sorted(groups.items())})
            print("Metadata-only samples", sorted(metadata.keys() - set(samples)))
            print("VCF-only sample count", len(set(samples) - metadata.keys()))
            for line in vcf:
                chrom, pos, _, ref, alt, _, _, _, fmt, *calls = line.rstrip().split("\t")
                if fmt != "GT" or "," in alt or len(calls) != len(samples):
                    raise ValueError(f"Unexpected VCF record at {chrom}:{pos}")
                key = (chrom, int(pos))
                seen.add(key)
                audit["VCF records"] += 1
                matches = annotations.get(key, ())
                if len(matches) == 1:
                    family, order, te_class, count = next(iter(matches))
                    audit["unambiguous annotation"] += 1
                    audit["count equals ALT carriers"] += count == sum(
                        gt in ("0/1", "1/0", "1/1") for gt in calls
                    )
                    audit["count equals REF carriers"] += count == sum(
                        gt in ("0/1", "1/0", "0/0") for gt in calls
                    )
                else:
                    family = order = te_class = ""
                    audit["ambiguous annotation" if matches else "missing annotation"] += 1
                for region, indices in sorted(groups.items()):
                    alt_count, called = map(sum, zip(*(GENOTYPES[calls[i]] for i in indices)))
                    frequency = alt_count / called if called else None
                    writer.writerow([chrom, pos, ref, alt, family, order, te_class,
                                     region, alt_count, called, frequency])
                    if frequency is not None:
                        stats = summary[(region, order, te_class)]
                        stats["loci"] += 1
                        stats["frequency_sum"] += frequency
                        stats["alt_absent"] += alt_count == 0
                        stats["alt_fixed"] += alt_count == called

    with (TABLES / "stage1_summary.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["region", "order", "te_class", "called_loci",
                         "mean_alt_frequency", "alt_absent_loci", "alt_fixed_loci"])
        for (region, order, te_class), stats in summary.items():
            writer.writerow([region, order, te_class, stats["loci"],
                             stats["frequency_sum"] / stats["loci"],
                             stats["alt_absent"], stats["alt_fixed"]])
    audit["unique VCF positions"] = len(seen)
    audit["annotation positions absent from VCF"] = len(annotations.keys() - seen)
    for label, value in audit.items():
        print(label, value)


if __name__ == "__main__":
    main()
