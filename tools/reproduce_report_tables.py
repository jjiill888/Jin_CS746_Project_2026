#!/usr/bin/env python3
"""Reproduce report tables from the released aggregate results (standard library only)."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    with (ROOT / relative).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def percent(value, places=3):
    unit = Decimal(1).scaleb(-places)
    return str((Decimal(str(value)) * 100).quantize(unit, rounding=ROUND_HALF_UP))


def mean(rows, key):
    return sum(Decimal(row[key]) for row in rows) / len(rows)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def build_tables():
    cohorts = read("results/p1_eda_v1/cohorts.csv")
    main_cohorts = [row for row in cohorts if row["cohort"] != "Study 1 evaluation (encoded)"]
    cohort_rows = [[r[k] for k in ("cohort", "speakers", "bona_fide", "spoof", "attacks")] for r in main_cohorts]
    expected_counts = [("86", "2580", "0"), ("40", "1200", "0"), ("60", "1800", "9600"), ("60", "3000", "4800")]
    if [(r[1], r[2], r[3]) for r in cohort_rows] != expected_counts:
        raise ValueError("Cohort counts differ from the report")

    records = read("results/asvspoof5_adaptation_v1/analysis/primary.csv")
    if len(records) != 20 or len({r["index"] for r in records}) != 20:
        raise ValueError("Expected 20 distinct models")
    conditions = [("original", "Original"), ("full_ce", "CE adaptation"), ("kd10", "KD10 adaptation"), ("source_only", "Source-only")]
    expected = {
        ("xlsr_lcnn", "original"): ("2.056", "79.646"),
        ("xlsr_lcnn", "full_ce"): ("2.963", "85.694"),
        ("xlsr_lcnn", "kd10"): ("3.093", "81.507"),
        ("xlsr_lcnn", "source_only"): ("2.963", "79.490"),
        ("aasist", "original"): ("1.889", "13.188"),
        ("aasist", "full_ce"): ("1.037", "19.372"),
        ("aasist", "kd10"): ("1.426", "21.049"),
        ("aasist", "source_only"): ("3.852", "13.368"),
    }
    study1, means = [], {}
    for family, name in [("xlsr_lcnn", "XLSR-LCNN"), ("aasist", "AASIST")]:
        for method, label in conditions:
            rows = [r for r in records if r["family"] == family and r["method"] == method]
            expected_n = 1 if method == "original" else 3
            if len(rows) != expected_n:
                raise ValueError(f"Wrong model count for {family}/{method}")
            if method != "original" and {r["seed"] for r in rows} != {"20260912", "20260913", "20260914"}:
                raise ValueError("Training seeds do not match the protocol")
            fa, tpr = mean(rows, "macro_FA"), mean(rows, "attack_macro_TPR")
            values = percent(fa), percent(tpr)
            if values != expected[family, method]:
                raise ValueError(f"Study-1 report mismatch for {family}/{method}: {values}")
            study1.append([name, label, *values])
            means[family, method] = fa, tpr

    confirmed = read("results/asvspoof5_selfcal_v1/analysis/family_primary.csv")
    if len(confirmed) != 2:
        raise ValueError("Expected two confirmation families")
    expected_confirmation = {
        "aasist": ("5.99", "1.84", "10.11", "0.32", "-1.63", "2.14", "0.0064"),
        "xlsr_lcnn": ("-1.05", "-2.55", "0.22", "-0.94", "-3.73", "1.28", "0.927"),
    }
    study2 = []
    for family, name in [("aasist", "AASIST"), ("xlsr_lcnn", "XLSR-LCNN")]:
        row = next(r for r in confirmed if r["family"] == family)
        if row["adapted_models"] != "6":
            raise ValueError("Each confirmation family must average six adapted models")
        numbers = [percent(row[k], 2) for k in ("delta_TPR", "TPR_low", "TPR_high", "delta_FA", "FA_low", "FA_high")]
        places = 4 if family == "aasist" else 3
        p = str(Decimal(row["holm_adjusted_p"]).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
        if tuple(numbers + [p]) != expected_confirmation[family]:
            raise ValueError(f"Confirmation report mismatch for {family}")
        study2.append([name, numbers[0], f"[{numbers[1]}, {numbers[2]}]", numbers[3], f"[{numbers[4]}, {numbers[5]}]", p])

    deltas = [percent(means[f, "full_ce"][1] - means[f, "original"][1], 2) for f in ("xlsr_lcnn", "aasist")]
    source_delta = percent(means["xlsr_lcnn", "full_ce"][1] - means["xlsr_lcnn", "source_only"][1], 2)
    if deltas != ["6.05", "6.18"] or source_delta != "6.20":
        raise ValueError("Detection differences do not match the report")

    return "\n\n".join([
        "# Reproduced project report tables",
        "## Table 1. Target-domain cohorts",
        markdown_table(["Role", "Speakers", "Genuine", "Spoofed", "Attacks"], cohort_rows),
        "## Table 2. Study 1 (percent; three-seed means except originals)",
        markdown_table(["Detector", "Condition", "FA", "TPR"], study1),
        "## Table 3. Study 2 (per-speaker minus pooled; percentage points)",
        markdown_table(["Detector", "Delta TPR", "95% CI", "Delta FA", "95% CI", "Holm p"], study2),
        f"CE detection gains: XLSR-LCNN +{deltas[0]} pp; AASIST +{deltas[1]} pp. XLSR-LCNN CE versus source-only: +{source_delta} pp.",
        "All report-value checks passed. These are calculations from saved aggregate results, not a new neural inference or statistical replication.",
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    args = parser.parse_args()
    text = build_tables()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
