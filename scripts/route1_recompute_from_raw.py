"""Recompute Route 1 endpoints from frozen E1/E4 raw JSON.

No model inference or training. E1 and E4 aggregates are converted to the
frozen co-primary and secondary endpoint set. Primary threshold contrasts use
paired cluster bootstrap over target regions and seeds.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
RAW_E1 = REPORTS / "e1_raw"
RAW_E4 = REPORTS / "e4_query_raw"
MANIFEST = PROJECT / "data/processed/benchmark_v2_regions_512/manifest_benchmark_v2.csv"
OUT = REPORTS / "route1_recomputed"
OUT.mkdir(parents=True, exist_ok=True)

E1_REGIMES = ["cas-single", "cas-multistream", "pooled"]
E4_MODES = ["full", "decoder-only"]
E4_BUFFERS = [0, 256, 512]
SEEDS = [42, 2026, 777]
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
LABELS = {
    "hokkaido_iburi_tobu": "Hokkaido",
    "lombok": "Lombok",
    "palu": "Palu",
    "target_macro": "Target Macro",
}
THRESHOLDS = ["0.3", "0.4", "0.5", "0.6", "0.7"]
PRIMARY_THRESHOLD = "0.5"
RNG_SEED = 20260925
N_BOOT = 10000

ENDPOINTS = [
    "iou", "dice", "precision", "recall", "f1", "accuracy",
    "mcc", "balanced_accuracy", "balanced_iou", "background_iou",
    "bf1_2", "bf1_4", "hd95",
]
HIGHER_IS_BETTER = {name: name != "hd95" for name in ENDPOINTS}


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return float(statistics.fmean(values)) if values else float("nan")


def reconstruct_counts(metrics: dict, total_pixels: int):
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    accuracy = float(metrics["accuracy"])
    denominator = 1.0 / precision + 1.0 / recall - 2.0
    tp = (1.0 - accuracy) * total_pixels / denominator
    fp = tp * (1.0 / precision - 1.0)
    fn = tp * (1.0 / recall - 1.0)
    tn = total_pixels - tp - fp - fn
    return tp, fp, fn, tn


def derived_metrics(metrics: dict, total_pixels: int):
    tp, fp, fn, tn = reconstruct_counts(metrics, total_pixels)
    mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0
    sensitivity = tp / max(tp + fn, 1e-12)
    specificity = tn / max(tn + fp, 1e-12)
    foreground_iou = tp / max(tp + fp + fn, 1e-12)
    background_iou = tn / max(tn + fp + fn, 1e-12)
    return {
        "iou": float(metrics["iou"]),
        "dice": float(metrics["dice"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "mcc": mcc,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "balanced_iou": 0.5 * (foreground_iou + background_iou),
        "background_iou": background_iou,
    }


def target_counts():
    counts = defaultdict(int)
    for row in read_csv(MANIFEST):
        counts[row["region"]] += 1
    return counts


def macro_metric(rows):
    result = {}
    for endpoint in ENDPOINTS:
        vals = [float(row[endpoint]) for row in rows if endpoint in row and not math.isnan(float(row[endpoint]))]
        result[endpoint] = mean(vals)
    return result


def add_macro(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    macro = []
    for key, entries in groups.items():
        values = macro_metric(entries)
        record = {name: value for name, value in zip(keys, key)}
        record.update(values)
        record["region"] = "target_macro"
        record["region_label"] = LABELS["target_macro"]
        record["n_eval"] = sum(int(row.get("n_eval", 0)) for row in entries)
        record["n_regions"] = len(entries)
        macro.append(record)
    return macro


def e1_rows(counts):
    rows = []
    for regime in E1_REGIMES:
        for seed in SEEDS:
            path = RAW_E1 / ("bench_v2_e1_" + regime + "_seed" + str(seed)) / "step_1120.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            for region in REGIONS:
                group = data["target"][region]
                n_eval = int(group.get("boundary_n_scored", counts[region]))
                total_pixels = n_eval * 128 * 128
                for threshold in THRESHOLDS:
                    values = derived_metrics(group["threshold_metrics"][threshold], total_pixels)
                    row = {
                        "experiment": "E1",
                        "regime_or_mode": regime,
                        "source_role": "intervention",
                        "buffer_m": "",
                        "seed": seed,
                        "step": 1120,
                        "region": region,
                        "region_label": LABELS[region],
                        "threshold": float(threshold),
                        "n_eval": n_eval,
                        **values,
                    }
                    if threshold == PRIMARY_THRESHOLD:
                        row.update({
                            "bf1_2": float(group.get("f1_2_mean", float("nan"))),
                            "bf1_4": float(group.get("f1_4_mean", float("nan"))),
                            "hd95": float(group.get("hd95_mean", float("nan"))),
                        })
                    else:
                        row.update({"bf1_2": float("nan"), "bf1_4": float("nan"), "hd95": float("nan")})
                    rows.append(row)
    rows.extend(add_macro(rows, ("experiment", "regime_or_mode", "source_role", "buffer_m", "seed", "step", "threshold")))
    return rows


def parse_e4_label(label):
    if label.startswith("source_"):
        return "source", "source", int(label.split("_")[1].replace("seed", ""))
    parts = label.split("_")
    return "adapted", parts[1], int(parts[-1].replace("seed", ""))


def binned_ece(calibration):
    total = float(calibration.get("pixels", 0.0))
    if total <= 0:
        return float("nan")
    value = 0.0
    for item in calibration.get("bins", []):
        n = float(item.get("pixels", 0.0))
        observed_correct = 1.0 - float(item.get("error_rate", 0.0))
        mean_confidence = float(item.get("mean_confidence", 0.0))
        value += n / total * abs(observed_correct - mean_confidence)
    return value


def e4_rows():
    rows = []
    for buffer_m in E4_BUFFERS:
        folder = RAW_E4 / ("buffer" + str(buffer_m))
        for path in sorted(folder.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            role, mode, seed = parse_e4_label(data["checkpoint_label"])
            region = data["region"]
            n_eval = int(data["n_eval"])
            total_pixels = n_eval * 128 * 128
            for threshold in THRESHOLDS:
                values = derived_metrics(data["threshold_metrics"][threshold], total_pixels)
                row = {
                    "experiment": "E4",
                    "regime_or_mode": mode,
                    "source_role": role,
                    "buffer_m": buffer_m,
                    "seed": seed,
                    "step": 400 if role == "adapted" else 0,
                    "region": region,
                    "region_label": LABELS[region],
                    "threshold": float(threshold),
                    "n_eval": n_eval,
                    **values,
                }
                boundary = data.get("boundary", {})
                row.update({
                    "bf1_2": float(boundary.get("f1_2_mean", float("nan"))),
                    "bf1_4": float(boundary.get("f1_4_mean", float("nan"))),
                    "hd95": float(boundary.get("hd95_mean", float("nan"))),
                    "high_confidence_error": float(data.get("calibration", {}).get("high_confidence_error_rate", float("nan"))),
                    "binned_ece": binned_ece(data.get("calibration", {})),
                })
                rows.append(row)
    macro_source = [row for row in rows if row["threshold"] == float(PRIMARY_THRESHOLD)]
    macro_rows = add_macro(macro_source, ("experiment", "regime_or_mode", "source_role", "buffer_m", "seed", "step", "threshold"))
    for row in macro_rows:
        row["high_confidence_error"] = float("nan")
        row["binned_ece"] = float("nan")
    rows.extend(macro_rows)
    return rows


def paired_bootstrap(deltas, endpoint, random_seed):
    rng = np.random.default_rng(random_seed)
    by_pair = {(int(row["region_index"]), int(row["seed"])): float(row[endpoint]) for row in deltas}
    regions = sorted({int(row["region_index"]) for row in deltas})
    seeds = sorted({int(row["seed"]) for row in deltas})
    boot = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        sampled_regions = rng.choice(regions, size=len(regions), replace=True)
        values = []
        for region in sampled_regions:
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            for seed in sampled_seeds:
                values.append(by_pair[(int(region), int(seed))])
        boot[i] = float(np.mean(values))
    observed = mean(row[endpoint] for row in deltas)
    return {
        "mean_delta": observed,
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "probability_positive": float(np.mean(boot > 0)),
        "probability_negative": float(np.mean(boot < 0)),
        "n_pairs": len(deltas),
    }


def seed_only_bootstrap(deltas, endpoint, random_seed):
    rng = np.random.default_rng(random_seed)
    by_seed = {int(row["seed"]): float(row[endpoint]) for row in deltas}
    boot = []
    for _ in range(N_BOOT):
        sampled = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        boot.append(float(np.mean([by_seed[int(seed)] for seed in sampled])))
    return {
        "mean_delta": mean(row[endpoint] for row in deltas),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "probability_positive": float(np.mean(np.asarray(boot) > 0)),
        "probability_negative": float(np.mean(np.asarray(boot) < 0)),
        "n_pairs": len(deltas),
    }


def index_rows(rows, keys):
    return {tuple(row[key] for key in keys): row for row in rows}


def e1_contrasts(rows):
    out = []
    idx = index_rows(rows, ("regime_or_mode", "seed", "region", "threshold"))
    metric_seed = 0
    for region in REGIONS + ["target_macro"]:
        for endpoint in ENDPOINTS:
            deltas = []
            source_regions = REGIONS if region == "target_macro" else [region]
            for source_region in source_regions:
                for seed in SEEDS:
                    a = idx[("pooled", seed, source_region, float(PRIMARY_THRESHOLD))]
                    b = idx[("cas-multistream", seed, source_region, float(PRIMARY_THRESHOLD))]
                    deltas.append({"region_index": REGIONS.index(source_region), "seed": seed, endpoint: float(a[endpoint]) - float(b[endpoint])})
            stat = seed_only_bootstrap(deltas, endpoint, RNG_SEED + metric_seed) if region != "target_macro" else paired_bootstrap(deltas, endpoint, RNG_SEED + metric_seed)
            out.append({
                "experiment": "E1",
                "contrast": "pooled-cas-multistream",
                "region": region,
                "region_label": LABELS.get(region, region),
                "threshold": float(PRIMARY_THRESHOLD),
                "endpoint": endpoint,
                "higher_is_better": HIGHER_IS_BETTER[endpoint],
                **stat,
            })
            metric_seed += 1
    return out


def e4_contrasts(rows):
    out = []
    idx = index_rows(rows, ("regime_or_mode", "source_role", "buffer_m", "seed", "region", "threshold"))
    metric_seed = 1000
    for mode in E4_MODES:
        for buffer_m in E4_BUFFERS:
            for region in REGIONS + ["target_macro"]:
                for endpoint in ENDPOINTS:
                    deltas = []
                    source_regions = REGIONS if region == "target_macro" else [region]
                    for source_region in source_regions:
                        for seed in SEEDS:
                            a = idx[(mode, "adapted", buffer_m, seed, source_region, float(PRIMARY_THRESHOLD))]
                            b = idx[("source", "source", buffer_m, seed, source_region, float(PRIMARY_THRESHOLD))]
                            deltas.append({"region_index": REGIONS.index(source_region), "seed": seed, endpoint: float(a[endpoint]) - float(b[endpoint])})
                    stat = seed_only_bootstrap(deltas, endpoint, RNG_SEED + metric_seed) if region != "target_macro" else paired_bootstrap(deltas, endpoint, RNG_SEED + metric_seed)
                    out.append({
                        "experiment": "E4",
                        "contrast": mode + "-source",
                        "region": region,
                        "region_label": LABELS.get(region, region),
                        "threshold": float(PRIMARY_THRESHOLD),
                        "mode": mode,
                        "buffer_m": buffer_m,
                        "endpoint": endpoint,
                        "higher_is_better": HIGHER_IS_BETTER[endpoint],
                        **stat,
                    })
                    metric_seed += 1
    return out


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(path)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(e1_contrast, e4_contrast):
    lines = [
        "# Route 1 Recomputed Metrics from Existing Raw JSON",
        "",
        "Protocol: route1-q2-freeze-v1-20260925",
        "",
        "No model inference or training was run. E1 and E4 endpoint tables were reconstructed from existing aggregate raw JSON. Intervals use paired cluster bootstrap over target regions and seeds at threshold 0.5.",
        "",
        "## E1 Exposure-Matched Primary Contrast",
        "",
        "Contrast: pooled source minus CAS multi-stream replay placebo.",
        "",
        "| Region | Endpoint | Delta | 95% interval | P(delta>0) |",
        "|---|---|---:|---:|---:|",
    ]
    selected = ["iou", "balanced_iou", "mcc", "precision", "recall", "bf1_2", "bf1_4", "hd95"]
    for region in ["target_macro", "hokkaido_iburi_tobu", "lombok", "palu"]:
        for endpoint in selected:
            row = next(item for item in e1_contrast if item["region"] == region and item["endpoint"] == endpoint)
            lines.append("| " + LABELS.get(region, region) + " | " + endpoint + " | " + format(row["mean_delta"], "+.4f") + " | [" + format(row["ci95_low"], "+.4f") + ", " + format(row["ci95_high"], "+.4f") + "] | " + format(row["probability_positive"], ".3f") + " |")
    lines += [
        "",
        "## E4 Few-Shot Primary Contrasts",
        "",
        "Contrast: adapted ResUNet minus its same-seed source checkpoint. Full fine-tuning is primary; decoder-only is secondary.",
        "",
        "| Mode | Buffer m | Region | IoU delta | Balanced IoU delta | MCC delta | Precision delta | Recall delta | BF1@2 delta | HD95 delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in E4_MODES:
        for buffer_m in E4_BUFFERS:
            for region in ["target_macro", "hokkaido_iburi_tobu", "lombok", "palu"]:
                lookup = {(r["mode"], r["buffer_m"], r["region"], r["endpoint"]): r for r in e4_contrast}
                values = {ep: lookup[(mode, buffer_m, region, ep)]["mean_delta"] for ep in ["iou", "balanced_iou", "mcc", "precision", "recall", "bf1_2", "hd95"]}
                lines.append("| " + mode + " | " + str(buffer_m) + " | " + LABELS.get(region, region) + " | " + format(values["iou"], "+.4f") + " | " + format(values["balanced_iou"], "+.4f") + " | " + format(values["mcc"], "+.4f") + " | " + format(values["precision"], "+.4f") + " | " + format(values["recall"], "+.4f") + " | " + format(values["bf1_2"], "+.4f") + " | " + format(values["hd95"], "+.4f") + " |")
    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- Positive IoU must be checked against balanced IoU and MCC before retaining a general segmentation-improvement claim.",
        "- HD95 and boundary deltas require separate interpretation and can disagree with pixel-overlap gains.",
        "- E4 remains a benchmark-tile result. It is not a full-map or operational validation result.",
        "",
    ]
    (OUT / "route1_metric_report_20260925.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    counts = target_counts()
    e1 = e1_rows(counts)
    e4 = e4_rows()
    write_csv(OUT / "route1_e1_metrics_long.csv", e1)
    write_csv(OUT / "route1_e4_metrics_long.csv", e4)
    e1_contrast = e1_contrasts(e1)
    e4_contrast = e4_contrasts(e4)
    write_csv(OUT / "route1_e1_contrasts_bootstrap.csv", e1_contrast)
    write_csv(OUT / "route1_e4_contrasts_bootstrap.csv", e4_contrast)
    write_report(e1_contrast, e4_contrast)
    print(json.dumps({
        "e1_rows": len(e1),
        "e4_rows": len(e4),
        "e1_contrasts": len(e1_contrast),
        "e4_contrasts": len(e4_contrast),
        "output_dir": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
