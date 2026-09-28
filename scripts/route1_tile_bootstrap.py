"""Tile-level bootstrap and calibration summaries for frozen Route 1.

Reads the eval-only outputs produced by route1_evalonly_tile_calibration.py.
No model inference or training is performed here.
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
EVAL = REPORTS / "route1_evalonly"
OUT = REPORTS / "route1_bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
LABELS = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu", "target_macro": "Target Macro"}
SEEDS = [42, 2026, 777]
MODES = ["full", "decoder-only"]
BUFFERS = [0, 256, 512]
RNG_SEED = 20260925
N_BOOT = 10000
THRESHOLD = 0.5


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number

def mean(values):
    return float(statistics.fmean(values)) if values else float("nan")


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(path)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(counts, boundary=None):
    tp, fp, fn, tn = [float(x) for x in counts]
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    foreground_iou = tp / (tp + fp + fn + eps)
    background_iou = tn / (tn + fp + fn + eps)
    mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    values = {
        "iou": foreground_iou,
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "balanced_iou": 0.5 * (foreground_iou + background_iou),
        "background_iou": background_iou,
    }
    if boundary:
        for key in ("bf1_2", "bf1_4", "hd95"):
            vals = [float(x) for x in boundary.get(key, []) if str(x) != "" and not math.isnan(float(x))]
            values[key] = mean(vals)
    return values


def load_tiles():
    records = {}
    for path in sorted(EVAL.glob("*__tiles.csv")):
        for row in read_csv(path):
            if float(row["threshold"]) != THRESHOLD:
                continue
            if finite_float(row.get("f1_2")) is None or finite_float(row.get("hd95")) is None:
                continue
            key = (
                row["experiment"], row["run_id"], str(row["buffer_m"]), row["region"],
                int(row["seed"]), row.get("mode", ""), row.get("role", ""), row["id"],
            )
            existing = records.get(key)
            if existing is None:
                records[key] = row
    return records


def prefer_row(bucket, identifier, row):
    existing = bucket.get(identifier)
    row_ok = finite_float(row.get("f1_2")) is not None and finite_float(row.get("hd95")) is not None
    existing_ok = bool(existing) and finite_float(existing.get("f1_2")) is not None and finite_float(existing.get("hd95")) is not None
    if existing is None or (row_ok and not existing_ok):
        bucket[identifier] = row


def boundary_array(bucket, ids, field):
    values = []
    for identifier in ids:
        value = bucket[identifier].get(field)
        if value is None or value == "" or str(value).lower() == "nan":
            values.append(float("nan"))
        else:
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def collect_pair_rows(tiles, experiment, left, right, mode_left="", mode_right="", buffer_m=None):
    """Return paired tile arrays keyed by region and seed."""
    pairs = {}

    def store_bucket(bucket, identifier, row):
        if identifier not in bucket:
            bucket[identifier] = {
                "counts": np.asarray([int(row[k]) for k in ("tp", "fp", "fn", "tn")], dtype=np.float64),
                "bf1_2": float("nan"),
                "bf1_4": float("nan"),
                "hd95": float("nan"),
            }
        for field in ("bf1_2", "bf1_4", "hd95"):
            value = row.get(field)
            if value is None or value == "" or str(value).lower() == "nan":
                continue
            try:
                bucket[identifier][field] = float(value)
            except ValueError:
                pass

    for region in REGIONS:
        for seed in SEEDS:
            a = {}
            b = {}
            for key, row in tiles.items():
                exp, run_id, btxt, reg, s, mode, role, identifier = key
                if exp != experiment or reg != region or s != seed:
                    continue
                if buffer_m is not None:
                    canonical = str(buffer_m)
                    if buffer_m == 512 and btxt == "256":
                        canonical = "256"
                    if btxt != canonical:
                        continue
                if experiment == "E1":
                    if row.get("regime") == left:
                        store_bucket(a, identifier, row)
                    if row.get("regime") == right:
                        store_bucket(b, identifier, row)
                else:
                    if role == left and (not mode_left or mode == mode_left):
                        store_bucket(a, identifier, row)
                    if role == right and (not mode_right or mode == mode_right):
                        store_bucket(b, identifier, row)
            ids = sorted(set(a) & set(b))
            if not ids:
                continue
            pairs[(region, seed)] = {
                "ids": ids,
                "a_counts": np.asarray([a[i]["counts"] for i in ids], dtype=np.float64),
                "b_counts": np.asarray([b[i]["counts"] for i in ids], dtype=np.float64),
                "a_boundary": {field: np.asarray([a[i][field] for i in ids], dtype=np.float64) for field in ("bf1_2", "bf1_4", "hd95")},
                "b_boundary": {field: np.asarray([b[i][field] for i in ids], dtype=np.float64) for field in ("bf1_2", "bf1_4", "hd95")},
            }
    return pairs

def augment_boundary_pairs(pairs, tiles, experiment, left, right, mode_left="", mode_right="", buffer_m=None):
    fields = ("bf1_2", "bf1_4", "hd95")
    for (region, seed), pair in pairs.items():
        a_values = {field: {} for field in fields}
        b_values = {field: {} for field in fields}
        for key, row in tiles.items():
            exp, run_id, btxt, reg, s, mode, role, identifier = key
            if exp != experiment or reg != region or s != seed:
                continue
            if buffer_m is not None:
                canonical = str(buffer_m)
                if buffer_m == 512 and btxt == "256":
                    canonical = "256"
                if btxt != canonical:
                    continue
            is_a = (row.get("regime") == left) if experiment == "E1" else (role == left and (not mode_left or mode == mode_left))
            is_b = (row.get("regime") == right) if experiment == "E1" else (role == right and (not mode_right or mode == mode_right))
            if not is_a and not is_b:
                continue
            destination = a_values if is_a else b_values
            for field in fields:
                value = row.get(field)
                if value is None or value == "" or str(value).lower() == "nan":
                    continue
                try:
                    destination[field][identifier] = float(value)
                except ValueError:
                    pass
        ids = pair["ids"]
        pair["a_boundary"] = {field: np.asarray([a_values[field].get(identifier, float("nan")) for identifier in ids], dtype=np.float64) for field in fields}
        pair["b_boundary"] = {field: np.asarray([b_values[field].get(identifier, float("nan")) for identifier in ids], dtype=np.float64) for field in fields}


def draw_metric(pair, indices):
    a_counts = pair["a_counts"][indices].sum(axis=0)
    b_counts = pair["b_counts"][indices].sum(axis=0)
    a_bound = {}
    b_bound = {}
    for key in ("bf1_2", "bf1_4", "hd95"):
        av = pair["a_boundary"][key][indices]
        bv = pair["b_boundary"][key][indices]
        a_bound[key] = av[~np.isnan(av)]
        b_bound[key] = bv[~np.isnan(bv)]
    ma = metrics(a_counts, a_bound)
    mb = metrics(b_counts, b_bound)
    return {key: ma[key] - mb[key] for key in ma}


def bootstrap_contrast(pairs, region, rng):
    endpoints = ["iou", "balanced_iou", "mcc", "precision", "recall", "f1", "dice", "accuracy", "hd95"]
    if region == "target_macro":
        draws = []
        for _ in range(N_BOOT):
            sampled_regions = rng.choice(REGIONS, size=len(REGIONS), replace=True)
            per_stratum = []
            for reg in sampled_regions:
                sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
                for seed in sampled_seeds:
                    pair = pairs[(str(reg), int(seed))]
                    idx = rng.integers(0, len(pair["ids"]), size=len(pair["ids"]))
                    per_stratum.append(draw_metric(pair, idx))
            draws.append({key: mean(item[key] for item in per_stratum) for key in endpoints})
    else:
        draws = []
        for _ in range(N_BOOT):
            sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
            per_stratum = []
            for seed in sampled_seeds:
                pair = pairs[(str(region), int(seed))]
                idx = rng.integers(0, len(pair["ids"]), size=len(pair["ids"]))
                per_stratum.append(draw_metric(pair, idx))
            draws.append({key: mean(item[key] for item in per_stratum) for key in endpoints})
    point = {}
    if region == "target_macro":
        per_stratum = []
        for reg in REGIONS:
            for seed in SEEDS:
                pair = pairs[(reg, seed)]
                per_stratum.append(draw_metric(pair, np.arange(len(pair["ids"]))))
        point = {key: mean(item[key] for item in per_stratum) for key in endpoints}
    else:
        point = {key: mean(draw_metric(pairs[(region, seed)], np.arange(len(pairs[(region, seed)]["ids"])))[key] for seed in SEEDS) for key in endpoints}
    rows = []
    for endpoint in endpoints:
        vals = np.asarray([item[endpoint] for item in draws], dtype=float)
        rows.append({
            "region": region,
            "region_label": LABELS.get(region, region),
            "endpoint": endpoint,
            "mean_delta": point[endpoint],
            "ci95_low": float(np.percentile(vals, 2.5)),
            "ci95_high": float(np.percentile(vals, 97.5)),
            "probability_positive": float(np.mean(vals > 0)),
            "probability_negative": float(np.mean(vals < 0)),
            "n_bootstrap": N_BOOT,
        })
    return rows


def load_calibration_summaries():
    candidates = []
    for path in sorted(EVAL.glob("*__calibration.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        candidates.append((path, data))
    selected = {}
    for path, data in candidates:
        key = (data.get("experiment"), data.get("run_id"), str(data.get("buffer_m")))
        current = selected.get(key)
        if current is None or int(data.get("n_eval", 10**12)) < int(current[1].get("n_eval", 10**12)):
            selected[key] = (path, data)
    return selected


def calibration_summary_rows(selected):
    grouped = defaultdict(list)
    for path, data in selected.values():
        experiment = data.get("experiment")
        if experiment == "E1":
            group = ("E1", data.get("regime", ""), "", "", "")
        else:
            group = ("E4", "", data.get("mode", ""), data.get("role", ""), str(data.get("buffer_m", "")))
        grouped[group].append(data)
    rows = []
    for group, items in sorted(grouped.items()):
        experiment, regime, mode, role, buffer_m = group
        rows.append({
            "experiment": experiment,
            "regime": regime,
            "mode": mode,
            "role": role,
            "buffer_m": buffer_m,
            "n_runs": len(items),
            "n_eval_mean": mean(float(item.get("n_eval", 0.0)) for item in items),
            "pixels_mean": mean(float(item.get("pixels", 0.0)) for item in items),
            "brier_mean": mean(float(item.get("brier", float("nan"))) for item in items),
            "ece_mean": mean(float(item.get("ece", float("nan"))) for item in items),
            "high_confidence_error_rate_mean": mean(float(item.get("high_confidence_error_rate", float("nan"))) for item in items),
        })
    return rows


def risk_coverage_rows(selected):
    grouped = defaultdict(list)
    for path, data in selected.values():
        risk_path = Path(str(path).replace("__calibration.json", "__risk_coverage.csv"))
        if not risk_path.exists():
            continue
        if data.get("experiment") == "E1":
            group = ("E1", data.get("regime", ""), "", "", "")
        else:
            group = ("E4", "", data.get("mode", ""), data.get("role", ""), str(data.get("buffer_m", "")))
        for row in read_csv(risk_path):
            grouped[(group, float(row["confidence_threshold"]))].append(row)
    out = []
    for (group, threshold), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        experiment, regime, mode, role, buffer_m = group
        out.append({
            "experiment": experiment,
            "regime": regime,
            "mode": mode,
            "role": role,
            "buffer_m": buffer_m,
            "confidence_threshold": threshold,
            "coverage_mean": mean(float(row["coverage"]) for row in rows),
            "selective_risk_mean": mean(float(row["selective_risk"]) for row in rows),
            "n_runs": len(rows),
        })
    return out


def find_row(rows, **criteria):
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def write_report(e1_rows, e4_rows, calibration_rows):
    lines = [
        "# Route 1 Tile-Level Bootstrap and Calibration",
        "",
        "Protocol: route1-q2-freeze-v1-20260925",
        "",
        "No additional model training was performed. Model forward passes used frozen checkpoints only. Intervals use 10,000 bootstrap draws over target region, seed, and tiles within each region-seed stratum.",
        "",
        "## E1 Exposure-Matched Pair",
        "",
        "| Region | IoU | Balanced IoU | MCC | Precision | Recall | HD95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in ["target_macro", "hokkaido_iburi_tobu", "lombok", "palu"]:
        values = []
        for endpoint in ["iou", "balanced_iou", "mcc", "precision", "recall", "hd95"]:
            row = find_row(e1_rows, region=region, endpoint=endpoint)
            values.append(format(row["mean_delta"], "+.4f") + " [" + format(row["ci95_low"], "+.4f") + ", " + format(row["ci95_high"], "+.4f") + "]")
        lines.append("| " + LABELS[region] + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## E4 Few-Shot Tile Pair",
        "",
        "Full fine-tuning is primary. Values are adapted minus source on identical tiles and query sets.",
        "",
        "| Mode | Buffer m | Region | IoU | Balanced IoU | MCC | Precision | Recall | HD95 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for buffer_m in BUFFERS:
            for region in ["target_macro", "hokkaido_iburi_tobu", "lombok", "palu"]:
                values = []
                for endpoint in ["iou", "balanced_iou", "mcc", "precision", "recall", "hd95"]:
                    row = find_row(e4_rows, mode=mode, buffer_m=buffer_m, region=region, endpoint=endpoint)
                    values.append(format(row["mean_delta"], "+.4f") + " [" + format(row["ci95_low"], "+.4f") + ", " + format(row["ci95_high"], "+.4f") + "]")
                lines.append("| " + mode + " | " + str(buffer_m) + " | " + LABELS[region] + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Calibration",
        "",
        "| Experiment | Regime | Mode | Role | Buffer m | Runs | Brier | ECE | High-confidence error |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in calibration_rows:
        lines.append("| " + row["experiment"] + " | " + row["regime"] + " | " + row["mode"] + " | " + row["role"] + " | " + row["buffer_m"] + " | " + str(row["n_runs"]) + " | " + format(row["brier_mean"], ".4f") + " | " + format(row["ece_mean"], ".4f") + " | " + format(row["high_confidence_error_rate_mean"], ".4f") + " |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- E1 co-primary support is mixed: IoU and MCC intervals should be interpreted separately from balanced IoU.",
        "- E4 gains are retained as precision-led benchmark improvements only if recall and boundary/HD95 trade-offs are reported explicitly.",
        "- ECE and Brier are calibration summaries, not deployment-risk estimates. Risk-coverage curves are saved separately.",
        "",
    ]
    (OUT / "route1_tile_bootstrap_report_20260925.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    tiles = load_tiles()
    e1_pairs = collect_pair_rows(tiles, "E1", "pooled", "cas-multistream")
    augment_boundary_pairs(e1_pairs, tiles, "E1", "pooled", "cas-multistream")
    e1_rows = []
    for index, region in enumerate(["target_macro"] + REGIONS):
        rows = bootstrap_contrast(e1_pairs, region, np.random.default_rng(RNG_SEED + index))
        for row in rows:
            row["experiment"] = "E1"
            row["contrast"] = "pooled-cas-multistream"
            row["mode"] = ""
            row["buffer_m"] = "na"
        e1_rows.extend(rows)
    e4_rows = []
    offset = 100
    for mode in MODES:
        for buffer_m in BUFFERS:
            pairs = collect_pair_rows(tiles, "E4", "adapted", "source", mode_left=mode, mode_right="source", buffer_m=buffer_m)
            augment_boundary_pairs(pairs, tiles, "E4", "adapted", "source", mode_left=mode, mode_right="source", buffer_m=buffer_m)
            for index, region in enumerate(["target_macro"] + REGIONS):
                rows = bootstrap_contrast(pairs, region, np.random.default_rng(RNG_SEED + offset + index))
                for row in rows:
                    row["experiment"] = "E4"
                    row["contrast"] = mode + "-source"
                    row["mode"] = mode
                    row["buffer_m"] = buffer_m
                e4_rows.extend(rows)
            offset += 20
    selected = load_calibration_summaries()
    calibration_rows = calibration_summary_rows(selected)
    risk_rows = risk_coverage_rows(selected)
    write_csv(OUT / "route1_e1_tile_bootstrap.csv", e1_rows)
    write_csv(OUT / "route1_e4_tile_bootstrap.csv", e4_rows)
    write_csv(OUT / "route1_calibration_summary.csv", calibration_rows)
    write_csv(OUT / "route1_risk_coverage_summary.csv", risk_rows)
    write_report(e1_rows, e4_rows, calibration_rows)
    print(json.dumps({
        "e1_bootstrap_rows": len(e1_rows),
        "e4_bootstrap_rows": len(e4_rows),
        "calibration_rows": len(calibration_rows),
        "risk_coverage_rows": len(risk_rows),
        "output_dir": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
