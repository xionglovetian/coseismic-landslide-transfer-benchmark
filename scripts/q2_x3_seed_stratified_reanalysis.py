"""Seed-stratified reanalysis for the fixed three-region X3 contrast.

The fixed regions are Hokkaido, Lombok and Palu. The script reports:
1. seed-level macro effects;
2. two-sided Student t intervals across the three training seeds (df = 2);
3. a fixed-region bootstrap sensitivity that resamples seeds within regions and
   then connected components and tiles within each selected seed.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist

PROJECT = Path(r"D:\landslide_unet_project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import route1_spatial_cluster_bootstrap as spatial
from q2_x3_spatial_bootstrap import REGIONS, SEEDS, build_pairs

ENDPOINTS = ["iou", "balanced_iou", "mcc", "precision", "recall"]
ARCHITECTURES = {"resunet": "ResUNet", "bottleneckliteask": "Bottleneck-LiteASK"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fixed_region_seed_bootstrap(pairs, scheme, spatial_index, rng, draws_count, label):
    maps = {key: spatial.cluster_maps(pair, spatial_index, key[0], scheme) for key, pair in pairs.items()}
    draws = []
    for _ in range(draws_count):
        strata = []
        for region in REGIONS:
            sampled_seeds = [SEEDS[i] for i in rng.choice(len(SEEDS), size=len(SEEDS), replace=True)]
            for seed in sampled_seeds:
                pair = pairs[(region, seed)]
                indices = spatial.draw_indices(pair, maps[(region, seed)], rng)
                strata.append(spatial.metric_delta(pair, indices))
        draws.append({endpoint: float(np.mean([row[endpoint] for row in strata])) for endpoint in ENDPOINTS})
    point = {endpoint: float(np.mean([spatial.point_delta(pairs[(region, seed)])[endpoint] for region in REGIONS for seed in SEEDS])) for endpoint in ENDPOINTS}
    rows = []
    for endpoint in ENDPOINTS:
        values = np.asarray([row[endpoint] for row in draws], dtype=float)
        rows.append({
            "architecture": label,
            "region": "fixed_three_region_macro",
            "scheme": scheme,
            "endpoint": endpoint,
            "mean_delta": point[endpoint],
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
            "probability_positive": float(np.mean(values > 0)),
            "n_bootstrap": draws_count,
            "seed_strata": len(SEEDS),
            "regions_fixed": len(REGIONS),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-root", default=str(PROJECT / "reports" / "q2_x3_tiles"))
    parser.add_argument("--spatial-index", default=str(PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT / "reports" / "q2_x3_seed_stratified_bootstrap"))
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    spatial_index = spatial.load_spatial_index(Path(args.spatial_index))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    seed_rows, t_rows = [], []
    all_boot = []
    for architecture_index, (slug, label) in enumerate(ARCHITECTURES.items()):
        pairs = build_pairs(Path(args.tile_root), slug)
        seed_effects = []
        for seed in SEEDS:
            effect = {endpoint: float(np.mean([spatial.point_delta(pairs[(region, seed)])[endpoint] for region in REGIONS])) for endpoint in ENDPOINTS}
            seed_effects.append(effect)
            seed_rows.append({"architecture": label, "seed": seed, "fixed_region_count": len(REGIONS), **effect})
        for endpoint in ENDPOINTS:
            values = np.asarray([row[endpoint] for row in seed_effects], dtype=float)
            mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd / np.sqrt(len(values)); critical = float(tdist.ppf(0.975, len(values) - 1))
            t_rows.append({
                "architecture": label, "region_scope": "fixed_three_regions", "endpoint": endpoint,
                "mean_delta": mean, "sd_across_seeds": sd, "se_across_seeds": se,
                "df": len(values) - 1, "t_critical_95": critical,
                "ci95_low": mean - critical * se, "ci95_high": mean + critical * se,
                "n_seeds": len(values), "seed_values": ";".join(f"{value:.10g}" for value in values),
            })
        for scheme_index, scheme in enumerate(("component", "merged30")):
            rng = np.random.default_rng(20260928 + 1000 * architecture_index + 100 * scheme_index)
            all_boot.extend(fixed_region_seed_bootstrap(pairs, scheme, spatial_index, rng, args.draws, label))

    write_csv(out / "q2_x3_seed_level_macro_effects.csv", seed_rows)
    write_csv(out / "q2_x3_seed_stratified_t_interval.csv", t_rows)
    write_csv(out / "q2_x3_seed_stratified_bootstrap.csv", all_boot)
    lines = ["# Seed-stratified t intervals for the fixed three-region X3 contrast", "", "Regions are fixed to Hokkaido, Lombok and Palu. Each seed effect is the unweighted mean of the three region-level paired effects. The 95% interval uses Student t with 2 degrees of freedom across the three training seeds.", "", "| Architecture | Endpoint | Delta | SD across seeds | 95% seed-stratified CI |", "|---|---|---:|---:|---:|"]
    for row in t_rows:
        lines.append(f"| {row['architecture']} | {row['endpoint']} | {row['mean_delta']:+.4f} | {row['sd_across_seeds']:.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |")
    (out / "summary_t_interval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"analysis": "seed-stratified fixed-region reanalysis", "regions": REGIONS, "seeds": SEEDS, "draws": args.draws}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "t_rows": len(t_rows), "bootstrap_rows": len(all_boot)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
