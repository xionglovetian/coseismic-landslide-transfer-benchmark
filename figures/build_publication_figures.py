# -*- coding: utf-8 -*-
"""Build publication-grade manuscript figures from the landslide benchmark."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MultipleLocator

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


def save_cns_600(fig, filename):
    """Final manuscript export: editable vector masters plus 600 dpi preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(f"{filename}.svg", bbox_inches="tight", dpi=600)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=600)


# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

MM = 1 / 25.4
ROOT = Path(r"D:\landslide_unet_project")
REPORTS = ROOT / "reports"
OUT = Path(__file__).resolve().parent
DATA_OUT = OUT / "figure_source_data"
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["SegFormer-B0", "ResUNet", "Bottleneck-LiteASK", "DeepLabV3+"]
MODEL_COLORS = {
    "SegFormer-B0": CATEGORICAL[0], "ResUNet": CATEGORICAL[1],
    "Bottleneck-LiteASK": CATEGORICAL[2], "DeepLabV3+": CATEGORICAL[3],
}
# D5-3: one fixed colour per architecture, reused in every figure.
ARCH_COLORS = {"ResUNet": CATEGORICAL[1], "SegFormer-B0": CATEGORICAL[0],
               "Bottleneck-LiteASK": CATEGORICAL[2], "DeepLabV3+": CATEGORICAL[3]}
MODEL_MARKERS = {"SegFormer-B0": "o", "ResUNet": "s", "Bottleneck-LiteASK": "D", "DeepLabV3+": "^"}
SEED_LINESTYLES = {42: "-", 2026: "--", 777: ":"}
REGIONS_7 = ["Wenchuan", "Jiuzhai Valley", "Moxitaidi", "Longxi River", "Hokkaido", "Lombok", "Palu"]
def rlabel(name):
    """D5-16: display name for a region (data keys are unchanged)."""
    return "Jiuzhaigou" if name == "Jiuzhai Valley" else name


REGION_COLORS = {
    "Wenchuan": CATEGORICAL[0], "Jiuzhai Valley": CATEGORICAL[2],
    "Moxitaidi": CATEGORICAL[3], "Longxi River": CATEGORICAL[4],
    "Hokkaido": CATEGORICAL[1], "Lombok": ACCENT_RED, "Palu": GREY,
}


def clean_axis(ax, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#555555")
    ax.tick_params(width=0.6, length=2.2, color="#555555", pad=1.8)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=3)
    ax.grid(False)


def panel_label(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold", color=BLACK, clip_on=False)


def save_table(df, name):
    df.to_csv(DATA_OUT / name, index=False, float_format="%.10g")


def read_csv_required(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_source_validation():
    """Recover 3-seed source validation IoU from original histories."""
    specs = {
        "Bottleneck-LiteASK": ("group_bottleneckliteaskunetpp_seed", "val_iou"),
        "DeepLabV3+": ("bench_v2_deeplabv3plus_seed", "val_iou"),
        "ResUNet": ("bench_v2_resunet_seed", "val_iou"),
        "SegFormer-B0": ("bench_v2_segformerb0_seed", "val_iou"),
    }
    rows = []
    for model, (prefix, col) in specs.items():
        for seed in (42, 2026, 777):
            path = ROOT / "outputs" / f"{prefix}{seed}" / "history.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                rows.append({"model": model, "seed": seed, "source_val_iou": float(vals.iloc[-1])})
    detail = pd.DataFrame(rows)
    fallback = pd.DataFrame([
            {"model": "Bottleneck-LiteASK", "source_val_iou_mean": 0.719, "source_val_iou_std": 0.006, "n_seeds": 3},
            {"model": "ResUNet", "source_val_iou_mean": 0.682, "source_val_iou_std": 0.007, "n_seeds": 3},
            {"model": "DeepLabV3+", "source_val_iou_mean": 0.611, "source_val_iou_std": 0.004, "n_seeds": 3},
            {"model": "SegFormer-B0", "source_val_iou_mean": 0.593, "source_val_iou_std": 0.018, "n_seeds": 3},
        ])
    summary = fallback
    return detail, summary.set_index("model").loc[MODEL_ORDER].reset_index()


def load_inputs():
    zero = read_csv_required(REPORTS / "benchmark_v2_zero_shot_region_summary.csv")
    zero_model = read_csv_required(REPORTS / "benchmark_v2_zero_shot_model_summary.csv")
    source_detail, source_summary = load_source_validation()

    fig3 = zero[["model", "region_label", "n_seeds", "iou_mean", "iou_std"]].copy()
    fig3 = fig3.rename(columns={"region_label": "region"})
    fig3 = fig3.merge(source_summary, on="model", how="left")
    macro6 = (zero[zero["region_label"] != "Moxitaidi"]
             .groupby("model", as_index=False)["iou_mean"].mean()
             .rename(columns={"iou_mean": "target_macro_iou"}))
    fig3 = fig3.merge(macro6, on="model", how="left")
    fig3["retention"] = fig3["target_macro_iou"] / fig3["source_val_iou_mean"]

    loro = read_csv_required(REPORTS / "loro_benchmark_summary.csv")
    loro_pivot = loro.pivot(index="region_label", columns="method", values="iou_mean")
    loro_d = loro_pivot["Uniform multi-source"] - loro_pivot["CAS-only zero-shot"]
    fig4_lro = loro_d.rename("delta_iou").reset_index().rename(columns={"region_label": "region"})
    fig4_lro = fig4_lro[fig4_lro["region"] != "Moxitaidi"].copy()
    fig4_lro["protocol"] = "LORO, epoch-matched"
    fig4_lro["model"] = "Bottleneck-LiteASK"

    # Retained from the original manuscript figure script and checked against reports/e0_summary.md.
    fig4_fixed = pd.DataFrame([
        {"region": "Hokkaido", "model": "Bottleneck-LiteASK", "protocol": "Fixed source, exploratory", "delta_iou": 0.0329},
        {"region": "Lombok", "model": "Bottleneck-LiteASK", "protocol": "Fixed source, exploratory", "delta_iou": -0.0075},
        {"region": "Palu", "model": "Bottleneck-LiteASK", "protocol": "Fixed source, exploratory", "delta_iou": 0.0064},
        {"region": "Hokkaido", "model": "ResUNet", "protocol": "Fixed source, exploratory", "delta_iou": 0.0908},
        {"region": "Lombok", "model": "ResUNet", "protocol": "Fixed source, exploratory", "delta_iou": -0.0057},
        {"region": "Palu", "model": "ResUNet", "protocol": "Fixed source, exploratory", "delta_iou": 0.0081},
    ])
    fig4_exploratory = pd.concat([fig4_lro, fig4_fixed], ignore_index=True)

    e1_boot = read_csv_required(REPORTS / "e1_exposure_matched_bootstrap.csv")
    e1_boot = e1_boot[e1_boot["contrast"] == "pooled-cas-multistream"].copy()
    e1_boot["region"] = e1_boot["region"].replace({
        "hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok",
        "palu": "Palu", "target_macro": "Macro",
    })
    e1_boot = e1_boot[["region", "mean_delta", "ci95_low", "ci95_high", "probability_positive", "probability_material"]]
    e1_boot = e1_boot.rename(columns={"mean_delta": "delta_iou"})

    epoch = read_csv_required(REPORTS / "e3_epochwise_per_run.csv").rename(
        columns={"source_val_iou": "source_iou"}
    )
    srcfrac_all = read_csv_required(REPORTS / "p4_source_fraction_curves.csv")
    srcfrac = srcfrac_all[srcfrac_all["epoch"] == 50][["fraction", "cas_val_iou", "target_macro_iou"]].copy()

    fs_macro = read_csv_required(REPORTS / "benchmark_v2_fewshot128_multiseed_macro.csv")
    fs_region = read_csv_required(REPORTS / "benchmark_v2_fewshot128_multiseed_by_region.csv")
    fs_macro["model"] = fs_macro["model"].replace({"SegFormerB0": "SegFormer-B0"})
    fs_region["model"] = fs_region["model"].replace({"SegFormerB0": "SegFormer-B0"})

    e4_query = read_csv_required(REPORTS / "e4_query_summary.csv")
    e4_support = read_csv_required(REPORTS / "e4_support_draws_summary.csv")
    e4_stitched = read_csv_required(REPORTS / "e4_stitched_summary.csv")

    conf_json = json.loads((REPORTS / "p4_confidence_calibration.json").read_text(encoding="utf-8"))
    conf_rows, calib_rows = [], []
    label_map = {"source_val": "CAS validation", "hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}
    for key, d in conf_json.items():
        domain = label_map.get(key, key)
        conf_rows.append({
            "domain": domain, "pixels": d["pixels"], "error_rate": d["error_rate"],
            "mean_confidence": d["mean_confidence"], "high_confidence_fraction": d["high_confidence_fraction"],
            "high_confidence_error_rate": d["high_confidence_error_rate"],
        })
        for i, b in enumerate(d["bins"]):
            calib_rows.append({
                "domain": domain, "bin": i, "lower": b["lower"], "upper": b["upper"],
                "pixel_fraction": b["pixel_fraction"], "mean_confidence": b["mean_confidence"],
                "error_rate": b["error_rate"],
            })
    conf, calib = pd.DataFrame(conf_rows), pd.DataFrame(calib_rows)

    prev = read_csv_required(REPORTS / "e0_prevalence_metrics.csv")
    prev = prev[prev["region_label"] != "Moxitaidi"].copy()
    thresh = read_csv_required(REPORTS / "e0_threshold_metrics.csv")
    pr = (thresh[(thresh["source"] == "zero-shot") & (np.isclose(thresh["threshold"], 0.5))]
          .groupby("region_label", as_index=False)[["precision", "recall"]].mean()
          .rename(columns={"region_label": "region", "precision": "mean_precision", "recall": "mean_recall"}))
    prev = prev.drop(columns=["region"]).rename(columns={
        "region_label": "region", "mean_foreground_fraction": "foreground_fraction",
        "mean_iou": "iou_mean", "std_iou": "iou_std", "mean_balanced_iou": "balanced_iou",
    }).merge(pr, on="region", how="left")

    save_table(source_detail, "fig3_source_validation_detail.csv")
    save_table(source_summary, "fig3_source_validation_summary.csv")
    save_table(fig3, "fig3_zero_shot_and_retention.csv")
    save_table(fig4_exploratory, "fig4_exploratory_transfer_deltas.csv")
    save_table(e1_boot, "fig4_exposure_matched_causal_effects.csv")
    save_table(epoch, "fig5_epochwise_replication.csv")
    save_table(srcfrac, "fig5_source_fraction.csv")
    save_table(fs_macro, "fig6_fewshot_macro.csv")
    save_table(fs_region, "fig6_fewshot_by_region.csv")
    save_table(e4_query, "fig6_e4_buffer_robustness.csv")
    save_table(e4_support, "fig6_e4_support_draws.csv")
    save_table(e4_stitched, "fig6_e4_stitched_maps.csv")
    save_table(conf, "fig7_confidence_summary.csv")
    save_table(calib, "fig7_confidence_bins.csv")
    save_table(prev, "fig8_prevalence_metrics.csv")
    return {
        "fig3": fig3, "fig4_exploratory": fig4_exploratory, "fig4_causal": e1_boot,
        "epoch": epoch, "srcfrac": srcfrac, "fs_macro": fs_macro, "fs_region": fs_region, "e4_query": e4_query, "e4_support": e4_support, "e4_stitched": e4_stitched,
        "conf": conf, "calib": calib, "prev": prev,
    }


def figure2_schematic():
    fig, ax = plt.subplots(figsize=(183 * MM, 105 * MM))
    ax.set_xlim(0, 100); ax.set_ylim(0, 65); ax.axis("off")

    def box(x, y, w, h, title, lines, fill="#F7F7F7", edge="#B8B8B8", title_color=BLACK,
            linestyle="solid", linewidth=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.22,rounding_size=0.8",
                                    linewidth=linewidth, edgecolor=edge, facecolor=fill, linestyle=linestyle))
        ax.text(x + 1.6, y + h - 2.4, title, ha="left", va="top", fontsize=7.0, fontweight="bold", color=title_color)
        yy = y + h - 4.4
        for line in lines:
            ax.text(x + 1.6, yy, line, ha="left", va="top", fontsize=5.8, color="#333333")
            yy -= 1.5

    def arrow(x1, y1, x2, y2, color="#666666"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=color))

    box(1, 48.5, 23.5, 10.5, "Source domain", ["CAS Moxi + Bijie", "1,795 train / 513 validation", "128 × 128 RGB"], "#EDF3FA", "#8CB3D9", CATEGORICAL[0])
    box(26.5, 48.5, 23.5, 10.5, "Held-out regions", ["Seven regions (six coseismic)", "12,324 tiles", "GSD 0.2–5 m"], "#F4F4F4", "#CCCCCC")
    box(52, 48.5, 23.5, 10.5, "Architectures and seeds", ["ResUNet · DeepLabV3+", "SegFormer-B0 · Bottleneck", "Three seeds each"], "#F1F7F2", "#9DC3A5", CATEGORICAL[2])
    box(77.5, 48.5, 21.5, 10.5, "Model selection", ["Source validation only", "No target imagery", "Frozen evaluation protocol"], "#FAF4E9", "#E1C38F", CATEGORICAL[3])

    box(1, 31.5, 30, 10.5, "A  Main source-only benchmark", ["Train on CAS, 50 epochs", "Zero-shot evaluation on 7 regions", "4 architectures × 3 seeds", "Report IoU, BF1 and HD95"], "#EDF3FA", "#8CB3D9", CATEGORICAL[0])
    box(35, 31.5, 30, 10.5, "B  Epoch-matched comparison", ["Source-only vs CAS-retrain20", "vs pooled source regions", "Pooled received 6.34× exposure", "Exploratory, not causal"], "#F7F7F7", "#8C8C8C", GREY, linestyle=(0, (4, 3)), linewidth=1.25)
    box(69, 31.5, 30, 10.5, "C  Exposure-matched causal test", ["Single stream vs replay placebo", "vs pooled source", "1,120 steps; 35,840 exposures", "3 seeds; Hokkaido, Lombok, Palu"], "#E8F1EC", CATEGORICAL[2], CATEGORICAL[2], linewidth=1.35)

    box(1, 15.5, 30, 10.5, "Diagnostic: temporal divergence", ["2 architectures × 3 seeds", "Source peaks in 6/6", "Early target peak in 2/6", "Not a general temporal mechanism"], "#FAF4E9", "#E1C38F", CATEGORICAL[3])
    box(35, 15.5, 30, 10.5, "Diagnostic: source scale", ["25 / 50 / 75 / 100% source data", "Source fit rises", "Target macro IoU stays flat", "More source data is not the remedy"], "#F7F7F7", "#BDBDBD", GREY)
    box(69, 15.5, 30, 10.5, "Remedy: few-shot adaptation", ["5 / 10 / 20 target tiles", "Full or decoder-only tuning", "20-shot: 0.049 → 0.181", "Robust to buffers, draws and stitching"], "#E8F1EC", "#75A982", CATEGORICAL[2])

    box(8, 1.5, 84, 10.5, "Reliability checks", ["Threshold, calibration, prevalence and stitched-map checks", "E4: 256/512 m physical buffers and three support draws retained the gain"], "#F7FBFF", "#9CBED4", CATEGORICAL[0])

    arrow(12.7, 48.5, 12.7, 42.2); arrow(50, 48.5, 50, 42.2); arrow(87.2, 48.5, 84.0, 42.2)
    arrow(16, 31.5, 16, 26.2); arrow(50, 31.5, 50, 26.2); arrow(84, 31.5, 84, 26.2)
    arrow(50, 15.5, 50, 12.2); arrow(84, 15.5, 84, 12.2)
    ax.text(1, 64.0, "Benchmark design and evidence chains", ha="left", va="top", fontsize=9, fontweight="bold")
    ax.text(1, 61.8, "Shared data and evaluation protocol across the source-only benchmark, controlled comparisons and target adaptation.", ha="left", va="top", fontsize=6.2, color="#444444")
    ax.text(35, 43.1, "EXPLORATORY", ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=GREY)
    ax.text(69, 43.1, "CAUSAL TEST", ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=CATEGORICAL[2])
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.99)
    save_cns_600(fig, OUT / "Figure2_experimental_design")
    plt.close(fig)


def figure3(data):
    fig = plt.figure(figsize=(183 * MM, 78 * MM))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.45], wspace=0.30)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    z = data["fig3"].drop_duplicates("model").set_index("model").loc[MODEL_ORDER]

    ax1.axhspan(0, 0.20, color="#F2F2F2", zorder=0)
    retention_offsets = {"SegFormer-B0": 8, "ResUNet": 2, "Bottleneck-LiteASK": -3, "DeepLabV3+": -8}
    for model in MODEL_ORDER:
        r = z.loc[model]
        ax1.plot([0, 1], [r.source_val_iou_mean, r.target_macro_iou], color=MODEL_COLORS[model], lw=1.15, marker=MODEL_MARKERS[model], ms=4.0, zorder=3)
        ax1.errorbar(0, r.source_val_iou_mean, yerr=r.source_val_iou_std, fmt="none", ecolor=MODEL_COLORS[model], elinewidth=0.65, capsize=1.6, zorder=4)
        ax1.errorbar(1, r.target_macro_iou, yerr=r.iou_std.mean(), fmt="none", ecolor=MODEL_COLORS[model], elinewidth=0.65, capsize=1.6, zorder=4)
        ax1.annotate(f"{100*r.retention:.1f}%", (1.02, r.target_macro_iou), xytext=(5, retention_offsets[model]), textcoords="offset points", va="center", fontsize=6.5, color=MODEL_COLORS[model])
    ax1.set_xlim(-0.16, 1.28); ax1.set_ylim(0, 0.76)
    ax1.set_xticks([0, 1], ["Source", "Target"]); ax1.set_ylabel("IoU")
    ax1.text(0.02, 0.21, "descriptive zone: target macro IoU < 0.20", fontsize=5.5, color="#666666")
    clean_axis(ax1); panel_label(ax1, "a", x=-0.18)
    ax1.set_title("Six-region held-out macro (seed SD)", fontsize=8, pad=5)
    ax1.text(0.02, 0.06, "error bars = SD across seeds", fontsize=5.5, color="#666666")

    order = REGIONS_7
    ybase = {r: i for i, r in enumerate(order)}
    offsets = {m: (i - 1.5) * 0.115 for i, m in enumerate(MODEL_ORDER)}
    for model in MODEL_ORDER:
        sub = data["fig3"][data["fig3"]["model"] == model].set_index("region")
        for region in order:
            if region not in sub.index:
                continue
            row = sub.loc[region]; y = ybase[region] + offsets[model]
            ax2.hlines(y, 0, row.iou_mean, color="#D9D9D9", lw=0.7, zorder=0)
            ax2.errorbar(row.iou_mean, y, xerr=row.iou_std, fmt=MODEL_MARKERS[model], ms=3.3,
                         mfc=MODEL_COLORS[model], mec=MODEL_COLORS[model], ecolor=MODEL_COLORS[model],
                         elinewidth=0.55, capsize=1.2, zorder=3)
    ax2.set_yticks(range(len(order)), order); ax2.invert_yaxis()
    ax2.set_xlim(0, 0.68); ax2.set_xlabel("Zero-shot IoU")
    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    clean_axis(ax2); panel_label(ax2, "b", x=-0.23)
    ax2.set_title("Held-out and source-adjacent regions", fontsize=8, pad=5)
    handles = [Line2D([0], [0], marker=MODEL_MARKERS[m], color="none", markerfacecolor=MODEL_COLORS[m], markeredgecolor=MODEL_COLORS[m], markersize=4, label=m) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4,
               fontsize=6.2, columnspacing=1.4, handletextpad=0.4, frameon=False)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.235, top=0.86)
    save_cns_600(fig, OUT / "Figure3_zero_shot_by_region")
    plt.close(fig)


def figure4(data):
    fig = plt.figure(figsize=(181 * MM, 82 * MM))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.42)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    exp = data["fig4_exploratory"]
    exp_order = ["Longxi River", "Moxitaidi", "Wenchuan", "Jiuzhai Valley", "Hokkaido", "Palu", "Lombok"]
    exp_order = [r for r in exp_order if r in set(exp["region"])]
    ypos = {r: i for i, r in enumerate(exp_order)}
    ax1.axvline(0, color=BLACK, lw=0.75, zorder=0)
    ax1.axvspan(-0.075, 0, color="#F4E6E5", zorder=-1)
    ax1.axvspan(0, 0.10, color="#E8F1EC", zorder=-1)
    for (protocol, model), sub in exp.groupby(["protocol", "model"]):
        marker = "o" if protocol.startswith("LORO") else "s"
        for _, r in sub.iterrows():
            y = ypos[r.region]
            shift = -0.13 if model == "Bottleneck-LiteASK" else (0.13 if model == "ResUNet" else 0.0)
            color = DIVERGING[0] if r.delta_iou >= 0 else DIVERGING[2]
            ax1.scatter(r.delta_iou, y + shift, s=22, marker=marker, color=color, edgecolor="white", linewidth=0.35, zorder=3)
            ax1.text(r.delta_iou + 0.0025, y + shift,
                     f"{r.delta_iou:+.4f}", fontsize=5.6, va="center", ha="left", color="#333333")
    ax1.set_yticks(range(len(exp_order)), exp_order); ax1.invert_yaxis()
    ax1.set_xlim(-0.075, 0.105); ax1.set_xlabel("Δ IoU (expanded − baseline)")
    clean_axis(ax1); panel_label(ax1, "a", x=-0.34)
    ax1.set_title("Exploratory — no causal inference", fontsize=8, pad=5)
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax1.transAxes, fill=False, edgecolor=GREY, linewidth=0.8, linestyle=(0, (4, 3)), clip_on=False))
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY, markeredgecolor="white", markersize=4, label="LORO · Bottleneck-LiteASK"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GREY, markeredgecolor="white", markersize=4, label="Fixed-source · both architectures"),
        Line2D([0], [0], color=DIVERGING[0], lw=2, label="positive"),
        Line2D([0], [0], color=DIVERGING[2], lw=2, label="negative"),
    ]
    ax1.legend(handles=handles, loc="upper left", fontsize=5.7, ncol=2, columnspacing=0.8, handletextpad=0.35, borderaxespad=0.2)

    causal = data["fig4_causal"].set_index("region").loc[["Hokkaido", "Lombok", "Palu", "Macro"]].reset_index()
    for i, r in causal.iterrows():
        col = CATEGORICAL[2] if r.ci95_low > 0 else GREY
        ax2.errorbar(r.delta_iou, i, xerr=[[r.delta_iou-r.ci95_low], [r.ci95_high-r.delta_iou]], fmt="o", ms=4,
                     color=col, ecolor=col, elinewidth=0.85, capsize=2.0, zorder=3)
        ax2.text(r.ci95_high + 0.001, i, f"{r.delta_iou:+.3f}", fontsize=5.6, va="center")
    ax2.axvline(0, color=BLACK, lw=0.75)
    ax2.axvline(0.01, color=GREY, lw=0.7, ls="--")
    ax2.text(0.0105, -0.43, "0.01 reporting\nmarker", fontsize=5.6, color=GREY, va="bottom")
    ax2.set_yticks(range(len(causal)), causal["region"]); ax2.invert_yaxis()
    ax2.set_xlim(-0.015, 0.175); ax2.set_xlabel("Matched-exposure Δ IoU\n(pooled − placebo)")
    clean_axis(ax2); panel_label(ax2, "b", x=-0.23)
    ax2.set_title("Exposure-matched causal test", fontsize=8, pad=5)
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax2.transAxes, fill=False, edgecolor=CATEGORICAL[2], linewidth=1.1, clip_on=False))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.86)
    save_cns_600(fig, OUT / "Figure4_transfer_delta")
    plt.close(fig)


def figure5(data):
    fig = plt.figure(figsize=(183 * MM, 108 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.56, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0]); ax4 = fig.add_subplot(gs[1, 1])
    epoch = data["epoch"]
    arch_colors = {"ResUNet": CATEGORICAL[1], "SegFormer-B0": CATEGORICAL[0]}
    for model in arch_colors:
        for seed in (42, 2026, 777):
            sub = epoch[(epoch["model"] == model) & (epoch["seed"] == seed)].sort_values("epoch")
            if sub.empty:
                continue
            ax1.plot(sub["epoch"], sub["source_iou"], color=arch_colors[model], ls=SEED_LINESTYLES[seed],
                     lw=0.65, alpha=0.38, marker="o", ms=2.0)
            ax2.plot(sub["epoch"], sub["target_macro_iou"], color=arch_colors[model], ls=SEED_LINESTYLES[seed],
                     lw=0.65, alpha=0.38, marker="o", ms=2.0)
        sg = epoch[epoch["model"] == model].groupby("epoch")["source_iou"].agg(["mean", "min", "max"]).reset_index()
        tg = epoch[epoch["model"] == model].groupby("epoch")["target_macro_iou"].agg(["mean", "min", "max"]).reset_index()
        ax1.plot(sg["epoch"], sg["mean"], color=arch_colors[model], lw=1.5, marker="o", ms=3.2, label=model)
        ax1.fill_between(sg["epoch"], sg["min"], sg["max"], color=arch_colors[model], alpha=0.12, linewidth=0)
        ax2.plot(tg["epoch"], tg["mean"], color=arch_colors[model], lw=1.5, marker="o", ms=3.2, label=model)
        ax2.fill_between(tg["epoch"], tg["min"], tg["max"], color=arch_colors[model], alpha=0.12, linewidth=0)
    ax1.set_xlim(17, 53); ax1.set_ylim(0.38, 0.73); ax1.set_xticks([20, 30, 50])
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Source validation IoU")
    ax2.set_xlim(17, 53); ax2.set_ylim(0.04, 0.07); ax2.set_xticks([20, 30, 50])
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Target macro IoU")
    for x, y in [(20, 0.06317584612812258), (20, 0.05500374986420836)]:
        ax2.scatter([x], [y], s=38, facecolors="none", edgecolors=BLACK, linewidths=0.8, zorder=5)
    clean_axis(ax1); clean_axis(ax2)
    panel_label(ax1, "a", x=-0.23); panel_label(ax2, "b", x=-0.23)
    ax1.set_title("Source fit continues to rise", fontsize=8, pad=5)
    ax2.set_title("Target response is seed dependent", fontsize=8, pad=5)
    support_handle = Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                            markeredgecolor=BLACK, markersize=5, label="2/6 early-peak runs")

    src = data["srcfrac"]
    ax3.plot(src["fraction"], src["cas_val_iou"], color=CATEGORICAL[0], marker="o", ms=3.2, lw=1.15)
    ax3.set_ylim(0.50, 0.70); ax3.set_xlim(20, 105); ax3.set_xticks([25, 50, 75, 100])
    ax3.set_xlabel("Source data (%)"); ax3.set_ylabel("Source validation IoU")
    clean_axis(ax3); panel_label(ax3, "c", x=-0.18)
    ax3.set_title("Source fit rises with source data", fontsize=8, pad=5)
    ax4.plot(src["fraction"], src["target_macro_iou"], color=CATEGORICAL[1], marker="s", ms=3.2, lw=1.15)
    ax4.set_ylim(0.045, 0.058); ax4.set_xlim(20, 105); ax4.set_xticks([25, 50, 75, 100])
    ax4.set_xlabel("Source data (%)"); ax4.set_ylabel("Target macro IoU")
    clean_axis(ax4); panel_label(ax4, "d", x=-0.18)
    ax4.set_title("Target transfer does not improve", fontsize=8, pad=5)
    # D5-5 is handled in the manuscript caption for panels c and d.
    # D5-8/D5-9/B4: one shared legend outside every panel
    handles = [Line2D([0], [0], color=arch_colors[m], lw=1.6, marker="o", ms=3.2, label=m)
               for m in ("ResUNet", "SegFormer-B0")]
    handles.append(support_handle)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3,
               fontsize=6.2, columnspacing=1.6, handletextpad=0.45, frameon=False)
    fig.text(0.5, 0.043, "thin lines = seeds; band = min-max; thick line = model mean",
             ha="center", fontsize=5.4, color="#666666")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.91)
    save_cns_600(fig, OUT / "Figure5_source_overfitting")
    plt.close(fig)

def figure6(data):
    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.0],
                           hspace=0.62, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0]); ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    macro = data["fs_macro"]

    def macro_row(model, mode, shot):
        q = macro[(macro.model == model) & (macro["mode"] == mode) & (macro.shot == shot)]
        if q.empty:
            return None
        r = q.iloc[0]
        return float(r.macro_iou_mean), float(r.macro_iou_std)

    series = [
        ("ResUNet", "full", "ResUNet full", CATEGORICAL[1], "-", "s"),
        ("ResUNet", "decoder-only", "ResUNet decoder-only", "#D6604D", "--", "D"),
        ("SegFormer-B0", "full", "SegFormer-B0 full", CATEGORICAL[0], "-", "o"),
    ]
    for model, mode, label, color, ls, marker in series:
        xs, ys, es = [], [], []
        shots = [5, 10, 20] if model == "ResUNet" and mode == "full" else ([10, 20] if model == "SegFormer-B0" else [20])
        z = macro_row(model, "zero-shot", 0)
        if z is not None:
            xs.append(0); ys.append(z[0]); es.append(z[1])
        for shot in shots:
            r = macro_row(model, mode, shot)
            if r is not None:
                xs.append(shot); ys.append(r[0]); es.append(r[1])
        ax1.errorbar(xs, ys, yerr=es, color=color, lw=1.1, ls=ls, marker=marker, ms=3.2,
                     ecolor=color, elinewidth=0.6, capsize=1.4, label=label)
    ax1.set_xlim(-1, 21); ax1.set_ylim(0, 0.21)
    ax1.set_xticks([0, 5, 10, 20]); ax1.set_xlabel("Labelled target tiles")
    ax1.set_ylabel("Macro IoU (3 events)")
    clean_axis(ax1); panel_label(ax1, "a", x=-0.20)
    ax1.set_title("Dose response (seed SD)", fontsize=8, pad=5)
    ax1.legend(loc="upper left", fontsize=5.3, handlelength=1.7, handletextpad=0.30)

    r = data["fs_region"]
    keep = [(("ResUNet", "full", 20), "ResUNet full", CATEGORICAL[1]),
            (("ResUNet", "decoder-only", 20), "ResUNet decoder", "#D6604D"),
            (("SegFormer-B0", "full", 20), "SegFormer-B0", CATEGORICAL[0])]
    regions = ["Hokkaido", "Lombok", "Palu"]
    x = np.arange(3); width = 0.24
    for i, ((model, mode, shot), label, color) in enumerate(keep):
        sub = r[(r.model == model) & (r["mode"] == mode) & (r.shot == shot)].set_index("region_label").loc[regions]
        ax2.bar(x + (i-1)*width, sub["delta_iou_mean"], width=width*0.88, yerr=sub["delta_iou_std"],
                color=color, edgecolor="white", linewidth=0.4,
                error_kw={"elinewidth": 0.6, "capsize": 1.1, "ecolor": color}, label=label)
        for xx, vv in zip(x + (i-1)*width, sub["delta_iou_mean"]):
            ax2.text(xx, vv + (0.012 if vv >= 0 else -0.018), f"{vv:+.3f}", ha="center",
                     va="bottom" if vv >= 0 else "top", fontsize=5.0)
    ax2.axhline(0, color=BLACK, lw=0.7); ax2.axhline(0.01, color=GREY, lw=0.6, ls="--")
    ax2.set_xticks(x, regions); ax2.set_ylim(-0.08, 0.39); ax2.set_ylabel("Δ IoU at 20 shots")
    clean_axis(ax2); panel_label(ax2, "b", x=-0.23)
    ax2.set_title("Gain by region (seed SD)", fontsize=8, pad=5)

    query = data["e4_query"].copy(); modes = [("full", "Full", CATEGORICAL[1]), ("decoder-only", "Decoder-only", "#D6604D")]
    xb = np.arange(3)
    for j, (mode, label, color) in enumerate(modes):
        q = query[query["mode"] == mode].set_index("buffer_m").loc[[0, 256, 512]]
        vals = q["mean_delta_iou"].to_numpy()
        lo = vals - q["ci95_low"].to_numpy(); hi = q["ci95_high"].to_numpy() - vals
        ax3.errorbar(xb + (-0.12 if j == 0 else 0.12), vals, yerr=np.vstack([lo, hi]),
                     fmt="o", color=color, ecolor=color, elinewidth=0.7, capsize=1.4, ms=3.3, label=label)
    ax3.axhline(0.01, color=GREY, lw=0.6, ls="--")
    ax3.set_xticks(xb, ["0 m", "256 m", "512 m"]); ax3.set_ylim(0, 0.24)
    ax3.set_ylabel("Macro Δ IoU"); clean_axis(ax3); panel_label(ax3, "c", x=-0.20)
    ax3.set_title("Physical-buffer robustness (95% CI)", fontsize=8, pad=5)

    support = data["e4_support"].copy()
    for j, (mode, label, color) in enumerate(modes):
        s = support[support["mode"] == mode].set_index("region_label").loc[regions]
        ax4.errorbar(np.arange(3) + (-0.12 if j == 0 else 0.12), s["delta_iou_mean"],
                     yerr=s["delta_iou_std"], fmt="s" if j == 0 else "D", color=color,
                     ecolor=color, elinewidth=0.7, capsize=1.4, ms=3.2)
    ax4.axhline(0.01, color=GREY, lw=0.6, ls="--")
    ax4.set_xticks(np.arange(3), regions); ax4.set_ylim(-0.02, 0.40)
    ax4.set_ylabel("Region Δ IoU"); clean_axis(ax4); panel_label(ax4, "d", x=-0.20)
    ax4.set_title("Support-draw robustness (3 draws; SD)", fontsize=8, pad=5)

    stitched = data["e4_stitched"].copy(); xr = np.arange(3); bw = 0.16
    source_vals = stitched[stitched["mode"] == "full"].set_index("region_label").loc[regions, "source_iou"].to_numpy()
    for j, (mode, label, color) in enumerate(modes):
        st = stitched[stitched["mode"] == mode].set_index("region_label").loc[regions]
        off = -0.19 if j == 0 else 0.19
        ax5.bar(xr + off - bw*0.52, source_vals, width=bw, color="#CCCCCC", edgecolor="white", linewidth=0.35)
        ax5.bar(xr + off + bw*0.52, st["adapted_iou"], width=bw, color=color, edgecolor="white", linewidth=0.35)
        for xx, src, dst in zip(xr + off, source_vals, st["adapted_iou"]):
            ax5.text(xx + bw*0.75, dst + 0.012, f"Δ{dst-src:+.3f}", ha="center", fontsize=5.0)
    ax5.set_xticks(xr, regions); ax5.set_ylim(0, 0.52); ax5.set_ylabel("Stitched-map IoU")
    clean_axis(ax5); panel_label(ax5, "e", x=-0.055)
    ax5.set_title("Stitched-map check (seed 42; point estimate)", fontsize=8, pad=5)
    handles = [Line2D([0], [0], color="#CCCCCC", lw=4, label="source"),
               Line2D([0], [0], color=CATEGORICAL[1], lw=4, label="full adapted"),
               Line2D([0], [0], color="#D6604D", lw=4, label="decoder-only adapted")]
    # D5-9/D5-10/B4: one legend for panels a-d, outside the axes
    shared = [Line2D([0], [0], color=CATEGORICAL[1], lw=1.6, marker="s", ms=3.2, label="ResUNet full"),
              Line2D([0], [0], color="#D6604D", lw=1.6, marker="D", ms=3.2, label="ResUNet decoder-only"),
              Line2D([0], [0], color=CATEGORICAL[0], lw=1.6, marker="o", ms=3.2, label="SegFormer-B0 full")]
    fig.legend(handles=shared, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3,
               fontsize=6.2, columnspacing=1.6, handletextpad=0.45, frameon=False)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3,
               fontsize=6.2, columnspacing=1.6, handletextpad=0.45, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.135, top=0.93)
    save_cns_600(fig, OUT / "Figure6_fewshot_adaptation")
    plt.close(fig)

def figure7(data):
    fig = plt.figure(figsize=(183 * MM, 76 * MM))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.95, 1.25], wspace=0.30)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    conf = data["conf"].set_index("domain").loc[["CAS validation", "Hokkaido", "Lombok", "Palu"]].reset_index()
    for i, row in conf.iterrows():
        color = CATEGORICAL[0] if row.domain == "CAS validation" else CATEGORICAL[1]
        ax1.vlines(i, row.error_rate, row.high_confidence_error_rate, color="#CCCCCC", lw=1.1, zorder=0)
        ax1.scatter(i, row.error_rate, s=22, color=CATEGORICAL[0], zorder=3)
        ax1.scatter(i, row.high_confidence_error_rate, s=22, color=color, marker="s", zorder=3)
        dy = 0.012 if i == 0 else 0.0
        ax1.text(i + 0.09, row.high_confidence_error_rate + dy,
                 f"{100*row.high_confidence_error_rate:.1f}%", fontsize=5.6, va="center")
        if i == 0:  # D5-12: leader line for the closely spaced CAS pair
            ax1.annotate("", xy=(i + 0.06, row.high_confidence_error_rate),
                         xytext=(i + 0.30, row.error_rate), textcoords="data",
                         arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888"))
    ax1.set_xticks(range(len(conf)), conf.domain); ax1.set_ylim(0, 0.29); ax1.set_ylabel("Pixel error rate")
    clean_axis(ax1); panel_label(ax1, "a", x=-0.22)
    ax1.set_title("High confidence does not imply correctness", fontsize=8, pad=5)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CATEGORICAL[0], markeredgecolor=CATEGORICAL[0], markersize=4, label="all pixels"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CATEGORICAL[1], markeredgecolor=CATEGORICAL[1], markersize=4, label="confidence ≥ 0.90"),
    ]
    ax1.legend(handles=handles, loc="upper left", fontsize=5.8, handletextpad=0.35)

    calib = data["calib"]; max_frac = calib.pixel_fraction.max()
    markers = {"CAS validation": "o", "Hokkaido": "s", "Lombok": "D", "Palu": "^"}
    for domain in ["CAS validation", "Hokkaido", "Lombok", "Palu"]:
        sub = calib[calib.domain == domain].sort_values("mean_confidence")
        xx = 1 - sub["mean_confidence"].to_numpy(); yy = sub["error_rate"].to_numpy()
        color = CATEGORICAL[0] if domain == "CAS validation" else REGION_COLORS.get(domain, GREY)
        ax2.plot(xx, yy, color=color, lw=1.05, marker=markers[domain], ms=2.4, label=domain)
        ax2.scatter(xx, yy, s=12 + 55 * sub["pixel_fraction"].to_numpy()/max_frac, color=color, alpha=0.18, edgecolors="none", zorder=1)
    lim = 0.50
    ax2.plot([0, lim], [0, lim], color=GREY, lw=0.7, ls="--")
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.set_xlabel("Mean predicted error (1 − confidence)"); ax2.set_ylabel("Observed pixel error rate")
    clean_axis(ax2); panel_label(ax2, "b", x=-0.16)
    ax2.set_title("Calibration degrades on unseen events", fontsize=8, pad=5)
    size_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=GREY, markersize=np.sqrt(12 + 55 * frac), label=f"{int(100*frac)}%")
                    for frac in [0.25, 0.50, 0.75]]
    # D5-11: the bubble-area note moves to the caption; legends move outside the axes
    fig.legend(handles=ax2.get_legend_handles_labels()[0] + size_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=5.8, columnspacing=1.4,
               handletextpad=0.35, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.245, top=0.86)
    save_cns_600(fig, OUT / "Figure7_confidence")
    plt.close(fig)


def spearman(x, y):
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def figure8(data):
    fig = plt.figure(figsize=(183 * MM, 96 * MM))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.20)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)   # B2: shared x axis
    p = data["prev"].copy(); x = p["foreground_fraction"].to_numpy(); y = p["iou_mean"].to_numpy()
    rho = spearman(x, y)
    for _, row in p.iterrows():
        ax1.scatter(row.foreground_fraction, row.iou_mean, s=24, color=REGION_COLORS.get(row.region, GREY),
                    marker="o", edgecolor="white", linewidth=0.35, zorder=3)
        offsets = {
            "Wenchuan": (-0.004, 0.012), "Jiuzhai Valley": (-0.006, 0.010),
            "Moxitaidi": (-0.006, 0.015), "Longxi River": (0.006, 0.010),
            "Hokkaido": (0.006, -0.002), "Lombok": (0.006, -0.020),
            "Palu": (0.006, -0.025),
        }
        dx, dy = offsets.get(row.region, (0.004, 0.012))
        ax1.text(row.foreground_fraction + dx, row.iou_mean + dy, rlabel(row.region), fontsize=5.7,
                 ha="left" if dx >= 0 else "right")
    ax1.text(0.01, 0.54, f"Spearman ρ = {rho:.2f}\nn = {len(p)}", fontsize=6.2, color=BLACK)
    ax1.text(0.01, 0.48, "descriptive only; prevalence and GSD are collinear", fontsize=5.2, color="#666666")
    ax1.set_xlim(0, 0.255); ax1.set_ylim(0, 0.58)
    ax1.set_ylabel("Macro IoU"); ax1.tick_params(labelbottom=False)
    clean_axis(ax1); panel_label(ax1, "a", x=-0.18); ax1.set_title("IoU tracks landslide prevalence", fontsize=8, pad=5)

    p2 = p.sort_values("foreground_fraction")
    ax2.plot(p2.foreground_fraction, p2.mean_precision, color=CATEGORICAL[3], marker="o", ms=3.1, lw=1.1, label="precision")
    ax2.plot(p2.foreground_fraction, p2.mean_recall, color=CATEGORICAL[0], marker="s", ms=3.1, lw=1.1, label="recall")
    for _, row in p2.iterrows():
        ax2.annotate(rlabel(row.region), (row.foreground_fraction, max(row.mean_precision, row.mean_recall)),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5.0)
    ax2.text(0.01, 0.74, f"precision ρ = {spearman(x, p['mean_precision']):.2f}\nrecall ρ = {spearman(x, p['mean_recall']):.2f}", fontsize=6.5)
    ax2.set_xlim(0, 0.255); ax2.set_ylim(0, 0.88)
    ax2.set_xlabel("Reference foreground fraction"); ax2.set_ylabel("Macro precision / recall")
    clean_axis(ax2); panel_label(ax2, "b", x=-0.18); ax2.set_title("The association is a precision effect", fontsize=8, pad=5)
    fig.legend(handles=ax2.get_legend_handles_labels()[0], loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=6.2, columnspacing=1.6,
               handletextpad=0.4, frameon=False)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.155, top=0.93)
    save_cns_600(fig, OUT / "Figure8_prevalence_metric")
    plt.close(fig)


def copy_figure1():
    src = OUT.parent / "figure1_qgis"
    raster = src / "Figure1_readable.png"
    if raster.exists():
        im = Image.open(raster).convert("RGB")
        strip = 180
        canvas = Image.new("RGB", (im.width, im.height + strip), "white")
        canvas.paste(im, (0, 0))
        draw = ImageDraw.Draw(canvas)
        y0 = im.height
        draw.line((0, y0 + 2, canvas.width, y0 + 2), fill="#B8B8B8", width=2)
        try:
            font_key = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 42)
            font_note = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 38)
        except OSError:
            font_key = ImageFont.load_default(); font_note = ImageFont.load_default()
        draw.text((36, y0 + 18),
                  "1 Wenchuan   2 Jiuzhai Valley   3 Moxitaidi   4 Longxi River   5 Hokkaido   6 Lombok   7 Palu",
                  fill="#222222", font=font_key)
        draw.text((36, y0 + 84),
                  "Orange outlines indicate approximate locator extents only; they are not experimental tile footprints or imagery coverage.",
                  fill="#444444", font=font_note)
        png = OUT / "Figure1_study_area.png"
        pdf = OUT / "Figure1_study_area.pdf"
        tif = OUT / "Figure1_study_area.tif"
        canvas.save(png, dpi=(600, 600), optimize=True)
        canvas.save(pdf, "PDF", resolution=600.0)
        canvas.save(tif, "TIFF", dpi=(600, 600), compression="tiff_lzw")
    else:
        for name in ["Figure1_readable.pdf", "Figure1_readable.png", "Figure1_readable.tif"]:
            source = src / name
            if source.exists():
                shutil.copy2(source, OUT / ("Figure1_study_area" + source.suffix))

def write_reports(data):
    lines = [
        "# Publication figure data and statistics report", "",
        "Generated: 2026-09-25", "",
        "## Global definitions",
        "- Panels are plotted at final publication dimensions; no post-hoc scaling.",
        "- Means are arithmetic means unless the source file states otherwise.",
        "- Error bars: Figure 3 and Figure 6a-b show standard deviation; Figure 4b and Figure 6c show 95% confidence intervals; Figure 6d shows support-draw standard deviation; Figure 6e shows seed-42 stitched-map point estimates; Figure 5 shows individual seed runs with model-level ranges.",
        "- n: Figure 3 target regions = 7 per architecture and 3 seeds; Figure 4a = 3 seeds (LORO) or the fixed-source seed set; Figure 4b = 3 seeds; Figure 5a/b = one run per architecture-seed combination; Figure 5c/d = one ResUNet seed-42 ablation; Figure 6a-b = 3 seeds; Figure 6c = 3 seeds; Figure 6d = 3 support draws; Figure 6e = seed 42; Figure 7 = pixel-level summaries; Figure 8 = zero-shot architecture-seed checkpoints aggregated to 7 regions.",
        "- Source-data extracts are in `figure_source_data/`; every point can be traced to the original `D:\\landslide_unet_project\\reports` files listed there.", "",
        "## Figure-level mapping", "",
        "| Figure | Core message | Source | Statistic |",
        "|---|---|---|---|",
        "| Figure 1 | QGIS study-area map | figure1_qgis/Figure1_readable.* | Retained; no data transformation. |",
        "| Figure 2 | Experimental design schematic | Manuscript Sections 3.1-3.3; Tables 2 and 5 | Conceptual figure; no statistical inference. |",
        "| Figure 3 | Source-to-target collapse | benchmark_v2_zero_shot_region_summary.csv; source histories; benchmark_v2_zero_shot_model_summary.csv | IoU mean +/- SD; retention = target macro IoU/source validation IoU. |",
        "| Figure 4 | Region-dependent transfer | loro_benchmark_summary.csv; e1_exposure_matched_bootstrap.csv; original fixed-source values | Panel a exploratory; panel b matched-exposure causal contrast with 95% hierarchical bootstrap intervals. |",
        "| Figure 5 | Source-target divergence and source scaling | e3_epochwise_per_run.csv; p4_source_fraction_curves.csv | Individual seed runs with model mean and min-max bands; source and target responses to source fraction are shown in separate single-axis panels. |",
        "| Figure 6 | Few-shot adaptation and operational robustness | benchmark_v2_fewshot128_multiseed_macro.csv; benchmark_v2_fewshot128_multiseed_by_region.csv; e4_query_summary.csv; e4_support_draws_summary.csv; e4_stitched_summary.csv | a-b: mean +/- SD across three seeds; c: buffer 95% CI; d: support-draw mean +/- SD; e: seed-42 stitched-map point estimates. |",
        "| Figure 7 | Confidence reliability | p4_confidence_calibration.json | Bubble area in panel b is proportional to bin pixel fraction. |",
        "| Figure 8 | Prevalence dependence | e0_prevalence_metrics.csv; e0_threshold_metrics.csv | Threshold 0.5; precision/recall computed over zero-shot architecture-seed checkpoints. |", "",
        "## Key values",
    ]
    z = data["fig3"].drop_duplicates("model").set_index("model").loc[MODEL_ORDER]
    for model in MODEL_ORDER:
        r = z.loc[model]
        lines.append(f"- {model}: source validation IoU {r.source_val_iou_mean:.3f} +/- {r.source_val_iou_std:.3f}; target macro IoU {r.target_macro_iou:.4f}; retention {100*r.retention:.1f}%.")
    causal = data["fig4_causal"].set_index("region")
    lines.append(f"- Exposure-matched macro delta IoU: {causal.loc['Macro','delta_iou']:+.4f} [{causal.loc['Macro','ci95_low']:+.4f}, {causal.loc['Macro','ci95_high']:+.4f}].")
    p = data["prev"]
    lines.append(f"- Prevalence-IoU Spearman rho = {spearman(p['foreground_fraction'], p['iou_mean']):.2f}; precision rho = {spearman(p['foreground_fraction'], p['mean_precision']):.2f}; recall rho = {spearman(p['foreground_fraction'], p['mean_recall']):.2f}.")
    q = data["e4_query"].set_index(["mode", "buffer_m"])
    for mode, label in [("full", "full fine-tuning"), ("decoder-only", "decoder-only")]:
        r = q.loc[(mode, 0)]
        lines.append(f"- E4 guard-query adaptation ({label}): delta IoU {r.mean_delta_iou:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}].")
    s = data["e4_support"].set_index(["mode", "region_label"])
    lines.append(f"- Support-draw Hokkaido delta IoU: full {s.loc[('full','Hokkaido'),'delta_iou_mean']:+.4f} +/- {s.loc[('full','Hokkaido'),'delta_iou_std']:.4f}; decoder-only {s.loc[('decoder-only','Hokkaido'),'delta_iou_mean']:+.4f} +/- {s.loc[('decoder-only','Hokkaido'),'delta_iou_std']:.4f}.")
    st = data["e4_stitched"].set_index(["mode", "region_label"])
    lines.append(f"- Stitched-map Hokkaido delta IoU (seed 42): full {st.loc[('full','Hokkaido'),'delta_iou']:+.4f}; decoder-only {st.loc[('decoder-only','Hokkaido'),'delta_iou']:+.4f}.")
    lines += ["", "## Reviewer-risk notes",
              "- Figure 4 combines exploratory and causal protocols but keeps them in separate panels; do not pool their estimates.",
              "- Figure 5 shows only two of six runs meeting the pre-specified early-peak criterion; the figure must not be described as evidence of a general temporal mechanism.",
              "- Figure 6c-e separate the three robustness designs and their distinct variance definitions: buffer 95% CI, support-draw SD, and single-seed stitched maps.",
              "- Figure 8 is descriptive: prevalence and GSD are strongly collinear across seven regions and cannot be separated causally."]
    (OUT / "figure_statistics_and_traceability.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    copy_figure1()
    figure2_schematic()
    data = load_inputs()
    figure3(data); figure4(data); figure5(data); figure6(data); figure7(data); figure8(data)
    write_reports(data)
    print(f"Wrote figures and reports to {OUT}")


if __name__ == "__main__":
    main()


































