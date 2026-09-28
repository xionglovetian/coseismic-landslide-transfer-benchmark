import csv, json, statistics, sys, time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
from src.models import build_model

MODELS = [
    ("UNet", "UNet"),
    ("NestedUNet", "NestedUNet"),
    ("AS_UNet", "AS_UNet"),
    ("U-Net++", "UNetPlusPlus"),
    ("ASK-UNet++", "ASKUNetPlusPlus"),
    ("LiteASK-UNet++", "LiteASKUNetPlusPlus"),
    ("DeepLiteASK-UNet++", "DeepLiteASKUNetPlusPlus"),
    ("Bottleneck-LiteASK", "BottleneckLiteASKUNetPlusPlus"),
]


def benchmark_fps(model, input_size, batch_size, warmup=10, iterations=80):
    x = torch.randn(batch_size, 3, input_size, input_size, device="cuda")
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x)
        end.record()
        torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1000.0
    return (iterations * batch_size) / seconds


def main():
    reports = PROJECT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, model_name in MODELS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(model_name).cuda().eval()
        params = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        example = torch.randn(1, 3, 128, 128, device="cuda")
        with torch.inference_mode():
            with FlopCounterMode(display=False) as counter:
                model(example)
        flops = counter.get_total_flops()

        fps_b1 = benchmark_fps(model, 128, 1, warmup=15, iterations=100)
        fps_b32 = benchmark_fps(model, 128, 32, warmup=5, iterations=40)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x32 = torch.randn(32, 3, 128, 128, device="cuda")
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x32)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        row = {
            "model": label,
            "params": params,
            "trainable_params": trainable,
            "params_million": params / 1e6,
            "model_size_fp32_mb": params * 4 / 1024**2,
            "gflops_b1_128": flops / 1e9,
            "fps_b1_bf16": fps_b1,
            "fps_b32_bf16": fps_b32,
            "peak_inference_mb_b32": peak_mb,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del model, example, x32
        torch.cuda.empty_cache()

    fields = list(rows[0].keys())
    with (reports / "model_complexity.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Model Complexity And Inference Benchmark",
        "",
        "Device: NVIDIA GeForce RTX 4060 Laptop GPU",
        "",
        "Protocol: input 128x128, FP32 FLOP counting, BF16 autocast FPS, batch 1 and batch 32. FPS includes repeated forward passes after warm-up.",
        "",
        "| Model | Params (M) | FP32 Size (MB) | GFLOPs @1x3x128x128 | FPS batch 1 | FPS batch 32 | Peak VRAM batch 32 (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["model"]} | {row["params_million"]:.3f} | {row["model_size_fp32_mb"]:.1f} | '
            f'{row["gflops_b1_128"]:.3f} | {row["fps_b1_bf16"]:.1f} | {row["fps_b32_bf16"]:.1f} | '
            f'{row["peak_inference_mb_b32"]:.1f} |'
        )
    lines += [
        "",
        "Notes:",
        "",
        "- FLOPs count convolution and matrix-multiply operators; interpolation and normalization overhead may be undercounted by the counter.",
        "- VRAM is PyTorch peak allocated memory during one BF16 batch-32 forward pass, including model weights.",
        "- Randomly initialized weights do not affect operator counts or speed materially.",
        "",
        "File: D:\\landslide_unet_project\\reports\\model_complexity.csv",
    ]
    (reports / "model_complexity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", reports / "model_complexity_report.md")


if __name__ == "__main__":
    main()
