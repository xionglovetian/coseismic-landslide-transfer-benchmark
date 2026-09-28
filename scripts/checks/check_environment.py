"""Check the optional model-execution environment for benchmark reruns."""

from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    "albumentations",
    "einops",
    "matplotlib",
    "monai",
    "numpy",
    "pandas",
    "PIL",
    "skimage",
    "sklearn",
    "scipy",
    "seaborn",
    "segmentation_models_pytorch",
    "timm",
    "torch",
    "torchvision",
    "torchmetrics",
    "tqdm",
]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    if sys.version_info[:2] != (3, 10):
        print("WARN: the reference environment used Python 3.10")
    missing = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
        except Exception as exc:  # pragma: no cover - diagnostic output
            missing.append((package, str(exc)))
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"OK: {package} {version}")
    if missing:
        for package, error in missing:
            print(f"MISSING: {package} ({error})")
        return 1
    torch = importlib.import_module("torch")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
