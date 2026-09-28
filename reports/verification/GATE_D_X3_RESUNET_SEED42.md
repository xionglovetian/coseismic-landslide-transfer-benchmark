# Gate-D Verification: X3 ResUNet Pooled Seed 42

Date: 2026-09-28

## Purpose

This is an independent, model-level reproduction of one confirmatory Q2 run,
not a re-summary of packaged tables.

## Frozen input and configuration

- Code: published repository scripts
- Training entry: `scripts/train_exposure_matched.py`
- Evaluation entry: `scripts/evaluate_q2_tile_metrics.py`
- Model: ResUNet
- Regime: pooled source
- Seed: 42
- Optimizer steps: 1,120
- Batch size: 32
- Input size: 128
- Initial checkpoint:
  `outputs/q2_x2_aug_none_resunet_seed42/best_model.pth`
- Initial checkpoint SHA-256:
  `8312CB6D76E6A9B399F3619BB886DFAF6CA9FA1BEEC84BEC65E5D71BD2531C3F`
- Training-code SHA-256:
  `7D70E0743210A0E7CD704A5A87B8B25351E4FC17D979B206DA2396D3746D24AA`
- Tile-evaluation-code SHA-256:
  `79824565D5EC5C8C24F70052DBD9D102A356C698B350B3EC1A936B4D1DA9C159`
- Input manifest SHA-256:
  `0E63FA0647D0BA1052C29C6C17D8DA990A6FA28D6D39C11D6326AFF766D1F9CF`

## Training result

The rerun completed 1,120 steps in about 181 seconds. All source-validation
metrics and confusion counts were identical to the original run.

| Metric | Original | Reproduced | Absolute delta |
|---|---:|---:|---:|
| IoU | 0.518395166665147 | 0.518395166665147 | 0 |
| Dice | 0.682819832473122 | 0.682819832473122 | 0 |
| F1 | 0.6828197830210376 | 0.6828197830210376 | 0 |
| Precision | 0.7626582100492547 | 0.7626582100492547 | 0 |
| Recall | 0.6181130687874035 | 0.6181130687874035 | 0 |
| Accuracy | 0.9100716574150111 | 0.9100716574150111 | 0 |
| TP | 813,587 | 813,587 | 0 |
| FP | 253,191 | 253,191 | 0 |
| FN | 502,656 | 502,656 | 0 |
| TN | 6,835,558 | 6,835,558 | 0 |

The original and reproduced `metrics.json` files have the same SHA-256:

`159FDFFC52F85583DF654BC9328D224B6C6568CCB8B1EABDD329CF96B854C88A`

## Target tile evaluation

The reproduced final checkpoint was evaluated on Hokkaido, Lombok and Palu
using the same 2,737 evaluation chips and the same preprocessing.

- Original tile CSV SHA-256:
  `3E2BB3810184BFDE5E4ABDF5FA2566303AEAD513AF112C5C2E7688139FAFA95C`
- Reproduced tile CSV SHA-256:
  `3E2BB3810184BFDE5E4ABDF5FA2566303AEAD513AF112C5C2E7688139FAFA95C`
- Row count: 2,737 versus 2,737.
- Different `tp`, `fp`, `fn`, `tn`, `f1_2`, `f1_4`, or `hd95` values: 0.

The tile files are bitwise identical.

## Checkpoint comparison

The serialized `.pth` files have different file hashes because PyTorch stores
serialization/run metadata inside the container. The learned tensors are
identical:

- Source tensor count: 278.
- Source tensor elements: 24,455,423.
- Tensors with different values: 0.
- Different elements: 0.
- Maximum absolute tensor difference: 0.0.
- Canonical state-dict SHA-256:
  `68f623295f1e16d871a196e1601ab182c59905476afdfa73ff3c7432a2107143`.

## Verdict

- Training determinism: **PASS**
- Aggregate source-validation reproduction: **PASS**
- Target tile-metric reproduction: **PASS**
- Learned-weight equality: **PASS**

This verifies the X3 ResUNet pooled seed-42 training and evaluation path
end-to-end. The full project still requires the complete queue to be run on a
clean environment before calling the entire multi-day study independently
reproduced.