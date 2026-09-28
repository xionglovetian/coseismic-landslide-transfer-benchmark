# Q2 Major-Revision Protocol: X2 Augmentation Decision

Date frozen: 2026-09-27
Protocol ID: `q2-major-x2-aug-20260927-v2`
Amended by: `q2_x2_augmentation_amendment_01_20260927.md`

## Purpose

Resolve the manuscript-internal contradiction that the main benchmark uses
`current` augmentation although a one-seed ablation reports better source and
target performance for `none`.

## Design

- Primary models: ResUNet and Bottleneck-LiteASK.
- Seeds: 42, 2026, 777.
- Augmentation arms: `current`, `none`, `strong`.
- Input size: 128.
- Training: identical 50-epoch source-only protocol, batch 32, lr 1e-4,
  PolyGHMDiceLoss, AMP, source-validation checkpoint selection.
- Pairing: every model x seed is trained under all three arms.
- Total: 2 x 3 x 3 = 18 paired runs.

## Primary Selection Rule

The augmentation used by the primary benchmark will be selected using source
validation only, before inspecting the new six-region target results.

1. For each primary architecture, compute the seed-paired source-validation IoU
   difference relative to `current`.
2. `none` replaces `current` only if it is non-inferior for both primary architectures
   using a prespecified non-inferiority margin of -0.010 IoU and has a
   non-negative mean difference across the six model-seed pairs.
3. If that rule is not met, retain `current`.
4. `strong` is not eligible as the primary configuration because the existing
   one-seed result already shows a large source-validation collapse. It is
   retained as an augmentation-sensitivity arm.

Target six-region results are reported after the source-only decision and are
not used to reverse the primary configuration choice.

## Target Evaluation

- Six independent held-out regions: Hokkaido, Lombok, Palu, Wenchuan,
  Longxi River and Jiuzhai Valley.
- Moxitaidi is reported separately because it is source-adjacent and is not
  included in held-out macro averages.
- Endpoints: IoU, balanced IoU, MCC, precision and recall at threshold 0.5.
- Uncertainty: report seed-level values and paired seed deltas. Do not use
  the target results for hyperparameter selection.

## Reuse of Existing Runs

Existing runs are not reused for the primary paired comparison because some
were trained before the current trainer version. The experiment creates a
clean, code-version-matched 18-run grid. Existing one-seed runs are retained
only as historical cross-checks.
