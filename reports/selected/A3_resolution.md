# A3 Resolution Experiment Audit

Date: 2026-09-27

## Fact

The 256 x 256 pilot models were independently trained from random initialization.

Configuration from `run_benchmark_v2_256_pilot.ps1`:

- input size: 256
- epochs: 50
- batch size: 8
- gradient accumulation: 4
- effective batch size: 32
- learning rate: 1e-4
- weight decay: 1e-4
- seed: 42
- loss: PolyGHMDiceLoss

They were not obtained by feeding 256-resize inputs into 128-trained checkpoints.

## Limitation

This is a seed-42, four-architecture pilot. The sharp Jiuzhai Valley decrease is not explained by available boundary metrics and should not be generalized to native-resolution training.

## Action

Methods, Table 12 caption and Section 4.9 were updated accordingly.
