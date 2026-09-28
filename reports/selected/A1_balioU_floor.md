# A1 Balanced IoU Floor Audit

Date: 2026-09-27

## Finding

The plan's proposed identity `Balanced IoU = (TPR + 1 - FPR) / 2` describes balanced accuracy, not balanced IoU.

The implementation defines Balanced IoU as:

`0.5 * (foreground IoU + background IoU)`

where:

- `foreground IoU = TP / (TP + FP + FN)`
- `background IoU = TN / (TN + FP + FN)`

Therefore the floor argument in the plan does not apply to the manuscript table.

## Aggregation

For each model-seed-region, the evaluator pools pixel confusion counts across all tiles before computing IoU, MCC and Balanced IoU. The manuscript region values are means of those 12 model-seed summaries.

## Action

The manuscript now states the aggregation and definition explicitly. No Table 15 numbers were changed.
