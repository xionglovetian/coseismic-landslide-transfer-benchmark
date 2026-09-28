# Q2 final manuscript revision summary

Date: 2026-09-28

## Canonical output

`C:\Users\ASUS\Desktop\初稿.docx`

The file was updated in place. No alternate DOCX version was retained.

## Final experiment-driven changes

### X2 augmentation selection

Protocol: `q2-major-x2-aug-20260927-v2`

- Grid: ResUNet + Bottleneck-LiteASK, 3 seeds, current/none/strong, 18 paired runs.
- Selection variable: source-validation IoU only.
- `none` source delta vs current: +0.0432 +/- 0.0158 across six paired model-seed pairs.
- Target six-region macro delta after selection: +0.0122 +/- 0.0108.
- Strong target macro delta: +0.0210 +/- 0.0152, but source-validation delta was -0.0552 +/- 0.0059.
- Final selection: **no augmentation (`none`)**.

Manuscript effect: Tables 3, 5-7, 9 and 12, abstract, contributions, Results, Discussion and Conclusion were updated. The four-architecture benchmark in Table 4 is explicitly labelled as the original current-augmentation reference benchmark.

### X1 resolution experiment

- ResUNet: 128 = 0.1648; retrained 256 = 0.1405 (-0.0243 +/- 0.0006); 128 weights at 256 = 0.1885 (+0.0237 +/- 0.0043).
- SegFormer-B0: 128 = 0.1377; retrained 256 = 0.1406 (+0.0029 +/- 0.0090); 128 weights at 256 = 0.1648 (+0.0271 +/- 0.0094).

Conclusion: the old 256 pilot mixed input-scale and weight-scale effects. Independent 256 retraining improved source-validation IoU but did not provide a consistent target gain. Evaluating 128-trained weights at 256 increased target IoU for both models, but source-validation IoU fell to 0.4146 (ResUNet) and 0.3211 (SegFormer-B0), so this arm diagnoses weight-scale mismatch rather than a viable higher-resolution configuration.

### X3 exposure-matched source expansion

Two architectures, three target regions, three seeds, 1,120 steps, batch 32, 35,840 exposures.

- ResUNet target macro IoU: +0.0773 [+0.0076, +0.1500]; balanced IoU +0.0603 [-0.0088, +0.1500]; MCC +0.1573 [+0.0199, +0.2952].
- Bottleneck-LiteASK target macro IoU: +0.0456 [+0.0010, +0.0961]; balanced IoU +0.0383 [-0.0208, +0.1055]; MCC +0.0990 [+0.0027, +0.1976].

Region-level means:
- Hokkaido: ResUNet +0.1995; Bottleneck +0.1256.
- Lombok: ResUNet +0.0107; Bottleneck +0.0074.
- Palu: ResUNet +0.0215; Bottleneck +0.0037.

Conclusion: positive average IoU and MCC effects with mixed balanced IoU and strong region dependence.

### X5 uncertainty correction

- Primary intervals now use the connected-component two-stage spatial bootstrap.
- 10,000 draws for the three-region X3 comparison.
- Merged-block sensitivity gives the same qualitative decision.
- The all-six-region LORO extension is not used for the primary causal claim because its placebo controls use the earlier augmentation setting.

### Figure

Figure 4 was replaced with:
- (a) target-macro endpoint deltas,
- (b) region-level IoU deltas,
- (c) target macro IoU for the X1 three arms; (d) source-validation IoU for the same arms.

## Verification

- 9 inline images retained.
- 16 tables retained.
- 0 tracked revisions.
- 0 comments.
- 31 rendered pages.
- PDF image count: 9.
- Page-boundary QA passed.


## Editable vector archive

- Figure 4: `C:\Users\ASUS\Desktop\论文图表_可编辑矢量\Figure4`.
- Table 12: `C:\Users\ASUS\Desktop\论文图表_可编辑矢量\Table12`.
- Each folder contains PPTX, live-text SVG, vector PDF, PNG preview, manifest and source data.
