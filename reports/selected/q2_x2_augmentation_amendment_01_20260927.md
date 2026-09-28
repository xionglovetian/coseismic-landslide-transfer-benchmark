# Formal Amendment 01: X2 Architecture Set

Date: 2026-09-27
Supersedes: q2-major-x2-aug-20260927-v1
Effective protocol: q2-major-x2-aug-20260927-v2

## Reason for amendment

The initial frozen protocol specified ResUNet + SegFormer-B0 for X2. That was
an experimenter error: the attached Q2 revision plan explicitly specifies
Bottleneck-LiteASK + ResUNet for the X2 augmentation decision, while
ResUNet + SegFormer-B0 belongs to the X1 resolution experiment.

## Timeline and deviation

- 10:08: X2 protocol v1 was created with ResUNet + SegFormer-B0.
- 10:10-13:13: the ResUNet X2 grid ran to completion (9/9 runs).
- 13:13: the extra SegFormer X2 child and parent were stopped before any
  SegFormer raw result was created.
- 13:14: Bottleneck X2 work started without a formal amendment.
- 13:49: the protocol deviation was identified. The Bottleneck queue and v2
  master were stopped immediately.

## Amendment decision

The primary X2 architecture set is changed to:

- ResUNet
- Bottleneck-LiteASK

The design remains 2 architectures x 3 seeds x 3 augmentation arms = 18 runs.
SegFormer-B0 is removed from X2 and remains reserved for X1 as specified by the
attached revision plan.

## Data disposition

The nine completed ResUNet runs were generated under the v1 protocol and are
retained because ResUNet is common to v1 and v2. Their hashes are recorded
below. No target-domain result was used to make the augmentation decision.

The pre-amendment Bottleneck current/seed42 result and the partial Bottleneck
none/seed42 run are quarantined and are not part of the v2 grid. All Bottleneck
runs used by v2 will be rerun from scratch after this amendment.

Quarantined material:

- reports/_protocol_deviation_quarantine_20260927_x2/q2_x2_aug_current_bottleneckliteask_seed42.json
- outputs/_protocol_deviation_quarantine_20260927_x2/q2_x2_aug_current_bottleneckliteask_seed42/
- outputs/_protocol_deviation_quarantine_20260927_x2/q2_x2_aug_none_bottleneckliteask_seed42/

## Retained ResUNet raw-data hashes (SHA256)

- 7B1BB906744EB6FF9ACBD6971BD74AD35D9B2A0A4BCCC28103162B3A82750338  q2_x2_aug_current_resunet_seed2026.json
- CBDB4DBE31A8AA973E41CF969AE6267C5353BC2F0157CAF229E48ECA8F4A1B6B  q2_x2_aug_current_resunet_seed42.json
- F28227D30F99C2F6DD68FB3612F119F32EFDF4832BA64A044B8B2F58265A3952  q2_x2_aug_current_resunet_seed777.json
- 44C7BBAD1B31AFCCABD78FD15BD74048C51460EE628E1D4AF3C52E32B01F0B48  q2_x2_aug_none_resunet_seed2026.json
- 0A3CB418F1BFEB3952F158642D5CB7F447AC857343DCD4733BCC9836F3BF3C3F  q2_x2_aug_none_resunet_seed42.json
- 42ABCC55BDF53A015DB5F3A0E4CCD2E8A371D734EC971F239AB7631F8FBC08D2  q2_x2_aug_none_resunet_seed777.json
- 2023B2E9056F85A816F39A96EEA527F5EEB6061D31048B95285A52773C14EF97  q2_x2_aug_strong_resunet_seed2026.json
- CF9C27E4A4542F9C24BF605DE35AF0A898DA52CCCA481B2F953448F4C91D4D9C  q2_x2_aug_strong_resunet_seed42.json
- 23EB8B8FCB3725F2C6B089C342E0F02BE7BBB12DFA0E2816EA6D246183B07297  q2_x2_aug_strong_resunet_seed777.json

## Quarantined pre-amendment result hash

- 51C196EE595302671B28748EEE94CBC7C86F0E8145506045107A96C7668AF3CB  q2_x2_aug_current_bottleneckliteask_seed42.json

## Analysis rules after amendment

1. Only v2 Bottleneck runs created after this amendment may enter the X2 grid.
2. The primary augmentation decision uses only source-validation IoU from the
   ResUNet + Bottleneck model-seed pairs.
3. Six-region target metrics remain post-selection evidence and cannot be used
   to reverse the primary configuration decision.
4. All result tables derived from this grid must cite q2-major-x2-aug-20260927-v2.
