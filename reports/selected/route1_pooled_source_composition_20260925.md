# Pooled-Source Composition and Sampling

Protocol: `route1-q2-freeze-v1-20260925`

All three E1 pooled seeds use the same source composition. The sampler assigns sample weight `1 / n_domain`, so every domain has equal expected sampling mass and the expected exposure budget is divided equally across the five domain slots.

| Treatment | Domain | Role | Samples | Domain mass | Expected exposures | Sample weight |
|---|---|---|---:|---:|---:|---:|
| pooled | cas | source-only continuation | 1795 | 0.200000 | 7168 | 1 / 1795 |
| pooled | longxi_river | added source region | 2504 | 0.200000 | 7168 | 1 / 2504 |
| pooled | wenchuan | added source region | 178 | 0.200000 | 7168 | 1 / 178 |
| pooled | jiuzhai_valley | added source region | 5925 | 0.200000 | 7168 | 1 / 5925 |
| pooled | moxitaidi | added source region | 980 | 0.200000 | 7168 | 1 / 980 |
| pooled | TOTAL |  | 11382 | 1 | 35840 |  |
| cas-multistream | cas_replay_0 | replay placebo | 1795 | 0.200000 | 7168 | 1 / 1795 |
| cas-multistream | cas_replay_1 | replay placebo | 1795 | 0.200000 | 7168 | 1 / 1795 |
| cas-multistream | cas_replay_2 | replay placebo | 1795 | 0.200000 | 7168 | 1 / 1795 |
| cas-multistream | cas_replay_3 | replay placebo | 1795 | 0.200000 | 7168 | 1 / 1795 |
| cas-multistream | cas_replay_4 | replay placebo | 1795 | 0.200000 | 7168 | 1 / 1795 |

The pooled treatment uses CAS plus Jiuzhai Valley, Longxi River, Moxitaidi and Wenchuan. Hokkaido, Lombok and Palu are held out from every pooled training run and are used only for evaluation.

Sampling implementation: `WeightedRandomSampler(weights=[1/n_domain], num_samples=1120*32, replacement=True)` in `scripts/train_exposure_matched.py`.
