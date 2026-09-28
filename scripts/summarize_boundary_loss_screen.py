import csv, json, statistics
from pathlib import Path
import os as _repo_os
project = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
reports=project/'reports'
regions=['a1','a2','a3','b','c1','c2','c3']
runs={
 'Standard': 'group_bottleneckliteaskunetpp_seed42',
 'BoundaryAware': 'group_bottleneckliteaskunetpp_boundary_seed42',
}
rows=[]
for label,prefix in runs.items():
    metrics=json.loads((project/'outputs'/f'{prefix}'/'metrics.json').read_text(encoding='utf-8'))
    cross=json.loads((reports/'cross_group_raw'/f'{prefix}.json').read_text(encoding='utf-8'))
    boundary=json.loads((reports/'boundary_metrics_raw'/f'{prefix}.json').read_text(encoding='utf-8'))
    rows.append({
      'variant':label,
      'loss':metrics['loss'],
      'cas_iou':metrics['cas_val']['iou'],
      'c3_iou':metrics['resunet_bfa_external_test']['iou'],
      'macro_all_iou':statistics.mean(cross['groups']['all'][r]['iou'] for r in regions),
      'macro_original_iou':statistics.mean(cross['groups']['original'][r]['iou'] for r in regions),
      'bf1_2':statistics.mean(boundary['groups'][r]['f1_2_mean'] for r in regions),
      'bf1_4':statistics.mean(boundary['groups'][r]['f1_4_mean'] for r in regions),
      'hd95':statistics.mean(boundary['groups'][r]['hd95_mean'] for r in regions),
    })
fields=list(rows[0].keys())
with (reports/'boundary_loss_screening.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
std=rows[0]; boundary=rows[1]
lines=[
 '# Boundary-Aware Loss Screening',
 '',
 'Date: 2026-09-21',
 '',
 '| Variant | CAS IoU | c3 IoU | Macro IoU all | Macro IoU original | BF1@2 | BF1@4 | HD95 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|',
]
for row in rows:
    lines.append(f'| {row["variant"]} | {row["cas_iou"]:.4f} | {row["c3_iou"]:.4f} | {row["macro_all_iou"]:.4f} | {row["macro_original_iou"]:.4f} | {row["bf1_2"]:.4f} | {row["bf1_4"]:.4f} | {row["hd95"]:.2f} |')
lines += [
 '',
 '## Decision',
 '',
 '- Boundary-aware loss improves macro BF1@2 by '+f'{boundary["bf1_2"]-std["bf1_2"]:+.4f}'+' and lowers HD95 by '+f'{boundary["hd95"]-std["hd95"]:+.2f}'+' pixels.',
 '- However, it lowers CAS IoU by '+f'{boundary["cas_iou"]-std["cas_iou"]:+.4f}'+', c3 IoU by '+f'{boundary["c3_iou"]-std["c3_iou"]:+.4f}'+', and Macro IoU by '+f'{boundary["macro_all_iou"]-std["macro_all_iou"]:+.4f}'+'.',
 '- The loss is rejected for the main model and was not expanded to five seeds.',
 '- The standard PolyGHMDiceLoss remains the main protocol.',
 '',
 '## Files',
 '',
 '- Screening table: D:\\landslide_unet_project\\reports\\boundary_loss_screening.csv',
 '- Standard checkpoint: D:\\landslide_unet_project\\outputs\\group_bottleneckliteaskunetpp_seed42',
 '- Boundary checkpoint: D:\\landslide_unet_project\\outputs\\group_bottleneckliteaskunetpp_boundary_seed42',
]
(reports/'boundary_loss_screening_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('wrote',reports/'boundary_loss_screening_report.md')
