import csv, json
from pathlib import Path
import os as _repo_os
project = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve(); reports=project/'reports'
runs={
 'CAS-only zero-shot': None,
 'Uniform multi-source': project/'outputs'/'dg_longxi_uniform_noalign_seed42'/'target_metrics.json',
 'Balanced multi-source': project/'outputs'/'dg_longxi_balanced_noalign_seed42'/'target_metrics.json',
 'Balanced + feature alignment': project/'outputs'/'dg_longxi_balanced_mmd_seed42'/'all_region_metrics.json',
}
baseline=json.loads((reports/'external_regions_raw'/'group_bottleneckliteaskunetpp_seed42.json').read_text(encoding='utf-8'))['groups']['longxi_river']
rows=[]
base_iou=baseline['threshold_metrics']['0.5']['iou']; rows.append({'method':'CAS-only zero-shot','longxi_iou':base_iou,'longxi_dice':baseline['threshold_metrics']['0.5']['dice'],'bf1_2':baseline['f1_2_mean'],'hd95':baseline['hd95_mean'],'source_val_iou':None})
for name,path in list(runs.items())[1:]:
 d=json.loads(path.read_text(encoding='utf-8'))
 g=d['target']['longxi_river'] if 'target' in d else d['groups']['longxi_river']
 rows.append({'method':name,'longxi_iou':g['threshold_metrics']['0.5']['iou'],'longxi_dice':g['threshold_metrics']['0.5']['dice'],'bf1_2':g['f1_2_mean'],'hd95':g['hd95_mean'],'source_val_iou':d['source_validation']['iou']})
with (reports/'multisource_dg_pilot_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
lines=['# Multi-Source Domain Generalization Pilot','','Date: 2026-09-22','','Target: Longxi River (never used for training or checkpoint selection).','','| Method | Longxi IoU | Dice | BF1@2 | HD95 | Source Val IoU |','|---|---:|---:|---:|---:|---:|']
for r in rows:
    source_value = "" if r["source_val_iou"] is None else f'{r["source_val_iou"]:.4f}'
    lines.append(f'| {r["method"]} | {r["longxi_iou"]:.4f} | {r["longxi_dice"]:.4f} | {r["bf1_2"]:.4f} | {r["hd95"]:.2f} | {source_value} |')
lines += ['','## Decision','','- Uniform multi-source training is the strongest pilot: Longxi IoU 0.1759 versus 0.1135 zero-shot.','- Region-balanced sampling reduces Longxi IoU to 0.1554.','- Adding feature-statistics alignment reduces it further to 0.1505.','- The observed gain therefore comes from source-domain expansion, not from the proposed balancing or alignment components.','- The method hypothesis fails the first gate; the project should pivot to a benchmark/application paper.','','Files:','- D:\\landslide_unet_project\\reports\\multisource_dg_pilot_results.csv','- D:\\landslide_unet_project\\outputs\\dg_longxi_uniform_noalign_seed42','- D:\\landslide_unet_project\\outputs\\dg_longxi_balanced_noalign_seed42','- D:\\landslide_unet_project\\outputs\\dg_longxi_balanced_mmd_seed42']
(reports/'multisource_dg_pilot_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('wrote',reports/'multisource_dg_pilot_report.md')
