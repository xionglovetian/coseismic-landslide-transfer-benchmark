import csv, json, statistics
from pathlib import Path
import os as _repo_os
project = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve(); reports=project/'reports'
regions=['longxi_river','wenchuan','jiuzhai_valley','moxitaidi']
models=[('UNet','group_unet'),('Bottleneck-LiteASK','group_bottleneckliteaskunetpp'),('Bottleneck-LiteASK','group_bottleneckliteaskunetpp')]
rows=[]
# Full AdaBN for both models.
for label,prefix in [('UNet','group_unet'),('Bottleneck-LiteASK','group_bottleneckliteaskunetpp')]:
 std=json.loads((reports/'external_regions_raw'/f'{prefix}_seed42.json').read_text(encoding='utf-8'))
 for region in regions:
  s=std['groups'][region]['threshold_metrics']['0.5']['iou']
  p=reports/'external_adabn_raw'/f'{prefix}_seed42_{region}.json'
  a=json.loads(p.read_text(encoding='utf-8'))['groups'][region]['threshold_metrics']['0.5']['iou']
  rows.append({'model':label,'method':'AdaBN full','region':region,'standard_iou':s,'adapted_iou':a,'delta':a-s})
# Alpha interpolation for candidate.
prefix='group_bottleneckliteaskunetpp'
std=json.loads((reports/'external_regions_raw'/f'{prefix}_seed42.json').read_text(encoding='utf-8'))
for alpha in ['0.10','0.25','0.50']:
 for region in regions:
  s=std['groups'][region]['threshold_metrics']['0.5']['iou']
  p=reports/'external_adabn_raw'/f'{prefix}_seed42_{region}_alpha{alpha}.json'
  a=json.loads(p.read_text(encoding='utf-8'))['groups'][region]['threshold_metrics']['0.5']['iou']
  rows.append({'model':'Bottleneck-LiteASK','method':f'AdaBN alpha={alpha}','region':region,'standard_iou':s,'adapted_iou':a,'delta':a-s})
with (reports/'adabn_pilot_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
lines=['# AdaBN Pilot Results','','Date: 2026-09-22','','## Full AdaBN','','| Model | Macro standard | Macro adapted | Delta | Longxi | Wenchuan | Jiuzhai | Moxitaidi |','|---|---:|---:|---:|---:|---:|---:|---:|']
for label in ['UNet','Bottleneck-LiteASK']:
 sub=[r for r in rows if r['model']==label and r['method']=='AdaBN full']
 sm=statistics.mean(r['standard_iou'] for r in sub); am=statistics.mean(r['adapted_iou'] for r in sub)
 vals={r['region']:r['delta'] for r in sub}
 lines.append(f'| {label} | {sm:.4f} | {am:.4f} | {am-sm:+.4f} | {vals["longxi_river"]:+.4f} | {vals["wenchuan"]:+.4f} | {vals["jiuzhai_valley"]:+.4f} | {vals["moxitaidi"]:+.4f} |')
lines += ['','## Bottleneck-LiteASK Alpha Sweep','','| Method | Macro adapted | Delta vs standard |','|---|---:|---:|']
base=statistics.mean(r['standard_iou'] for r in rows if r['model']=='Bottleneck-LiteASK' and r['method']=='AdaBN full')
for alpha in ['0.10','0.25','0.50']:
 sub=[r for r in rows if r['model']=='Bottleneck-LiteASK' and r['method']==f'AdaBN alpha={alpha}']
 am=statistics.mean(r['adapted_iou'] for r in sub)
 lines.append(f'| alpha={alpha} | {am:.4f} | {am-base:+.4f} |')
lines += ['','## Decision','','- Full AdaBN strongly improves Longxi but collapses Wenchuan and degrades Jiuzhai/Moxitaidi.','- Interpolating source and target statistics from alpha=0.1 to 0.5 does not recover a balanced improvement.','- Global BatchNorm-statistic adaptation is rejected as the main method.','- A method paper would require selective or confidence-gated adaptation, not global AdaBN.','','File: D:\\landslide_unet_project\\reports\\adabn_pilot_results.csv']
(reports/'adabn_pilot_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('wrote',reports/'adabn_pilot_report.md')
