import csv, json, statistics
from pathlib import Path

import os as _repo_os
project = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
outputs = project / 'outputs'
reports = project / 'reports'
seeds = [42, 2026, 777]
metrics_names = ['iou', 'dice', 'f1', 'precision', 'recall']
rows = []
for seed in seeds:
    path = outputs / f'group_asunet_seed{seed}' / 'metrics.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding='utf-8'))
    row = {'seed': seed, 'best_epoch': data['best_epoch']}
    for split_key, prefix in [('cas_val', 'cas_val'), ('resunet_bfa_external_test', 'external')]:
        for metric in metrics_names:
            row[f'{prefix}_{metric}'] = data[split_key][metric]
    rows.append(row)

if not rows:
    raise SystemExit('No completed seed metrics found')

fieldnames = list(rows[0].keys())
with (reports / 'group_asunet_seed_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

summary_rows = []
for prefix in ['cas_val', 'external']:
    for metric in metrics_names:
        values = [row[f'{prefix}_{metric}'] for row in rows]
        summary_rows.append({
            'split': prefix,
            'metric': metric,
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'n': len(values),
        })

with (reports / 'group_asunet_seed_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['split', 'metric', 'mean', 'std', 'n'])
    writer.writeheader()
    writer.writerows(summary_rows)

lines = ['# AS_UNet Group-Split Multi-Seed Summary', '', f'Completed seeds: {[r["seed"] for r in rows]}', '', '| Split | Metric | Mean | Std |', '|---|---|---:|---:|']
for row in summary_rows:
    lines.append(f'| {row["split"]} | {row["metric"]} | {row["mean"]:.4f} | {row["std"]:.4f} |')
(reports / 'group_asunet_seed_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('wrote', reports / 'group_asunet_seed_summary.md')
