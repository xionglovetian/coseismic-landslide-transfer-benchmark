import csv, os, shutil
from pathlib import Path

import os as _repo_os
project = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
manifest_path = project / 'data' / 'splits' / 'resunet_bfa_manifest.csv'
source_root = project / 'data' / 'processed' / 'resunet_bfa_224'
output_root = project / 'data' / 'processed' / 'resunet_bfa_grouped_224'
group_manifest = project / 'data' / 'splits' / 'resunet_bfa_group_manifest.csv'

VAL_PREFIX = 'c2'
TEST_PREFIX = 'c3'

rows = list(csv.DictReader(manifest_path.open(encoding='utf-8-sig', newline='')))
out_rows = []
for row in rows:
    prefix = row['region']
    split = 'test' if prefix == TEST_PREFIX else ('val' if prefix == VAL_PREFIX else 'train')
    src_img = source_root / row['split'] / 'images' / f"{row['id']}.png"
    src_mask = source_root / row['split'] / 'masks' / f"{row['id']}.png"
    if not src_img.exists() or not src_mask.exists():
        raise FileNotFoundError((src_img, src_mask))
    dst_img = output_root / split / 'images' / f"{row['id']}.png"
    dst_mask = output_root / split / 'masks' / f"{row['id']}.png"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    for src, dst in ((src_img, dst_img), (src_mask, dst_mask)):
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    out_rows.append({
        'dataset': 'ResUNet_BFA_grouped',
        'split': split,
        'id': row['id'],
        'image_path': str(dst_img),
        'mask_path': str(dst_mask),
        'region': prefix,
        'augmented': row['augmented'],
    })

with group_manifest.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    writer.writeheader()
    writer.writerows(out_rows)

for split in ('train', 'val', 'test'):
    ids = [r for r in out_rows if r['split'] == split]
    print(split, 'pairs', len(ids), 'groups', sorted({r['region'] for r in ids}))
print('manifest', group_manifest)
