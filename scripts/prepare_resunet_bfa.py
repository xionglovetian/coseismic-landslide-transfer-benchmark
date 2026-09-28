import argparse, csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def convert_one(item):
    image_path, mask_path, out_image, out_mask = item
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    mask_arr = (np.array(mask) > 0).astype(np.uint8) * 255
    image.save(out_image, format='PNG', optimize=False)
    Image.fromarray(mask_arr, mode='L').save(out_mask, format='PNG', optimize=False)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    output = Path(args.output)
    items = []
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            split = row['split']
            sid = row['id']
            img_dir = output / split / 'images'
            mask_dir = output / split / 'masks'
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            items.append((
                row['image_path'],
                row['mask_path'],
                str(img_dir / f'{sid}.png'),
                str(mask_dir / f'{sid}.png'),
            ))
    print(f'Converting {len(items)} image-mask pairs to {output}')
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures)):
            done += future.result()
    print(f'Converted {done} pairs')


if __name__ == '__main__':
    main()
