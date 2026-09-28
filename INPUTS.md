# Input Data and Exact Versions

## 1. Training-source data

The source-domain images and masks are contained in the public AS-UNet
repository.

- Repository: https://github.com/Youhaoran1512/AS-UNet
- Pinned commit: `1df78b3ea315fc9744db6de9f2917f9f309c7218`
- Dataset directory: `inputs/data_sum_moxizhen+bijie`
- Expected counts: 1,795 training images, 1,795 training masks, 513
  validation images, 513 validation masks.

Clone or verify it with:

```powershell
.\scripts\setup\download_inputs.ps1 -ProjectRoot (Get-Location).Path
```

The script fetches exactly the pinned commit rather than the moving `master`
branch.

## 2. Target benchmark imagery

The target imagery and masks are distributed through the CAS Landslide
Dataset:

- DOI: https://doi.org/10.5281/zenodo.10294997
- Zenodo record: `10294997`
- License: CC BY-NC 4.0, subject to the original provider terms.

The reproducibility setup downloads and MD5-verifies these exact archives:

| Region | Zenodo file |
|---|---|
| Hokkaido Iburi-Tobu | `Hokkaido Iburi-Tobu.zip` |
| Lombok | `Lombok.zip` |
| Palu | `palu.zip` |
| Wenchuan | `Wenchuan.zip` |
| Jiuzhai Valley | `Jiuzhai valley (UAV-0.2m).zip` |
| Moxitaidi | `Moxitaidi (UAV-0.6m).zip` |
| Longxi River | `Longxi River（UAV）.zip` |

The script reads the MD5 checksum and byte size directly from the Zenodo API
and rejects a downloaded file if either differs.

## 3. Processing sequence

After `download_inputs.ps1`, run:

```powershell
.\scripts\setup\prepare_primary_data.ps1 `
  -ProjectRoot (Get-Location).Path `
  -Python C:\path\to\python.exe
```

This performs the frozen sequence:

1. Convert four source archives into `external_regions_512`.
2. Convert Hokkaido, Lombok and Palu into `benchmark_v2_regions_512`.
3. Build `target_spatial_index.csv` from overlapping processed chips.
4. Generate the seed-42 and seed-2026/777 support-query splits.
5. Verify file counts and SHA-256 values recorded in the benchmark manifest.

The expected target counts are fixed as follows:

| Region | Count |
|---|---:|
| Hokkaido Iburi-Tobu | 1,484 |
| Jiuzhai Valley | 5,925 |
| Lombok | 436 |
| Longxi River | 2,504 |
| Moxitaidi | 980 |
| Palu | 817 |
| Wenchuan | 178 |
| Total | 12,324 |

No image data or derived rasters are redistributed by this repository.