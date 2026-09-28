# Data Manifests

These frozen CSVs define sample identities, region labels, split assignments,
spatial groups, source metadata and path/checksum fields.

The `image_path` and `mask_path` columns retain the reference workstation paths
for auditability. They are not part of the data distribution. To use another
root without modifying the frozen files, run:

```powershell
python scripts/checks/rebase_manifest_paths.py `
  --input data_manifests/manifest_benchmark_v2.csv `
  --output data/processed/benchmark_v2_regions_512/manifest_benchmark_v2.csv `
  --old-root "D:/landslide_unet_project" `
  --new-root $env:LANDSLIDE_PROJECT_ROOT
```

The repository does not grant rights to redistribute the images referenced by
these manifests. See `../DATA_LICENSES.md`.
