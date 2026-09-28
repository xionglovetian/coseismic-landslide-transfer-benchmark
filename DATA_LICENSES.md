# Data and Code Licensing

## Code

The code and documentation in this repository are licensed under the MIT
License. See `LICENSE`.

## Excluded image data and annotations

This repository does not redistribute source imagery, target imagery, processed
image chips, annotation rasters, or model checkpoints.

The benchmark manifest records mixed provider-specific terms. Users must obtain
data from the original providers and follow the most restrictive applicable
terms. In particular:

- CAS Landslide Dataset imagery: CC BY-NC 4.0 and provider terms; no
  redistribution in this repository.
- Lombok and Palu imagery: CC BY-NC 4.0; no redistribution in this repository.
- Longxi River, Hokkaido/Iburi-Tobu, Wenchuan, Jiuzhai Valley and other source
  material: provider-specific terms, including derivative-works, attribution,
  USGS LP DAAC, or original-distribution conditions.
- Source and target manifests are retained only as metadata, split records,
  identifiers and checksums. They are not a license to redistribute the
  referenced pixels.

## Excluded checkpoints

Model checkpoints are derived artifacts that may inherit restrictions from the
training data. They are not published here. Any future checkpoint release
should state the applicable license, exclude restricted imagery, and be archived
separately with a DOI.

## Attribution

Users should cite the original datasets and the accompanying manuscript when it
is published. Provider names and authorization fields are present in
`data_manifests/manifest_benchmark_v2.csv`.
