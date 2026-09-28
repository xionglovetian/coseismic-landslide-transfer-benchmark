# E0.4: Physical Support-Query Separation Audit

Distances use the reconstructed stride-256 grid plus per-chip ground sample distance from the CAS manifest. `edge gap` is the distance between axis-aligned 512-pixel chip footprints; 0 m means the chips touch but do not overlap.

| Seed | Region | Min grid Chebyshev | Min center distance (m) | Min edge gap (m) | GSD range (m) |
|---:|---|---:|---:|---:|---:|
| 42 | hokkaido_iburi_tobu | 2 | 1536.0 | 0.0 | 3.000-3.000 |
| 2026 | hokkaido_iburi_tobu | 2 | 1536.0 | 0.0 | 3.000-3.000 |
| 777 | hokkaido_iburi_tobu | 2 | 1536.0 | 0.0 | 3.000-3.000 |
| 42 | lombok | 2 | 2560.0 | 0.0 | 5.000-5.000 |
| 2026 | lombok | 2 | 2560.0 | 0.0 | 5.000-5.000 |
| 777 | lombok | 2 | 2560.0 | 0.0 | 5.000-5.000 |
| 42 | palu | 2 | 2560.0 | 0.0 | 5.000-5.000 |
| 2026 | palu | 2 | 2560.0 | 0.0 | 5.000-5.000 |
| 777 | palu | 2 | 2560.0 | 0.0 | 5.000-5.000 |

## Query Retention Under Physical Buffers

| Seed | Region | Current query | Buffer 0 m | Buffer 256 m | Buffer 512 m | Buffer 1000 m |
|---:|---|---:|---:|---:|---:|---:|
| 42 | hokkaido_iburi_tobu | 1341 | 1.000 | 0.858 | 0.858 | 0.714 |
| 2026 | hokkaido_iburi_tobu | 1346 | 1.000 | 0.856 | 0.856 | 0.699 |
| 777 | hokkaido_iburi_tobu | 1345 | 1.000 | 0.856 | 0.856 | 0.706 |
| 42 | lombok | 327 | 1.000 | 0.654 | 0.654 | 0.654 |
| 2026 | lombok | 323 | 1.000 | 0.607 | 0.607 | 0.607 |
| 777 | lombok | 330 | 1.000 | 0.658 | 0.658 | 0.658 |
| 42 | palu | 734 | 1.000 | 0.928 | 0.928 | 0.928 |
| 2026 | palu | 730 | 1.000 | 0.922 | 0.922 | 0.922 |
| 777 | palu | 730 | 1.000 | 0.934 | 0.934 | 0.934 |

## Conclusion

The current split guarantees non-overlap, but it does not guarantee a physical buffer: the minimum edge gap can be zero because adjacent stride-256 chips touch. The source train/validation split cannot be assigned physical distances from the current PNG manifest because source geolocation is absent. A revised few-shot split should use an explicit GSD-aware buffer, and the source split should be rebuilt from georeferenced source imagery if available.
