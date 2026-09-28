# A4 Physical Buffer Audit

Date: 2026-09-27

The 256 m and 512 m rows are identical because distances on the reconstructed stride-256 grid are discrete.

- Hokkaido: minimum edge gap 0 m; next admissible gap at least 1,536 m.
- Lombok and Palu: minimum edge gap 0 m; next admissible gap at least 2,560 m.

Consequently, all tested buffers above 0 m and below the next discrete separation retain the same query set. The identical rows are not a copy-paste error.

## Action

Table 11 caption was updated.
