# Figure generation

Scripts that produce the nine manuscript figures. Outputs (SVG with live text, PDF and a
PNG preview) are kept with the manuscript; this directory holds the generators so the
figures can be rebuilt and audited.

| Manuscript figure | Script | Notes |
|---|---|---|
| Figure 1 - study area | `figure1_study_area.py` | matplotlib + pyshp + Natural Earth 1:10m/1:110m country polygons; basemap shapefiles are not redistributed here. |
| Figure 2 - experimental design | `figure2_design_schematic.py` | Three dashed stage frames, native vector shapes, no raster. Writes SVG, PPTX, PDF and a PNG preview. |
| Figure 3 - zero-shot by region | `build_publication_figures.py` (`figure3`) | Shared figure-level legend placed outside the axes. |
| Figure 4 - exposure-matched contrast and resolution arms | `figure4_seed_stratified.py` | Reads `q2_x3_component_summary.csv` and `q2_x3_seed_stratified_bootstrap/q2_x3_seed_stratified_t_interval.csv`. |
| Figure 5 - source fit versus transfer | `build_publication_figures.py` (`figure5`) | Panels c-d are a single-run ResUNet seed-42 ablation. |
| Figure 6 - few-shot adaptation | `build_publication_figures.py` (`figure6`) | Panels a-d share one legend outside the axes. |
| Figure 7 - error examples | `figure7_error_examples.py` | Assembles 20 independent raster tiles (4 rows x 5 columns) inside a vector grid. The error maps are recoloured to a colour-blind-safe scheme (light grey TP, dark blue FP, sky blue FN). |
| Figure 8 - confidence calibration | `build_publication_figures.py` (`figure7`) | Bubble-size note lives in the caption, not in the axes. |
| Figure 9 - prevalence sensitivity | `build_publication_figures.py` (`figure8`) | Two panels share one x axis; no regression line is fitted. |

## Inputs

The generators read packaged metrics from this repository and the local experiment
tree:

- `build_publication_figures.py` reads `D:\landslide_unet_project\reports\*.csv`
  (zero-shot summaries, exposure-matched bootstrap, epoch-wise per-run, source-fraction
  curves, few-shot macro/region summaries, E4 query/support/stitched summaries,
  prevalence and threshold metrics). Set `ROOT` at the top of the script to relocate.
- `figure4_seed_stratified.py` reads `q2_x3_component_summary.csv` and
  `q2_x3_seed_stratified_bootstrap/q2_x3_seed_stratified_t_interval.csv`.
- `figure7_error_examples.py` expects the panel PNGs under `figure7_panels/` plus
  `figure7_panels.json`; the panels themselves are image data and are not redistributed.

No manuscript prose, source imagery, processed chips or checkpoints are included.
