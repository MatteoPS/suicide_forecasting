# suicide_forecast

**Status:** active research, in progress. The spatiotemporal detection layer runs end to end on real data. The forecasting and ensemble layers are scaffolding. Component-level status is in the table below.

Detection, and eventually forecasting, of suicide clusters in the United States at ZIP Code Tabulation Area (ZCTA) resolution. Columbia University, Shaman Group.

## Question

Suicide clusters, meaning localized excesses consistent with social contagion, are well documented, but the detection literature works mostly at county level and coarse time aggregation. Two problems follow. County aggregation dilutes small clusters. And large regions of the US carry a chronically elevated baseline, so a naive scan keeps returning the same "suicide belt" instead of the transient events of interest.

This repo works at ZCTA resolution over 2010 to 2023 and separates the two signals explicitly: a chronic baseline layer, and a space-time excess layer measured against each location's own history.

## Status by component

| Component | Location | Status |
| --- | --- | --- |
| ETL, ZCTA boundary harmonization, Census / BRFSS / HCUP ingest | `src/etl/` | Working - real data |
| SaTScan input generation, coordinate recovery, missingness accounting | `src/geospatial/prep_satscan.py` | Working; dropped cases are logged to file, not silent |
| Regional subsetting of SaTScan artifacts | `src/geospatial/filter_satscan.py` | Working |
| SaTScan runs and parameter justification | `notebooks/1.1` | Five configurations run and interpreted; two more specified|
| ST-DBSCAN, rate-weighted, dense | `src/geospatial/st_dbscan.py`, `notebooks/1.2` | Implemented and running; parameters not calibrated (see Known issues) |
| Time series forecasting (ETS, AutoARIMA, LightGBM) | `notebooks/0.1` | Exploratory. No held-out evaluation yet; first passes were noise dominated |
| Ensemble and dynamical layers | `config/ensemble.yaml`, `config/dynamical.yaml` | Empty scaffolding |
| Tests | `tests/` | Not implemented |

## What the detection layer has shown

- A purely spatial scan with no distance cap returns continent-scale structure, including a cluster roughly 1,800 km across covering about 32% of the US population, sustained at RR 1.8 to 2.7 across the full period. That is chronic baseline, not contagion.
- Because of that, the space-time scan requires a nonparametric spatial adjustment (`SpatialAdjustmentType=2`) paired with a 12-month time-stratified baseline. With both applied, the geographic redundancy disappears and over 250 clusters reach p <= 0.05, distributed nationwide.
- Open limitation: the strongest clusters saturate the 150 km and 15-month caps inherited from county-level literature (Platt et al. 2022). Those caps need widening before cluster extents can be read at face value.

## Known issues

- **The ST-DBSCAN density threshold is currently non-binding.** Sample weights are incidence per 100,000 population. Since no ZCTA approaches 1M people, a single case already exceeds `min_threshold=0.1` on its own, so every record qualifies as a core point and the density criterion does no work. The NYC run returning 981 clusters from 3,830 records reflects this. The threshold needs raising by two to three orders of magnitude and re-tuning against the cluster size distribution.
- **Dense distance matrix.** `run_small_st_dbscan` builds an N x N matrix and is capped near 10,000 records. National scale needs a sparse or tiled rewrite.
- **Rate weighting favors low-population ZCTAs** by construction. That is intended, but it shapes which clusters surface and has not been quantified.

## Data

No data is in this repo. `data/` and `outputs/` are gitignored.

- NVDRS-RAD, restricted, requires a signed data use agreement with CDC
- HCUP SEDD and SID, restricted, accessed through Redivis
- Census ACS 5-year and BRFSS, public
  
*"The National Violent Death Reporting System (NVDRS) is administered by the Centers for Disease Control and Prevention (CDC) by participating NVDRS jurisdictions. The findings and conclusions of this study are those of the authors alone and do not necessarily represent the official position of the CDC or of participating NVDRS jurisdictions."*

## Setup

```bash
conda env create -f environment.yaml
conda activate suicide_forecast
```

`.env` requires `CENSUS_API_KEY`, `REDIVIS_USERNAME`, `REDIVIS_ORGANIZATION`, and `BRFSS_PATH`.

SaTScan is a separate install. This repo generates its input files and interprets its output, it does not vendor the binary.

## Scope

This is population-level surveillance methodology. Nothing here operates on, or is intended for, individual-level risk prediction, and the restricted datasets it depends on come with agreements barring re-identification.
