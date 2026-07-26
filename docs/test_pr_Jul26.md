## **386 passing, 34 skipped, 2 xfail, ~4 seconds.**

```bash
pytest
```

## No test touches a real dataset

NVDRS and BRFSS are large and restricted, so every fixture builds a handful of synthetic rows with the same *shape* as the real files. The Census API, Redivis, `uszipcode` and `pd.read_sas` are stubbed at the boundary. `tests/conftest.py` installs placeholder credentials before anything under `src` is imported, so a real `.env` sitting next to the repo can never leak a live API key into a test run.

## What the tests found

Ranked by how quietly they do damage.

**1. `aggregate_nvdrs_daily(df, geo_level='state')` returns county columns.** It picks the pivot column from `geo_col` (default `'DeathFIPS'`), not from `geo_level` — unlike `aggregate_nvdrs_monthly` and `aggregate_nvdrs_daily_injury`, which both branch on `geo_level`. Notebook 0.5 guarded with `if state in df.columns`, so **the NVDRS series was silently absent from every panel** of the WONDER comparison plot. Notebook 0.1 would have raised `KeyError`. Recorded as a strict xfail.

**2. `filter_nvdrs_suicides` undercounts multiple-suicide incidents.** The multi-person branch uses `idxmax()` on `PersonType == 'Both victim and suspect'`. A `Multiple suicides` incident carries no such label, so `idxmax()` falls back to the first row and the other victims are dropped. Strict xfail.

**3. Notebook 0.5's injury-date branch would crash.** NVDRS loads with `dtype=str` and `filter_nvdrs_suicides` converts only `DeathDate`, so `InjuryDate` stays a string and `.resample('ME')` raises `TypeError`. The new `resample_counts` coerces its index, so that path works now.

**4. `harmonize_zcta_boundaries` silently drops ZIPs missing from the crosswalk** (inner join). A test pins a case where 900 people disappear unreported — relevant when reconciling national population totals.

**5. Format gap between pipeline stages 2 and 3.** `create_regional_satscan_files` writes headed CSVs; `run_small_st_dbscan` reads headerless whitespace files. Nothing in the repo bridges them — it's a manual step today. Documented in `test_data_pipeline.py`.

Smaller: `prep_satscan_gui()` from `__main__` writes `None_full_cas.csv`; `clean_hcup_missing_codes` stringifies NaN identifiers to `'nan'` so rows with a missing HOSPID merge to each other; `src/utils/config.py` raises at *import* time on missing credentials, which makes `src` unimportable in CI; plotly 6 deprecated `scatter_mapbox`.

## Code moved out of notebooks

| From                                                                   | To                                                                      |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 0.1 — darts TimeSeries construction ×3                               | `src/etl/features.py` — `build_nvdrs_frames`, `to_darts_dict`    |
| 0.1 —`plot_counties`, forecast subplot grid                         | `src/utils/viz.py` — `plot_grouped_series`, `plot_forecast_grid` |
| 0.1 — ETS/ARIMA/LGBM fit + pickle cache loop                          | `src/statistical/baselines.py`                                        |
| 0.4 —`smart_merge_ahal`, HCUP missing-code cleaning                 | `src/etl/transform.py`                                                |
| 0.5 — WONDER load + pivot                                             | `src/etl/ingest.py` — `load_wonder_state`                          |
| 0.5 — NVDRS-vs-WONDER grid                                            | `src/utils/viz.py` — `plot_source_comparison`                      |
| 1.2 +`st_dbscan.py` — duplicated `.cas`/`.geo`/`.pop` readers | `src/geospatial/satscan_io.py`                                        |

HCUP behaviour is unchanged from the notebook version, warts included — the `'nan'` stringification is now documented and tested rather than fixed, since changing it would change analysis results.

## Reviewer notes

**Two behaviour changes worth confirming:**

- Notebooks 0.1 and 0.5 now pass `geo_col='DeathState'` explicitly, so they finally produce the state columns their downstream code always assumed. **The NVDRS-vs-WONDER plot will look different, because it was drawing nothing before.**
- `plot_grouped_series` and `plot_source_comparison` now print which columns they skipped instead of dropping them silently.

The four model-layer test files keep the originally intended test names, but as real skipped tests with reasons rather than comments — `pytest -m notimplemented` lists them. The config-scaffolding assertions in each (section names, prior names, sampling keys) run today and will catch a renamed key before the model code lands.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
