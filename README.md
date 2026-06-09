## Repo Structure
```text
.
├── config/          # Model configs, hyperparameters, YAML files
├── data/ 
│   └── raw          # All data — this is gitignored
├── docs/            # Project documentation and data access instructions
├── notebooks/       # Exploratory and communicative work  
├── src/             
│   ├── dynamical/   # Dynamic model of suicide contagion       
│   ├── ensemble/    # Multi-model combination logic; weighted averaging; scoring rules
│   ├── ml/          # LightGBM, RF, XGBoost, CatBoost, NN training and evaluation      
│   ├── nlp/         # distilBERT pipelines; NVDRS-RAD narrative processing
│   └── statistical/ # Bayesian hierarchical models; exponetial smoothing; ARIMA, CAR   
└── test/            # Unit and integration tests for src/ modules
```

## Datasets

- ***SEDD*** — ED attempt data; State Emergency Department Databases (HCUP)
- ***SID*** — Inpatient attempt data; State Inpatient Databases (HCUP)
- ***NVDRS-RAD*** — Violent death registry
- ***Contextual*** — 76 contextual predictors: assembled from 26 data sources across 4 domains (demographic/environmental, socioeconomic/structural, social/community, individual-level risk factors); includes Census, BRFSS, and other public-use sources
- ***Narrative*** — NVDRS-RAD qualitative text narratives from original medical examiner, coroner, and law enforcement reports; provide critical context
- ***988 Lifeline*** — Lifeline call data; measures contact volume (calls, texts, and chats), center performance, and response efficiency

---
## Aims
**▸ Preliminary Work — Cluster Detection & Spatial Analysis**
| # | Analysis / Model Type | Datasets Required | Citations |
|---|---|---|---|
| 1 | **Spatiotemporal Scan Statistics** <br> *Space-time cluster detection; SaTScan-style methods* | SEDD, SID, NVDRS-RAD / Death certs, Census denominators | [2](https://doi.org/10.1016/j.jaac.2021.12.012) [5](https://doi.org/10.1007/S00127-019-01736-4) [72](https://doi.org/10.1016/j.ssmph.2025.101866) |
| 2 | **Growth Mixture Models** <br> *Trajectory-based clustering of county-level suicide rates* | NVDRS-RAD / Death certs, Census denominators | [4](https://doi.org/10.1093/aje/kwad205) |

**▸ Aim 1 — Forecasting System: Statistical, Machine Learning & Dynamical Approaches** 
| # | Analysis / Model Type | Datasets Required | Citations |
|---|---|---|---|
| 3 | **Bayesian Hierarchical Model / Conditional Autoregressive (CAR)** <br> *Statistical approach; spatiotemporal autocorrelation; county & state-level* | SEDD, SID, NVDRS-RAD (historical, lagged), 76 contextual predictors, Census / BRFSS, Daily weather / monthly unemployment (high-freq), 988 Lifeline | [3](https://doi.org/10.1016/S2468-2667(22)00290-0) [26](https://doi.org/10.1371/journal.pcbi.1010945) [75](https://doi.org/10.1371/journal.pone.0260931) *Prior use of conditional autoregressive models for suicide risk estimation* |
| 4 | **LightGBM** <br> *Gradient Boosting Decision Tree; ML approach; time-series forecasting of attempt counts* | SEDD, SID, 76 contextual predictors (ZIP, county, state scales), High-freq dynamic predictors (weather, unemployment) | [97](https://github.com/Microsoft/LightGBM) *Demonstrated strong performance in time series forecasting* |
| 5 | **Neural Networks (NN)** <br> *Deep learning; quantile-based probabilistic forecasting; non-linear patterns* | SEDD, SID, 76 contextual predictors, High-freq dynamic predictors | [98](https://doi.org/10.3390/make5010013) *Valuable for infectious disease forecasting, modeling non-linear relationships, and processing large volumes of data* |
| 6 | **Mathematical Contagion Model + (EAKF)** <br> *Dynamical systems approach; Bayesian data assimilation; real-time parameter optimization; epidemic-style forecasting* | SEDD (assimilated), SID (assimilated) | [25](https://doi.org/10.1126/sciadv.adq4074) *Our prior structure simulating suicide ideation* / [99](https://doi.org/10.1175/MWR-D-16-0106.1) *Inference algorithm* / [27](https://doi.org/10.1038/ncomms2837) [28](https://doi.org/10.1038/ncomms14592) [100](https://doi.org/10.1175/MWR-D-16-0106.1) [101](https://doi.org/10.1256/QJ.05.135) *Similar real-time weather and infectious disease forecasts* |
| 7 | **Multi-Model Ensemble** <br> *Combines outputs of statistical, LightGBM, NN, and dynamical models; corrects individual system biases* | Outputs of models #3–6 above, SEDD, SID, 76 contextual predictors | [102](https://doi.org/10.1371/journal.pcbi.1007486) *Accuracy of Flu ensemble forecasts* |

**▸ Aim 2 — Mechanistic Classification: Natural Language Processing**
| # | Analysis / Model Type | Datasets Required | Citations |
|---|---|---|---|
| 8 | **NLP: Topic Modeling & Supervised Text Classifiers** <br> *Extracts social isolation, precipitating events, contagion markers from death narratives; F1 ~0.86 reported* | NVDRS-RAD narrative text, NVDRS-RAD circumstances data | [46](https://arxiv.org/pdf/2506.15030) [103](https://doi.org/10.1038/s43856-024-00631-7) [105](https://doi.org/10.1093/jamia/ocad068) *Validating extraction of risk factors like social isolation and context* |
| 9 | **distilBERT / Compact Large Language Model** <br> *Transformer-based NLP; detects subtle contextual patterns; replicates annotations ~85%; identifies discrepancies in 38% of discordant cases* | NVDRS-RAD narrative text, NVDRS-RAD circumstances data | [47](https://doi.org/10.2196/68212) [104](https://arxiv.org/pdf/2508.18541v2) *Validating extraction of risk factors like social isolation and context* |

**▸ Aim 2 — Mechanistic Classification: Machine Learning Ensemble**
| # | Analysis / Model Type | Datasets Required | Citations |
|---|---|---|---|
| 10 | **Random Forest (RF)** <br> *Breiman's bootstrap aggregation; 10-fold cross-validation; ROC-AUC hyperparameter selection; classifies cluster mechanism types* | NVDRS-RAD (circumstances + NLP-extracted features), 76 contextual predictors (county-level linkage) | [52](https://doi.org/10.1016/j.eclinm.2020.100281) [53](https://doi.org/10.1186/1471-244X-14-76) [54](https://doi.org/10.3390/ijerph17165929) [55](https://doi.org/10.1016/j.jad.2018.11.073) *Prior superior performance in suicide prediction tasks* / [106](https://doi.org/10.3389/fpsyt.2024.1291362) [107](https://doi.org/10.1186/S12911-024-02524-0) *Ensemble methods* / [108](https://doi.org/10.1023/A:1010933404324) *Breiman's bootstrap aggregating framework* |
| 11 | **XGBoost** <br> *Gradient tree boosting; scalable; classifies cluster mechanisms; combined into weighted ensemble* | NVDRS-RAD (circumstances + NLP-extracted features), 76 contextual predictors | [54](https://doi.org/10.3390/ijerph17165929) [55](https://doi.org/10.1016/j.jad.2018.11.073) *Prior superior performance in suicide prediction tasks* / [106](https://doi.org/10.3389/fpsyt.2024.1291362) [107](https://doi.org/10.1186/S12911-024-02524-0) *Ensemble methods* / [109](https://doi.org/10.1145/2939672.2939785) *XGBoost tree boosting* |
| 12 | **CatBoost** <br> *Categorical gradient boosting; handles mixed-type features; combined into weighted ensemble* | NVDRS-RAD (circumstances + NLP-extracted features), 76 contextual predictors | [106](https://doi.org/10.3389/fpsyt.2024.1291362) [107](https://doi.org/10.1186/S12911-024-02524-0) *Ensemble methods* / [CatBoost](https://doi.org/10.48550/arXiv.1810.11363) *CatBoost Manuscript* |
| 13 | **Weighted Ensemble (RF + XGBoost + CatBoost)** <br> *Final classifier; optimizes classification accuracy + interpretability; internal & external validation; temporal holdout (train 2000–2015, test 2016–2023)* | NVDRS-RAD (circumstances + NLP features), 76 contextual predictors | [106](https://doi.org/10.3389/fpsyt.2024.1291362) [107](https://doi.org/10.1186/S12911-024-02524-0) *Ensemble methods* / [109](https://doi.org/10.1145/2939672.2939785) *Optimizing classification accuracy and interpretability* |

**▸ Aim 2 — Model Interpretability** 
| # | Analysis / Model Type | Datasets Required | Citations |
|---|---|---|---|
| 14 | **SHAP Values & Gini Importance** <br>  *Post-hoc feature attribution for tree-based models; identifies which predictors differentiate contagion-driven vs. structural clusters* | Applied to trained RF, XGBoost, CatBoost models; All predictors from NVDRS-RAD + contextual set | [110](https://doi.org/10.7717/peerj-cs.2051) *Ensemble machine learning for Inmate suicidal behavior* |
