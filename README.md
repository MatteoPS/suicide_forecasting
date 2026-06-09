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
