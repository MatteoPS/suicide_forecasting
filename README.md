## Repo Structure
```text
.
├── config/            # Model configs, hyperparameters, YAML files
├── data/              # All data — this is gitignored
│   ├── external/
│   ├── processed/
│   └── raw           
├── docs/              # Project documentation and data access instructions
├── notebooks/         # Exploratory and communicative work  
├── src/ 
│   ├── etl/           # Data import, cleaning, transform
│   ├── geospatial/    # SaTScan Analysis  
│   └── utils/   
└── tests/           
```

## Datasets

- ***SEDD*** — ED attempt data; State Emergency Department Databases (HCUP)
- ***SID*** — Inpatient attempt data; State Inpatient Databases (HCUP)
- ***NVDRS-RAD*** — Violent death registry
- ***Contextual*** — 76 contextual predictors: assembled from 26 data sources across 4 domains (demographic/environmental, socioeconomic/structural, social/community, individual-level risk factors); includes Census, BRFSS, and other public-use sources
- ***Narrative*** — NVDRS-RAD qualitative text narratives from original medical examiner, coroner, and law enforcement reports; provide critical context
