```text
.
├── config/          # Model configs, hyperparameters, YAML files
├── data/ 
│   └── raw          # All data — this is gitignored
├── docs/            # Project documentation and data access instructions
├── notebooks/       # Exploratory and communicative work — not production code     
├── src/             # All production-grade code
│   ├── dynamical/   # Dynamic model of suicide contagion       
│   ├── ensemble/    # Multi-model combination logic; weighted averaging; scoring rules
│   ├── ml/          # LightGBM, RF, XGBoost, CatBoost, NN training and evaluation      
│   ├── nlp/         # distilBERT pipelines; NVDRS-RAD narrative processing
│   └── statistical/ # Bayesian hierarchical models; spatial smoothing; temporal integration   
└── test/            # Unit and integration tests for src/ modules
```

[https://github.com/MatteoPS/suicide_forecasting/blob/main/docs/AFSP_models_table.html](https://matteops.github.io/suicide_forecasting/docs/AFSP_models_table.html)
