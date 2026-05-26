```text
.
├── config/                     # Model configs, hyperparameters, YAML files
├── data/                       # All data — this is gitignored
├── docs/                       # Project documentation and data access instructions
├── notebooks/                  # Exploratory and communicative work — not production code
├── README.md                   
├── src/                        # All production-grade code
│   ├── dynamical/              # Dynamic model of suicide contagion
│   │   └── README.md           
│   ├── ensemble/               # Multi-model combination logic; weighted averaging; scoring rules
│   │   └── README.md           
│   ├── ml/                     # LightGBM, RF, XGBoost, CatBoost, NN training and evaluation
│   │   └── README.md           
│   ├── nlp/                    # distilBERT pipelines; NVDRS-RAD narrative processing
│   │   └── README.md           
│   └── statistical/            # Bayesian hierarchical models; spatial smoothing; temporal integration
│       └── README.md           
└── test/                       # Unit and integration tests for src/ modules
```
