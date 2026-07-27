```
.
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Automated testing and linting
│       └── docker-publish.yml     # Automated container builds
├── config/
│   ├── data_schema.yaml           # Expected column types and bounds
│   ├── lgbm_params.yaml           # LightGBM hyperparameter space
│   └── nlp_configs.yaml           # distilBERT tokenization settings
├── data/
│   ├── raw/
│   │   └── .gitkeep               # Keeps folder in git; actual data is ignored
│   └── processed/
│       └── .gitkeep
├── docs/
│   ├── setup.md                   # Environment setup guide
│   └── architecture.md            # System design and pipeline flow
├── notebooks/
│   ├── 01_eda_target_variable.ipynb
│   └── 02_error_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py              # Fetches and validates raw data
│   │   ├── clean.py               # Handles missing values and outliers
│   │   └── features.py            # Generates lag features and rolling means
│   ├── dynamical/
│   │   ├── __init__.py
│   │   ├── contagion_sir.py       # Core compartmental model logic
│   │   └── ode_solver.py          # Differential equation integration
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── stacking.py            # Meta-learner for combined predictions
│   │   └── scoring.py             # Weighted evaluation metrics (WIS, MAE)
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── train_lgbm.py          # LightGBM training loop
│   │   ├── train_nn.py            # Neural network training loop
│   │   └── evaluate.py            # Generates residual plots and scores
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── text_clean.py          # Strips PII/formatting from narratives
│   │   └── bert_embed.py          # distilBERT tokenization and embedding
│   └── statistical/
│       ├── __init__.py
│       ├── arima.py               # Auto-ARIMA fitting
│       └── exp_smoothing.py       # Holt-Winters implementation
├── tests/
│   ├── conftest.py                # Shared test fixtures (e.g., dummy data)
│   ├── test_data_pipeline.py      # Checks for data leakage
│   ├── test_ml_shapes.py          # Ensures model input/output dimensions match
│   └── test_nlp_cleaning.py       # Validates PII removal
├── pyproject.toml                 # Dependencies (Poetry or standard pip)
├── .gitignore                     # Ignores data/, __pycache__/, .env
├── Dockerfile                     # OS and Python environment instructions
├── README.md                      # High-level project overview
└── main.py                        # CLI entry point (e.g., python main.py --run nlp)
```