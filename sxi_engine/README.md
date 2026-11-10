# SXI Engine (Standalone Machine Learning & Decision Intelligence)

This folder contains the self-contained **SXI / DXI Analytics & Machine Learning Engine** extracted from the platform. You can copy this entire `sxi_engine/` folder directly into any other Python backend (FastAPI, Flask, Django, or a standalone ML service).

---

## 📁 Folder Structure & File Manifest

```text
sxi_engine/
├── __init__.py                          # Package initialization & path resolution
├── requirements.txt                     # Standalone Python dependencies
├── README.md                            # Documentation & integration guide
│
├── core/                                # ML Models, Training & Statistical Engine
│   ├── __init__.py
│   ├── sxi_exe.py                       # The Core: SxiProcess, model_execution, sxirl_engine
│   └── sxi_executor.py                  # High-level pipeline executor wrapper
│
├── preprocessing/                       # Data Cleaning, Targets & Feature Engineering
│   ├── __init__.py
│   ├── problem_contracts.py             # Problem definition registry (Classification & Regression)
│   ├── problem_fe.py                    # Automated feature engineering & column builders
│   ├── problem_build.py                 # Master dataset builder
│   ├── eda_utils.py                     # Automated Exploratory Data Analysis & profiling
│   ├── csv_extractor.py                 # Safe CSV readers & schema extractors
│   └── target_creator.py                # Target outcome creation logic
│
├── utils/                               # Explainability, Visualization & Helpers
│   ├── __init__.py
│   ├── leakage_guard.py                 # Target leakage detection & safety filters
│   ├── plotly_export.py                 # Plotly interactive chart exporter (HTML/PNG)
│   ├── regression_tree_explainer.py     # Decision tree rule extractor into human English
│   ├── shap_engine.py                   # SHAP explainability & feature attributions
│   ├── timeseries_executor.py           # Prophet / ARIMA time-series forecaster
│   ├── dashboard4_friction_traction.py  # Traction vs Friction feature scores
│   ├── subindex_dxi.py                  # Subindex calculation engine
│   ├── target_info_store.py             # Schema & metadata configuration store
│   ├── json_utils.py                    # JSON serializer for numpy/pandas structures
│   ├── path_utils.py                    # Output path resolution
│   └── llm_provider.py                  # LLM provider wrapper (OpenAI, Anthropic, Groq)
│
└── examples/
    ├── __init__.py
    └── standalone_demo.py               # Complete working demo (runs out-of-the-box)
```

---

## 🚀 How to Add to Your New Project

### Step 1: Copy the Folder
Copy the `sxi_engine/` directory into your new project root:
```bash
cp -r sxi_engine /path/to/your_new_project/sxi_engine
```

### Step 2: Install Dependencies
```bash
pip install -r sxi_engine/requirements.txt
```

---

## 💡 Quick Start Code Examples

### 1. Automated EDA (Exploratory Data Analysis)
Scan any uploaded CSV or DataFrame, calculate distributions, missing values, and generate summary charts:

```python
from sxi_engine.preprocessing.eda_utils import generate_eda_dashboard

results = generate_eda_dashboard(
    dataset_path="data/my_dataset.csv",
    output_path="outputs/eda"
)
print("Dataset summary:", results["rows"], "rows,", results["columns"], "columns")
print("Key findings:", results["findings"])
```

---

### 2. Feature Engineering & Target Extraction
Transform raw e-commerce/business data using pre-engineered contracts:

```python
import pandas as pd
from sxi_engine.preprocessing.problem_fe import apply_problem_fe

df = pd.read_csv("data/raw_data.csv")

# problem_id 1 = Full-Price Buyers (Classification)
# problem_id 2 = Repeat Buyer / Product Value (Classification)
# problem_id 4 = Repurchase Timing (Regression)
# problem_id 6 = Supply-Chain Demand (Time-Series / Regression)
fe_result = apply_problem_fe(df, problem_id=1)

print("Features engineered:", fe_result.features_used)
print("Target column:", fe_result.target)
clean_df = fe_result.df
```

---

### 3. Running the SXI ML Engine (Classification or Regression)

```python
from sxi_engine.core.sxi_executor import SXIExecutor

target_info = {
    "Target Name": "is_converted",        # Name of target column
    "Target Type": "categorical",         # "categorical" or "continuous"
    "Task Type": "classification",        # "classification" or "regression"
    "Selected Outcome": 1,                # Positive class (e.g. 1 or "Yes")
    "Good Outcome Value": 1,
    "Bad Outcome Value": 0,
    "Optimization Goal": "increasing",    # "increasing" or "decreasing"
    "Target Outcome Improvement": 5.0     # Target % improvement
}

executor = SXIExecutor(
    request=None,
    dataframe_path="data/clean_master.csv",
    target_info=target_info
)

# Executes model training (XGBoost, Random Forest), metrics, SHAP, and DXI scores
results = executor.execute()
```

---

### 4. Decision Tree Business Paths & Explanations

```python
from sxi_engine.utils.regression_tree_explainer import (
    extract_best_tree_paths,
    format_classification_business_paths
)

paths = extract_best_tree_paths(model, feature_names, max_depth=3)
business_rules = format_classification_business_paths(paths)
for rule in business_rules:
    print("Rule:", rule)
```

---

## 🛠 Framework Agnostic
This package is decoupled from Django models and can run directly in:
* **FastAPI** (`uvicorn`)
* **Flask**
* **Celery / Redis background workers**
* **Standalone CLI or Jupyter Notebooks**
