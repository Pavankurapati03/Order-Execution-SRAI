# Phase 0 Discovery Report: Quantellix Order Execution Architecture & Data Shapes

> **Status:** Completed  
> **Target:** Foundation for Agentic AI "Understand" Stage (Steps 1–4)  
> **Author:** Antigravity (Pair Programming with User)  
> **Reference:** Quantellix Agentic AI "Understand" Stage Implementation Spec  

---

## 1. Real Shape of the SXI Result Object (`sxi_res`)

The function `run_sxi_optimization()` in `portal/sxi_engine.py` (line 1210) returns a dictionary with the following verified structure:

```python
{
    # Solution metadata
    "solution_code": "E4",                                # str (e.g. "E1" .. "E10")
    "solution_name": "Reduce Delivery Lead Time",          # str
    "solution_slug": "reduce-delivery-lead-time",         # str
    "stage": "Demand Planning",                           # str: "Sales Forecasting" | "Demand Planning" | "Procurement"
    "task_type": "regression",                            # str: "regression" | "binary_classification"
    "target_name": "lead_time_days",                      # str (column name in dataset)
    "target_display": "Average Delivery Lead Time",       # str
    "kpi_name": "Average Delivery Lead Time",             # str
    "kpi_unit": "days",                                   # str (e.g. "%", "days", "units")

    # Baseline & Optimized KPIs
    "baseline_kpi": 8.4,                                  # float
    "optimized_kpi": 5.1,                                 # float
    "improvement_pct": 39.3,                              # float (signed or positive percentage)

    # Composite SXI Scores (0.0 to 10.0 scale)
    "current_sxi_score": 5.82,                            # float
    "target_sxi_score": 8.91,                             # float
    "current_sxi": "5.82",                                # str
    "lvl1_sxi": "7.10",                                   # str (Level 1: Conservative)
    "lvl2_sxi": "8.25",                                   # str (Level 2: Moderate / Recommended)
    "lvl3_sxi": "8.91",                                   # str (Level 3: Aggressive)

    # 3 Levels of Scenario Targets
    "lvl1_kpi": "7.2",                                    # str
    "lvl2_kpi": "6.1",                                    # str
    "lvl3_kpi": "5.1",                                    # str
    "lvl1_pct": "14.3",                                   # str (%)
    "lvl2_pct": "27.4",                                   # str (%)
    "lvl3_pct": "39.3",                                   # str (%)

    # Holdout Model Performance
    "model_performance": {
        "model_type": "XGBoost + CatBoost Ensemble",      # str
        "accuracy_pct": "98.2",                           # str (%)
        "r2": "0.87",                                     # str (for regression)
        "r2_pct": "87.0",                                 # str
        "rmse": "1.42",                                   # str
        "mae": "1.08",                                    # str
        "within_10_pct": "94.5",                          # str (% predictions within 10% tolerance)
        "precision_pct": "95.2",                          # Optional[str] (for binary classification)
        "recall_pct": "98.6",                             # Optional[str]
        "f1_score": "0.97",                               # Optional[str]
        "auc_score": "0.995",                             # Optional[str]
        "confusion_matrix": [[120, 5], [3, 142]],         # Optional[list]
        "train_samples": 24550,                           # int (n_train)
        "test_samples": 6138                              # int (n_test)
    },

    # Feature Importance & Weights
    "weights_table": [                                    # list[dict]
        {
            "feature": "safety_stock_days",
            "weight": 0.342,                              # float (0.0 to 1.0)
            "direction": "decreasing",                    # str: "increasing" | "decreasing"
            "actionable": True,                           # bool
            "rank": 1
        },
        ...
    ],
    "top_drivers": ["safety_stock_days", "carrier_rating", "reorder_point"], # list[str]
    "rules": [...],                                       # list[str] (decision tree rules)
    "interventions": {...},                               # dict (levers and recommended targets)

    # Navigation & Artifacts
    "dashboard_url": "/portal/dashboard/{session_id}?solution=reduce-delivery-lead-time",
    "run_id": "run_a1b2c3d4",
    "download_csv_url": "/api/sxi/download-csv/{session_id}",
    "correlation_chart": "<svg ...></svg>",               # rendered SVG string
    "dashboard_data": {...}                               # nested complete dashboard state
}
```

---

## 2. Real Shape of the Session Cache & Persistence

### 2.1 In-Memory Web Cache (`CHAT_SESSIONS`)
Located in `portal/views.py`:
- Key: `session_id` (UUID4 string, e.g. `"9f8e7d6c-..."`)
- Value structure:
  ```python
  CHAT_SESSIONS[session_id] = {
      'session_id': session_id,
      'email': user_email,
      'user_name': user_name,
      'stage': current_stage,                        # e.g. "post_sxi_qa"
      'solution_key': 'e4',                          # primary solution
      'solution_keys': ['e3', 'e4', 'e9'],           # list of chosen solutions
      'solution_cfg': {...},                         # primary solution config
      'solution_cfgs': [{...}, {...}, {...}],        # list of all configs
      'filename': 'E-commerce_order_execution_dataset.csv',
      'raw_df': <pd.DataFrame>,
      'current_df': <pd.DataFrame>,
      'has_master': True,
      'master_info': {
          'csv': '/uploads/master/master_{session_id}.csv',
          'download_url': '/api/master/download/{session_id}',
          'filename': 'master_{session_id}.csv',
          'path': 'C:/.../portal/uploads/master/master_{session_id}.csv',
          'rows': 30688,
          'cols': ['master_id', 'sku_id', ...]
      },
      'master_download_url': '/api/master/download/{session_id}',
      'actionable_features': ['safety_stock_days', 'reorder_point'],
      'sxi_results': {...},                          # single-solution SXI dict
      'multi_sxi_results': {                         # multi-solution SXI dict
          'e3': {...},
          'e4': {...},
          'e9': {...}
      },
      'sxi_dashboard_data': {...},
      'history': [...]                               # chat bubble history
  }
  ```

### 2.2 On-Disk Durable Storage (`portal/uploads/`)
Even if the in-memory process restarts, the SXI run persists:
- `uploads/master/master_{session_id}.csv` (Normalized master dataset)
- `uploads/master/master_{session_id}_encoding.json` (Categorical encoders)
- `uploads/sxi_dashboard_{session_id}.json` (Primary dashboard JSON)
- `uploads/sxi_dashboard_{session_id}_{solution_code}.json` (Per-solution dashboard JSON, e.g. `sxi_dashboard_xxx_e4.json`)
- `uploads/eda_report_{session_id}_{file_id}.html` (Automated EDA report)

---

## 3. Real Database Models & Schema (Neon PostgreSQL)

All database operations in `portal/database.py` execute raw SQL via `get_neon_connection()` using `psycopg2.pool.SimpleConnectionPool`:

| Table Name | Purpose | Key Columns |
| :--- | :--- | :--- |
| `licenses_licensekey` | Lifetime licenses | `id`, `key`, `owner_name`, `owner_email`, `solution_slug`, `credits_balance`, `plan`, `is_active`, `created_at` |
| `credit_ledger` | Transaction audit log | `id`, `license_id`, `amount`, `balance_after`, `description`, `reference_id`, `created_at` |
| `sessions` | Web sessions | `id`, `session_id`, `email`, `user_name`, `solution_slug`, `stage`, `created_at`, `updated_at` |
| `uploaded_files` | File upload tracking | `id`, `session_id`, `filename`, `file_path`, `file_size_bytes`, `row_count`, `col_count`, `uploaded_at` |
| `eda_reports` | EDA summaries & HTML | `id`, `session_id`, `file_id`, `summary_json`, `report_html_path`, `created_at` |

> **Architectural Note:** In Phase 1, our new agent tables (`agent_runs`, `agent_evidence`, `agent_stage_outputs`, `agent_dossiers`, `agent_approvals`, `agent_semantic_mappings`, `agent_customer_config`, `agent_trace_events`) will be created via an idempotent `init_agent_db()` migration cleanly in Neon PostgreSQL.

---

## 4. Existing Financial Assumptions & `FINANCIAL_MAP`

Located in `portal/views.py` (lines 1002–1013):

```python
FINANCIAL_MAP = {
    'E1': (620000, "recovered unfulfilled sales demand"),
    'E2': (485000, "eliminated buffer inventory waste"),
    'E3': (820000, "recovered stockout revenue & freed working capital"),
    'E4': (210000, "downstream buffer holding cost reduction"),
    'E5': (252000, "eliminated off-invoice margin leakage"),
    'E6': (340000, "prevented component scrap & warranty penalties"),
    'E7': (380000, "consolidated volume procurement savings"),
    'E8': (290000, "avoided expedited logistics fees"),
    'E9': (450000, "single-source disruption avoidance value"),
    'E10': (240000, "recovered billing discrepancies"),
}
```

In Phase 4, this map will be refactored into `portal/agents/exposure.py` using formulaic calculations backed by `portal/agents/config/financial_assumptions.default.yaml` with explicit customer overrides and overlap de-duplication.

---

## 5. UI Integration Point in `executive_dashboard.html`

In `portal/templates/executive_dashboard.html`:
- Top: Executive Navigation Header & Solution Switcher.
- Lines ~120–190: 4 Hero KPI Cards (`Aggregated SXI Score`, `Annualized Value Potential`, `Risk Reduction Index`, `Decision Confidence`).
- **Target Insertion Point for "Agentic Decision Center":** Immediately following the 4 Hero KPI Cards, directly preceding the Multi-Solution Rollup Grid.
- This provides maximum C-suite visibility: executive leaders see high-level KPIs, then immediately review the **Agentic Decision Dossier** and **Root-Cause Chain**, before drilling down into individual solution cards.

---

## 6. Adapter Mapping Plan (`portal/agents/adapters.py`)

To shield the agentic layer from any internal field renames or discrepancies in the SXI result dictionary, `adapters.py` will normalize `sxi_res` into the standard `SolutionResult` Pydantic model:

```python
def adapt_sxi_to_solution_result(sxi_dict: dict) -> SolutionResult:
    perf = sxi_dict.get("model_performance", {})
    holdout_dict = {}
    
    # Safely parse numeric metrics
    if perf.get("accuracy_pct"):
        holdout_dict["accuracy"] = float(perf["accuracy_pct"]) / 100.0
    if perf.get("r2"):
        holdout_dict["r2"] = float(perf["r2"])
    if perf.get("auc_score"):
        holdout_dict["auc"] = float(perf["auc_score"])
    if perf.get("f1_score"):
        holdout_dict["f1"] = float(perf["f1_score"])
    if perf.get("rmse"):
        holdout_dict["rmse"] = float(perf["rmse"])

    # Adapt feature weights
    features = []
    for item in sxi_dict.get("weights_table", []):
        features.append(FeatureWeight(
            name=item.get("feature", "unknown"),
            weight=float(item.get("weight", 0.0)),
            direction="down" if item.get("direction") == "decreasing" else "up",
            actionable=bool(item.get("actionable", True))
        ))

    return SolutionResult(
        solution=sxi_dict.get("solution_code", "E4"),
        target_column=sxi_dict.get("target_name", ""),
        target_type="continuous" if sxi_dict.get("task_type") == "regression" else "binary",
        kpi_name=sxi_dict.get("kpi_name", ""),
        kpi_baseline=float(sxi_dict.get("baseline_kpi", 0.0)),
        kpi_optimized={
            "L1": float(sxi_dict.get("lvl1_kpi", sxi_dict.get("optimized_kpi", 0.0))),
            "L2": float(sxi_dict.get("lvl2_kpi", sxi_dict.get("optimized_kpi", 0.0))),
            "L3": float(sxi_dict.get("lvl3_kpi", sxi_dict.get("optimized_kpi", 0.0))),
        },
        sxi_baseline=float(sxi_dict.get("current_sxi_score", 0.0)),
        sxi_optimized=float(sxi_dict.get("target_sxi_score", 0.0)),
        models_run=[perf.get("model_type", "XGBoost Ensemble")],
        primary_model=perf.get("model_type", "XGBoost Ensemble"),
        holdout=holdout_dict,
        n_train=int(perf.get("train_samples", 0)),
        n_test=int(perf.get("test_samples", 0)),
        features=features
    )
```

---

## 7. Phase 0 Conclusion

The repository is fully inspected and ready for **Phase 1: Foundations**:
1. Feature flag `AGENTIC_ENABLED = False` will protect existing routes.
2. SXI results are already persisted as JSON on disk and in session memory, allowing the `EvidenceBuilder` to execute without rerunning models.
3. Neon DB connection pool is active and ready for agent table migrations.
