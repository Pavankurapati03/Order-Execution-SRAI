"""
SXI Engine - Standalone Machine Learning & Analytical Modeling Package.

Provides automated feature engineering, task-specific ML training (Classification,
Regression, Time-Series Forecasting), explainability (SHAP, decision trees), and
interactive Plotly visualizations.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add current package directory to sys.path so internal imports resolve cleanly
_PKG_ROOT = Path(__file__).resolve().parent
for _sub in [_PKG_ROOT, _PKG_ROOT / "core", _PKG_ROOT / "preprocessing", _PKG_ROOT / "utils"]:
    _p_str = str(_sub)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

__version__ = "1.0.0"
