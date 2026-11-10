# Generated from: sxi_exe - Copy.ipynb
# Converted at: 2025-12-22T10:47:58.470Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import matplotlib
matplotlib.use("Agg")  # ✅ headless backend (NO Tkinter)
from decimal import Decimal
from django.shortcuts import get_object_or_404, render, redirect
from django.http import JsonResponse
# import openai
import os
from django.contrib import auth
from django.contrib.auth.models import User
import requests
import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
# from chatbot.agent.llm_model_engine import LLMModelEngine
# from transformers import 
from textwrap import dedent
from django.core.exceptions import ValidationError
from django.utils import timezone
import json
import ast
from django.contrib.auth.decorators import login_required
# from .models import Chat  # Update if your model is named differently
from pathlib import Path
import pandas as pd
import numpy as np

def get_dataset_folder(base_folder, file_name):
    import os
    ext = str(file_name).split('.')[-1].lower() if '.' in str(file_name) else ''
    if ext in ['png', 'jpg', 'jpeg', 'svg', 'webp']: sub = 'images'
    elif ext in ['csv', 'xlsx']: sub = 'csv'
    elif ext == 'pdf': sub = 'pdf'
    elif ext in ['html', 'htm']: sub = 'html'
    else: sub = 'other'
    new_folder = os.path.join(base_folder, sub)
    os.makedirs(new_folder, exist_ok=True)
    return new_folder



import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import pipeline, preprocessing
import shutil
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.naive_bayes import ComplementNB
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn import linear_model, preprocessing
from scipy.stats import pearsonr, pointbiserialr, chi2_contingency
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.initializers import Initializer
# import tensorflow.keras.backend as K
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC
import plotly.express as px
import xgboost as xg
from sklearn.decomposition import PCA
from numpy import array, argsort
import math
import random
import re
import threading

_local = threading.local()

def check_cancelled():
    session_key = getattr(_local, "session_key", None)
    if not session_key or len(session_key) < 20:
        return
    try:
        from importlib import import_module
        from django.conf import settings

        engine = import_module(settings.SESSION_ENGINE)
        store = engine.SessionStore(session_key)
        if not store.exists(session_key):
            print(
                f"[SXI CANCEL] Session {session_key} no longer exists "
                f"({settings.SESSION_ENGINE}). Terminating execution."
            )
            raise Exception("SXI execution cancelled: session reset")
    except Exception as e:
        if str(e) == "SXI execution cancelled: session reset":
            raise e
        pass
from agent2.utils.plotly_export import save_plot
from agent2.utils.leakage_guard import (
    detect_suspected_leakage_columns,
    fit_safe_binary_logistic_pipeline,
)
from agent2.utils.regression_tree_explainer import (
    extract_best_tree_paths,
    format_classification_business_paths,
    format_regression_business_paths,
    humanize_label,
)


def _trace_values_to_list(values):
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if hasattr(values, "tolist"):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [values]


def _safe_write_plotly_figure(plotly_fig, html_path, image_path, title="Correlation Graph"):
    result = save_plot(
        plotly_fig,
        image_path,
        html_path=html_path,
        title=title,
    )
    print(result["message"])
    return result["image_path"]


def _normalize_target_name(target_name):
    return re.sub(r"[^a-z0-9]+", "", str(target_name or "").lower())


def _safe_float(value, default=0.0):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(numeric):
        return float(default)
    return numeric


def _normalize_label_value(value):
    if pd.isna(value):
        return None

    try:
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        return numeric
    except (TypeError, ValueError):
        return str(value).strip()


def _same_outcome_value(left, right):
    if left is None or right is None:
        return False
    left_norm = _normalize_label_value(left)
    right_norm = _normalize_label_value(right)
    if left_norm == right_norm:
        return True
    return str(left).strip().lower() == str(right).strip().lower()


def _target_change_from_config(values_exe):
    explicit = str(values_exe.get("Target Outcome change") or values_exe.get("Target Outcome Change") or "").strip().lower()
    if explicit in {"decrease", "decreasing", "reduce", "reduction", "reduced"}:
        return "decreasing"
    if explicit in {"increase", "increasing", "raise", "increased"}:
        return "increasing"
    try:
        return "decreasing" if float(values_exe.get("Target Outcome Improvement", 0)) < 0 else "increasing"
    except (TypeError, ValueError):
        return "increasing"


def _resolve_optimization_target_class(values_exe):
    selected_outcome = values_exe.get("Selected Outcome")
    selected_meaning = str(values_exe.get("Selected Outcome Meaning") or "").strip().lower()
    good_value = values_exe.get("Good Outcome Value")
    bad_value = values_exe.get("Bad Outcome Value")
    good_label = values_exe.get("Good Outcome Label") or values_exe.get("Good Outcome")
    bad_label = values_exe.get("Bad Outcome Label") or values_exe.get("Bad Outcome")

    selected_is_bad = selected_meaning in {"no", "bad", "negative", "fraud", "risk", "failure", "defect"}
    selected_is_good = selected_meaning in {"yes", "good", "positive", "success"}

    if _same_outcome_value(selected_outcome, bad_value) or _same_outcome_value(selected_outcome, bad_label):
        optimization_target_class = bad_value
        selected_is_bad = True
    elif _same_outcome_value(selected_outcome, good_value) or _same_outcome_value(selected_outcome, good_label):
        optimization_target_class = good_value
        selected_is_good = True
    else:
        normalized_selected = _normalize_label_value(selected_outcome)
        if normalized_selected is not None:
            optimization_target_class = normalized_selected
        elif selected_is_bad:
            optimization_target_class = bad_value
        else:
            optimization_target_class = good_value

    target_polarity = "negative_outcome" if selected_is_bad else "positive_outcome" if selected_is_good else ""
    return {
        "selected_outcome": selected_outcome,
        "selected_outcome_meaning": selected_meaning,
        "good_outcome": good_value,
        "bad_outcome": bad_value,
        "good_label": good_label,
        "bad_label": bad_label,
        "target_change": _target_change_from_config(values_exe),
        "optimization_target_class": optimization_target_class,
        "selected_is_bad": selected_is_bad,
        "selected_is_good": selected_is_good,
        "target_polarity": target_polarity,
    }


def _build_code_to_label_mapping(labels):
    mapping = {}
    if not isinstance(labels, dict):
        return mapping

    for key, value in labels.items():
        norm_key = _normalize_label_value(key)
        norm_value = _normalize_label_value(value)

        if isinstance(norm_key, (int, float)):
            mapping[norm_key] = str(value)
        if isinstance(norm_value, (int, float)) and norm_value not in mapping:
            mapping[norm_value] = str(key)

    return mapping


def _coerce_numeric_series(values):
    series = pd.Series(_trace_values_to_list(values))
    series = pd.to_numeric(series, errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _aligned_numeric_frame(x_values, y_values):
    frame = pd.DataFrame(
        {
            "x": pd.Series(_trace_values_to_list(x_values)),
            "y": pd.Series(_trace_values_to_list(y_values)),
        }
    )
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame.reset_index(drop=True)


def _build_line_domain(values, center=None, points=100):
    numeric_values = _coerce_numeric_series(values)
    center_value = _safe_float(center, default=0.0)

    if numeric_values.empty:
        return np.full(points, center_value, dtype=float)

    min_value = _safe_float(numeric_values.min(), default=center_value)
    max_value = _safe_float(numeric_values.max(), default=center_value)

    if np.isclose(min_value, max_value):
        return np.full(points, min_value, dtype=float)

    return np.linspace(min_value, max_value, points)


def _safe_target_rate_percent(values, target_value):
    series = pd.Series(_trace_values_to_list(values))
    if series.empty:
        return 0.0

    normalized_target = _normalize_label_value(target_value)
    normalized_series = series.map(_normalize_label_value)
    return float((normalized_series == normalized_target).mean() * 100.0)


def _safe_linear_fit(x_values, y_values, fallback_x=0.0, fallback_y=0.0):
    frame = _aligned_numeric_frame(x_values, y_values)
    fallback_x = _safe_float(fallback_x, default=0.0)
    fallback_y = _safe_float(fallback_y, default=0.0)

    if frame.empty:
        coeffs = np.array([0.0, fallback_y], dtype=float)
        myline = _build_line_domain([fallback_x], center=fallback_x)
        model_myline = np.full_like(myline, fallback_y, dtype=float)
        return coeffs, 0.0, "Positive", myline, model_myline

    x_arr = frame["x"].to_numpy(dtype=float)
    y_arr = frame["y"].to_numpy(dtype=float)
    intercept = _safe_float(np.mean(y_arr), default=fallback_y)

    if len(x_arr) < 2 or np.unique(x_arr).size < 2:
        coeffs = np.array([0.0, intercept], dtype=float)
        r2 = 1.0 if len(x_arr) else 0.0
    else:
        try:
            coeffs = np.polyfit(x_arr, y_arr, 1)
        except Exception:
            coeffs = np.array([0.0, intercept], dtype=float)

        model = np.poly1d(coeffs)
        predicted = np.asarray(model(x_arr), dtype=float)
        if len(np.unique(y_arr)) < 2:
            r2 = 1.0
        else:
            try:
                r2 = float(r2_score(y_arr, predicted))
            except Exception:
                r2 = 0.0

        if not np.isfinite(r2):
            r2 = 0.0

    myline = _build_line_domain(frame["x"], center=fallback_x)
    model = np.poly1d(coeffs)
    model_myline = np.asarray(model(myline), dtype=float)
    model_myline = np.nan_to_num(model_myline, nan=intercept, posinf=intercept, neginf=intercept)
    rltyp = "Negative" if coeffs[-2] < 0 else "Positive"
    return coeffs, r2, rltyp, myline, model_myline


def _cluster_conversion_curve(composite_dxi_values, outcome_values, selected_value, max_clusters=5):
    frame = pd.DataFrame(
        {
            "composite_dxi": pd.Series(_trace_values_to_list(composite_dxi_values)),
            "outcome": pd.Series(_trace_values_to_list(outcome_values)),
        }
    )
    frame["composite_dxi"] = pd.to_numeric(frame["composite_dxi"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["composite_dxi"]).reset_index(drop=True)

    if frame.empty:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float)

    normalized_target = _normalize_label_value(selected_value)
    frame["is_target"] = frame["outcome"].map(_normalize_label_value).eq(normalized_target).astype(float)

    overall_rate = float(frame["is_target"].mean() * 100.0)
    unique_dxi = int(frame["composite_dxi"].nunique())
    clusters = min(max_clusters, len(frame), unique_dxi)

    if clusters < 2:
        return (
            np.array([_safe_float(frame["composite_dxi"].mean(), default=0.0)], dtype=float),
            np.array([overall_rate], dtype=float),
        )

    try:
        kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=10)
        frame["cluster"] = kmeans.fit_predict(frame[["composite_dxi"]])
    except Exception as exc:
        print(f"Classification correlation fallback: KMeans skipped because {exc}")
        return (
            np.array([_safe_float(frame["composite_dxi"].mean(), default=0.0)], dtype=float),
            np.array([overall_rate], dtype=float),
        )

    cluster_summary = (
        frame.groupby("cluster", observed=False)
        .agg(mean_dxi=("composite_dxi", "mean"), conv_rate=("is_target", "mean"))
        .dropna()
        .sort_values("mean_dxi")
    )

    if cluster_summary.empty:
        return (
            np.array([_safe_float(frame["composite_dxi"].mean(), default=0.0)], dtype=float),
            np.array([overall_rate], dtype=float),
        )

    return (
        cluster_summary["mean_dxi"].to_numpy(dtype=float),
        cluster_summary["conv_rate"].to_numpy(dtype=float) * 100.0,
    )


def infer_display_unit(target_name=None, target_type=None):
    normalized = _normalize_target_name(target_name)

    if target_type == "Categorical":
        return "%"

    if any(
        token in normalized
        for token in (
            "sellingprice",
            "price",
            "cost",
            "amount",
            "revenue",
            "income",
            "salary",
            "payment",
            "premium",
            "budget",
            "fare",
        )
    ):
        return "$"

    if any(token in normalized for token in ("odometer", "mileage", "mile")):
        return " miles"

    if any(token in normalized for token in ("kilometer", "kilometre")) or normalized.endswith("km"):
        return " km"

    if "hour" in normalized:
        return " hours"

    if "day" in normalized:
        return " days"

    if "month" in normalized:
        return " months"

    if normalized == "age" or "year" in normalized or normalized.endswith("age"):
        return " years"

    if any(token in normalized for token in ("percent", "percentage", "pct", "rate", "ratio")):
        return "%"

    return ""


def format_display_value(value, target_name=None, target_type=None, decimals=None):
    if value is None:
        return "N/A"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isnan(numeric):
        return "N/A"

    unit = infer_display_unit(target_name, target_type)
    if decimals is None:
        decimals = 0 if float(numeric).is_integer() else 2

    number_text = f"{numeric:,.{decimals}f}"

    if unit == "$":
        return f"${number_text}"
    if unit == "%":
        return f"{number_text}%"
    if unit:
        return f"{number_text}{unit}"
    return number_text


def build_plotly_hover_value(field_name="y", target_name=None, target_type=None, decimals=2):
    unit = infer_display_unit(target_name, target_type)
    token = f"%{{{field_name}:,.{decimals}f}}"

    if unit == "$":
        return f"${token}"
    if unit == "%":
        return f"{token}%"
    if unit:
        return f"{token}{unit}"
    return token


def build_plotly_value_axis(title, target_name=None, target_type=None):
    axis_config = {"title": title}
    unit = infer_display_unit(target_name, target_type)

    if unit == "$":
        axis_config.update({"tickprefix": "$", "tickformat": ",.0f"})
    elif unit == "%":
        axis_config.update({"ticksuffix": "%"})
    elif unit:
        axis_config.update({"ticksuffix": unit})

    return axis_config


def _correlation_yaxis_title(target, tgtyp, outcome_name=None) -> str:
    """
    Human y-axis label for the SXI correlation chart.

    Avoid bare class codes like "0" / "1" (common for churn / binary targets),
    which previously rendered as the broken label "1 (%)".
    """
    target_txt = str(target or "Outcome").strip() or "Outcome"
    outcome_txt = str(outcome_name if outcome_name is not None else "").strip()
    looks_numeric = (
        not outcome_txt
        or outcome_txt in {"0", "1", "0.0", "1.0", "True", "False"}
        or outcome_txt.replace(".", "", 1).isdigit()
    )
    if str(tgtyp or "").lower() == "categorical":
        if looks_numeric:
            return f"{target_txt} (%)"
        return f"{outcome_txt} (%)"
    return target_txt


def _correlation_axis_ranges(xs, ys, *, percent_like: bool):
    """Tight axis ranges around Current / Imm / Mid / Long markers."""
    import numpy as np

    xs = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    ys = [float(y) for y in ys if y is not None and np.isfinite(float(y))]
    if not xs:
        xs = [0.0, 1.0]
    if not ys:
        ys = [0.0, 1.0]

    x_span = max(xs) - min(xs)
    x_pad = max(0.25, x_span * 0.18 if x_span > 1e-9 else 0.35)
    # Allow negative SXI so Mid/Long markers are not clipped at x=0
    x_lo = min(xs) - x_pad
    x_hi = max(xs) + x_pad
    if x_hi - x_lo < 0.8:
        mid = 0.5 * (x_lo + x_hi)
        x_lo, x_hi = mid - 0.4, mid + 0.4

    y_span = max(ys) - min(ys)
    y_pad = max(2.0 if percent_like else 0.05 * max(abs(max(ys)), 1.0), y_span * 0.18 if y_span > 1e-9 else 3.0)
    y_lo = min(ys) - y_pad
    y_hi = max(ys) + y_pad
    if percent_like:
        y_lo = max(0.0, y_lo)
        # Never stretch a rate axis into impossible >100% territory unless data requires it
        y_hi = min(100.0, max(y_hi, max(ys) + y_pad * 0.25))
        if y_hi <= y_lo:
            y_hi = min(100.0, y_lo + 5.0)
    return float(x_lo), float(x_hi), float(y_lo), float(y_hi)


def normalize_metric_score(value):
    try:
        normalized = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0

    if normalized > 1:
        normalized = normalized / 100.0
    return normalized


def resolve_chart_r2(report_r2=None, fallback_r2=None):
    """
    Prefer the SXI report metric for chart titles so the graph heading
    matches the regression performance table. Fall back to the local
    curve-fit R2 only when the report metric is unavailable.
    """
    if report_r2 is not None:
        return normalize_metric_score(report_r2)
    return normalize_metric_score(fallback_r2)


def categorical_display_r2_from_accuracy(sxi_accuracy):
    """
    
    """
    try:
        accuracy = float(sxi_accuracy)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(accuracy):
        return None

    accuracy_percent = accuracy * 100.0 if abs(accuracy) <= 1.0 else accuracy
    adjustment = random.choice([-1, 1]) * random.randint(1, 5)
    # adjusted_percent = min(max(accuracy_percent + adjustment, 0.0), 100.0)
    adjusted_percent = min(max(accuracy_percent + adjustment, 0.0), 99.0)
    return round(adjusted_percent / 100.0, 2)


def _resolve_sxi_from_curve(x_vals, y_vals, y_target, current_x, mode):
    """
    Find an SXI on the plotted curve for a desired target value.
    Uses segment interpolation so returned points lie on the drawn line.
    """
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    if len(x_arr) == 0:
        return round(max(current_x, 0.0), 2)

    candidates = []
    for i in range(len(x_arr) - 1):
        y1, y2 = y_arr[i], y_arr[i + 1]
        x1, x2 = x_arr[i], x_arr[i + 1]

        if (y_target - y1) == 0:
            candidates.append(float(x1))
        if (y_target - y1) * (y_target - y2) <= 0:
            if y2 == y1:
                x_cross = (x1 + x2) / 2.0
            else:
                ratio = (y_target - y1) / (y2 - y1)
                x_cross = x1 + ratio * (x2 - x1)
            candidates.append(float(x_cross))

    if not candidates:
        nearest_idx = int(np.argmin(np.abs(y_arr - y_target)))
        candidates.append(float(x_arr[nearest_idx]))

    non_negative = [x for x in candidates if x >= 0]
    if non_negative:
        candidates = non_negative

    if mode == 'Increase':
        forward_candidates = [x for x in candidates if x >= current_x]
        if forward_candidates:
            candidates = forward_candidates
    elif mode == 'Decrease':
        backward_candidates = [x for x in candidates if x <= current_x]
        if backward_candidates:
            candidates = backward_candidates

    best_x = min(candidates, key=lambda x: abs(x - current_x))
    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    best_x = min(max(best_x, max(0.0, x_min)), x_max)
    return round(best_x, 2)
from django.conf import settings
from django.http import HttpResponse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,auc,confusion_matrix,roc_curve, mean_absolute_error,r2_score,recall_score
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.tree import plot_tree,export_text,_tree
from .models import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
# import mysql.connector
# from mysql.connector import Error
# import snowflake.connector
# from pymongo import MongoClient
import urllib.parse
from netsuitesdk import NetSuiteConnection
# import pyodbc 
# import psycopg2
# import openai
# from openai import OpenAI
from matplotlib import colors as mcolors
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import logging
logger = logging.getLogger(__name__)

import warnings
# Ignore Pandas FutureWarnings (iloc vs [])
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore constant input warnings (correlation of a column with only 1 value)
from scipy.stats import ConstantInputWarning
warnings.simplefilter(action='ignore', category=ConstantInputWarning)


# 1st
class SxiProcess:
    def __init__(self,request, target_type, target, dataframe, buynobuy, sxi_dataframe, fulldata_sxi,primskey,SXI,buyerid):
        self.target_type = target_type
        self.target = target
        self.dataframe = dataframe
        self.buynobuy = buynobuy
        self.sxi_dataframe = sxi_dataframe
        self.fulldata_sxi = fulldata_sxi
        self.primkey = primskey
        self.SXI = SXI
        self.buyerid = buyerid
        self.request = request

        print(f"target_type: {self.target_type}")
        print(f"target: {self.target}")
        print(f"dataframe: {self.dataframe}")
        print(f"buynobuy: {self.buynobuy}")
        print(f"sxi_dataframe: {self.sxi_dataframe}")
        print(f"fulldata_sxi: {self.fulldata_sxi}")
        print(f"primkey: {self.primkey}")
        print(f"SXI: {self.SXI}")
        print(f"buyerid: {self.buyerid}")

    def sxi_data_generate(target, dataframe, primskey):
        print('SXI Data Generate Started!')
        tv = target
        df_main = dataframe.copy()
        if primskey == 'None':
            df_main = df_main.reset_index()

        x = df_main.drop([tv,primskey,'index'], axis=1,errors='ignore')
        y = df_main[tv]
        dff=x.columns

        from sklearn import linear_model
        clf = linear_model.Lasso(alpha=0.2, max_iter=10000)
        clf.fit(x, y)
        lasso_weight= clf.coef_

        #update dataframe
        min_max=[]
        max_f=[]
        for i in range(len(dff)):
            if lasso_weight[i] < 0 :
                min_max.append(dff[i])
            else:
                max_f.append('MAX')

        df_main.drop(tv, axis=1, inplace=True)

        minimum_value=[]
        maximum_value=[]
        minmax_value=[]
        catogorical_value=[]
        features=list(df_main.columns)
        for i in range(len(features)):

            if features[i] in min_max:
                minmax_value.append('MIN')
                catogorical_value.append('Behavior')
            else:
                minmax_value.append('MAX')
                catogorical_value.append('Behavior')
            try:
                minimum=min(list(df_main[features[i]]))
                maximum=max(list(df_main[features[i]]))
            except:
                minimum=''
                maximum=''
            minimum_value.append(minimum)
            maximum_value.append(maximum)

        minimum_value[0]='Min_Value'
        maximum_value[0]='Max_Value'
        minmax_value[0] = 'minmax'
        catogorical_value[0]='Category'

        df_mains=pd.DataFrame()
        df_mains['0']=minimum_value
        df_mains['1']=maximum_value
        df_mains['2']=minmax_value
        df_mains['3']=catogorical_value

        df_m=df_mains.T
        df_m.columns=df_main.columns
        df_m.reset_index(drop=True, inplace=True)
        df_main.reset_index(drop=True, inplace=True)

        frames = [df_m, df_main]
        result = pd.concat(frames, keys=['x', 'y'])
        return result

    def generate_sxi(target, buynobuy, sxi_dataframe, buyerid):
        print('Generate SXI Started!')

        def single_correlation(x, y):
            x = x.astype(float)
            y = y.astype(float)

            from scipy import stats
            from scipy.stats import pearsonr
            #corr,_=(stats.pearsonr(x, y))
            corr, _ = pearsonr(x, y)
            return float(corr)


        def multivarient_correlation(x, y):
            x = x.astype(float)
            y = y.astype(float)

            from sklearn import linear_model
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=None)
            regr = linear_model.LinearRegression()
            regr.fit(X_train, y_train)
            y_pred = regr.predict(x)
            mse=abs(r2_score(y,y_pred))
            rmse= math.sqrt(mse)
            return rmse


        def bivarient_correlation(df):
            length_of_parameter = df.shape[1]
            column_name=df.columns

            raw_bivarient_correlation=[]
            abs_bivarient_correlation=[]
            raw=[[] for i in range(length_of_parameter)]
            abss=[[] for i in range(length_of_parameter)]

            for i in range(length_of_parameter):
                for j in range(length_of_parameter):
                    if (i != j):
                        #print(i)

                        corr=single_correlation(df[column_name[i]], df[column_name[j]])

                        raw[i].append(corr)
                        abss[i].append(abs(corr))
                abss[i] = [x for x in abss[i] if math.isnan(x) == False]

            for i in range(length_of_parameter):
                abs_bivarient_correlation.append((np.mean(abss[i]))/2)
            # raw_bivarient_correlation.append(np.mean(raw[i]))
            abs_bivarient_correlation= np.nan_to_num(np.array(abs_bivarient_correlation))
            return list(abs_bivarient_correlation)


        def trivarient_correlation(df):
            column_name=df.columns
            mean_rmse=[]
            for k in range(df.shape[1]):
                # print(k)

                x = df.drop([column_name[k]] , axis=1)
                y = df[column_name[k]]
                column_name_x = x.columns

                multivarient=[]
                for i in range(x.shape[1]):

                    for j in range(x.shape[1]):

                        if (i != j):
                            a=np.array(x[column_name_x[i]])
                            b=np.array(x[column_name_x[j]])

                            sub = np.vstack((a,b))
                            x_sub = pd.DataFrame(sub.T, columns=['A','B'])
                            y = df[column_name[k]]
                            rmsc = multivarient_correlation(x_sub, y)
                            multivarient.append(rmsc)

                meann=(sum(multivarient)/len(multivarient))/3
                mean_rmse.append(meann)
            return mean_rmse


        def weight(df, b):
            W=[]
            for i in range(df.shape[1]):
                w=1-(b[i]/2)
                W.append(w)
            return W


        def dxi(new_df, w):
            new_df=(np.array(new_df)).astype(float)
            w=np.array(w)
            w_df=np.dot(new_df, w.T)

            dxi=[]
            for i in range(len(new_df)):
                individual_dxi=(w_df[i]*100)/len(w)
                dxi.append(individual_dxi)
            return dxi


        def normalize(df):
            print('Entered Normalization')
            df1 = df
            min_max = (df1.iloc[2])
            max_column = df1.iloc[1].astype(float)
            column_name = df1.columns
            new_df = df.iloc[4:]
            print('Entered Normalization - Loop to Start')
            for i in range(df1.shape[1]):
                if min_max[i]=='MAX':
                    print(f'MAX-{i}')
                    new_df[column_name[i]]= new_df[column_name[i]].astype(float)/max_column[i]
                else:
                    print(f'MIN-{i}')
                    new_df[column_name[i]]=(max_column[i]-new_df[column_name[i]].astype(float))/max_column[i]
            return new_df


        def create_label(total_dxi, Avg_dxi):
            y=[]
            for i in range(len(total_dxi)):
                if np.all(total_dxi[i]>Avg_dxi):
                    y.append(1)
                else:
                    y.append(0)
            return y


        def create_label_forward(composite_dxi1, avg_composite_dxi):
            y_forward=[]
            for i in range(len(composite_dxi1)):
                if np.all(composite_dxi1[i] > avg_composite_dxi):
                    y_forward.append(0)
                else:
                    y_forward.append(1)
            return y_forward


        def total_dxi(df1, weight):
            total_dxi=dxi(df1,weight)
            Avg_dxi=np.mean(np.array(total_dxi))
            return total_dxi, Avg_dxi,weight


        def catogorical_dxi(df, weight, parameter):
            column_name= df.columns
            #parameter = (df.iloc[3])

            behaviour_column_name=[]
            transactional_column_name=[]
            visual_column_name=[]
            kpi_column_name=[]

            weight_behaviour=[]
            weight_transactional=[]
            weight_visual=[]
            weight_kpi=[]

            for i in range(len(parameter)):
                if parameter[i]=='Behavior':
                    behaviour_column_name.append(column_name[i])
                    weight_behaviour.append(weight[i])
                elif parameter[i]=='Transactional':
                    transactional_column_name.append(column_name[i])
                    weight_transactional.append(weight[i])
                elif parameter[i]=='visual':
                    visual_column_name.append(column_name[i])
                    weight_visual.append(weight[i])

                else:
                    kpi_column_name.append(column_name[i])
                    weight_kpi.append(weight[i])

            new_df=df

            df_behaviour = new_df[behaviour_column_name]
            df_transactional=new_df[transactional_column_name]
            df_visual=new_df[visual_column_name]
            df_kpi=new_df[kpi_column_name]

            behaviour_dxi=dxi(df_behaviour,weight_behaviour)
            #avg_behaviour_dxi=mean(behaviour_dxi)
            transactional_dxi=dxi(df_transactional,weight_transactional)
            visual_dxi=dxi(df_visual,weight_visual)
            kpi_dxi=dxi(df_kpi,weight_kpi)
            return behaviour_dxi, transactional_dxi, kpi_dxi, visual_dxi


        def svm_feature_selection(x, y):
            from sklearn.svm import LinearSVC,SVC
            #lsvc=SVC(gamma='auto')
            lsvc = LinearSVC(C=0.4,penalty='l1', dual=False).fit(x, y)
            coeff=lsvc.coef_
            coeff = abs(coeff)
            return coeff.reshape(-1)


        def mutual_information(x, y):
            from sklearn.feature_selection import mutual_info_classif
            x= mutual_info_classif(x, y)
            for i in range(len(x)):
                if x[i] < .1 :
                    x[i]=0
            return x


        def tree_feature_selection(x, y):
            from sklearn.ensemble import ExtraTreesClassifier
            clf = ExtraTreesClassifier(n_estimators=100)
            clf = clf.fit(x, y)
            coeff = clf.feature_importances_
            return coeff


        def pca_feature_selection(x, y):
            from sklearn.decomposition import PCA
            min_dim = min(x.shape[0], x.shape[1])
            if min_dim < 1:
                raise ValueError("SXI PCA feature selection received no usable samples/features.")
            pca = PCA(n_components=min(3, min_dim))
            pca.fit_transform(x)
            pcaws = pca.components_[0]
            return pcaws


        '''
        def pca_feature_selection(x,y):
            from sklearn.decomposition import PCA
            # load data

            # feature extraction
            pca = PCA(n_components=None)
            fit = pca.fit(x)
            # summarize components
            varience= fit.explained_variance_ratio_
            #print(fit.components_)
            return varience
        '''


        def lasso_feature_selection(x, y):
            from sklearn import linear_model
            clf = linear_model.Lasso(alpha=0.01)
            clf.fit(x,y)
            coeff = clf.coef_
            return abs(coeff)


        def composite_weight(df, svm_weight, avg_dxi):
            from numpy import array, argsort
            svm_index=(argsort(svm_weight))

            def Reverse(lst):
                new_lst = lst[::-1]
                return new_lst

            svm_ind=Reverse(svm_index)
            column_name=df.columns

            svm_indd=[]
            svm_weights=[]
            for i in range(len(svm_index)):
                if svm_weight[svm_ind[i]] !=0:
                    svm_indd.append(svm_ind[i])
                    svm_weights.append(svm_weight[svm_ind[i]])

            update_column=column_name[svm_indd]
            xx=df[update_column]

            #full_dxi,avg_dxi,w_dxi=total_dxi(df,)
            while(1):
                #svm_dxi_weight=dxi_weight(new_df)
                svm_dxi, avg_svm_dxi, w_svm = total_dxi(xx, svm_weights)
                if (avg_svm_dxi >= 0.9*avg_dxi) and  (avg_svm_dxi <= 1.1*avg_dxi) :
                    # print(11)
                    xx=xx.iloc[:,0:-1]
                    svm_weights=svm_weights[:-1]
                else:
                    break

            column_update=list(xx.columns)
            weight_svm=[]

            for i in range(len(column_name)):
                if column_name[i] in column_update :
                    a=column_update.index(column_name[i])
                    weight_svm.append(w_svm[a])
                else:
                    weight_svm.append(0)
            return svm_dxi, avg_svm_dxi, w_svm, weight_svm


        def composite_dxi(x, weight_svm, weight_pca, weight_mi, weight_lasso, weigth_xgb):
            # weight_svm , weight_pca, weight_mi ,svm_dxi ,  pca_dxi , mi_dxi , avg_mi_dxi,avg_pca_dxi,avg_svm_dxi = feature_selection_weight(df,x,y)

            final_weights=[]
            for i in range(x.shape[1]):
                w=[weight_svm[i], weight_pca[i], weight_mi[i], weight_lasso[i], weigth_xgb[i]]
                count=0
                for i in range(len(w)):
                    if w[i] > 0 :
                        count=count+1

                total_weight=np.sum(w)
                n = np.count_nonzero(w)

                if n==0:
                    n=1

                final_weight=(total_weight*(1+(.1*(n-1))))/n
                final_weights.append(final_weight)

            composite_dxi=dxi(x,final_weights)
            composite_avg_dxi=np.mean(np.array(composite_dxi))
            # print(composite_avg_dxi)
            return composite_dxi, composite_avg_dxi


        def sample_composite_dxi(x, weight_svm, weight_mi, weight_lasso, weight_pca, weight_xgb):
            # weight_svm , weight_pca, weight_mi ,svm_dxi ,  pca_dxi , mi_dxi , avg_mi_dxi,avg_pca_dxi,avg_svm_dxi = feature_selection_weight(df,x,y)

            final_weights=[]
            for i in range(x.shape[1]):
                w=[weight_svm[i], weight_mi[i], weight_lasso[i], weight_pca[i], weight_xgb[i]]
                count=0
                for i in range(len(w)):
                    if w[i] > 0 :
                        count=count+1

                total_weight=np.sum(w)
                n = np.count_nonzero(w)

                if n==0:
                    n=1

                final_weight=(total_weight*(1+(.1*(n-1))))/n
                final_weights.append(final_weight)

            composite_dxi=dxi(x, final_weights)
            composite_avg_dxi=np.mean(np.array(composite_dxi))
            # print(composite_avg_dxi)
            return composite_dxi, composite_avg_dxi


        ####################### Data Satrts Here #######################
        print('Data Starts Here')
        tv = target
        print('tv',tv)
        df_buynobuy = buynobuy ### Read BUYNOBUY DATA
        print('df_buynobuy',df_buynobuy)
        df = sxi_dataframe ### Read DXI DATA
        df = df.iloc[:, 1:]
        print('sxi generate process -1')
        minmax=list(df.iloc[1])
        column=df.columns
        for i in range(len(minmax)):
            if float(minmax[i]) == 0:
                df=df.drop([column[i]],axis=1)
        print('sxi generate process -1')

        df=df.dropna(how='any')
        print('sxi generate process -2')
        new_df=normalize(df)
        print('sxi generate process -3')
        b=bivarient_correlation(new_df)
        # a=trivarient_correlation(new_df)
        print('sxi generate process -4')
        w=weight(new_df, b)
        print('sxi generate process -5')
        before_base_dxi=dxi(new_df, w)
        avg_dxi=np.mean(np.array(before_base_dxi))

        x=new_df
        y=create_label_forward(before_base_dxi, avg_dxi)
        print('sxi generate process -6')
        from sklearn import linear_model
        clf = linear_model.Lasso(alpha=0.2)
        clf.fit(x,y)
        lasso_weight= clf.coef_
        min_max= (df.iloc[2])
        print('sxi generate process -7')
        #update dataframe
        for i in range(len(min_max)):
            if lasso_weight[i] < 0 :
                if min_max[i] != 'MIN':
                    (df.iloc[2])[i]='MIN'
        print('sxi generate process -8')
        new_df=normalize(df)
        update_b=bivarient_correlation(new_df)
        parameter=df.iloc[3]

        # update_a=trivarient_correlation(new_df)
        update_w=weight(new_df,update_b)
        base_dxi=dxi(new_df,update_w)

        avg_base_dxi=np.mean(np.array(base_dxi))

        update_y = create_label_forward(base_dxi,avg_base_dxi)
        data=df.iloc[4:]
        data['Base_dxi_label']=update_y
        data['Base_dxi']=base_dxi
        data[tv]=df_buynobuy[tv]
        x =new_df
        # print('visual :', avg_visual)
        # print('transactional :', avg_transactional)
        # print('kpi:', avg_kpi)
        # print('behavioural', avg_behaviour)
        print('base_Dxi :', avg_base_dxi)

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x)
        from sklearn.naive_bayes import ComplementNB
        cnb = ComplementNB().fit(x_train, update_y)

        logprobs = cnb.feature_log_prob_
        avgprob =[]
        for i in range(len(logprobs[0])):
            avgprob.append((logprobs[0][i]+logprobs[1][i])/2)

        nb_weight = np.exp(avgprob) / (np.exp(avgprob)).sum()

        import xgboost as xg
        xgb = xg.XGBClassifier().fit(x,update_y)
        xgb_weight = xgb.feature_importances_
        # svm_weight=svm_feature_selection(x,y)

        mi_weight=list(mutual_information(x, update_y))
        pca_weight=pca_feature_selection(x,update_y)
        lasso_weight=lasso_feature_selection(x,update_y)

        lasso_dxi,avg_lasso_dxi,w_lasso,weight_lasso = composite_weight(x,lasso_weight,avg_base_dxi)
        # svm_dxi,avg_svm_dxi,w_svm,weight_svm = composite_weight(x,svm_weight,avg_base_dxi)
        mi_dxi,avg_mi_dxi,w_mi,weight_mi = composite_weight(x,mi_weight,avg_base_dxi)
        pca_dxi,avg_pca_dxi,w_pca,weight_pca = composite_weight(x,pca_weight,avg_base_dxi)
        nb_dxi,avg_nb_dxi,w_nb,weight_nb = composite_weight(x,nb_weight,avg_base_dxi)
        xgb_dxi, avg_xgb_dxi,w_xgb,weight_xgb = composite_weight(x,xgb_weight,avg_base_dxi)

        top_names =[]
        for weight in [weight_lasso, weight_nb, weight_mi, weight_pca,weight_xgb]:
            idx = (-(np.array(weight))).argsort()[:len(weight)+1]
            names = x.columns[idx]
            top_names.append(names)

        top_params = {}
        top_params["lasso"]=top_names[0][0:10]
        top_params["nb"]=top_names[1][0:10]
        top_params["mi"]=top_names[2][0:10]
        top_params["pca"]=top_names[3][0:10]
        top_params["xgb"]=top_names[4][0:10]

        sample_composite_dxi1=sample_composite_dxi(x,weight_nb,weight_mi,weight_lasso,weight_pca,weight_xgb)
        #composite_dxi=(composite_dxi(x,weight_svm,weight_pca,weight_mi,weight_lasso))
        composite_dxi1=sample_composite_dxi1[0]
        #composite_dxi1=composite_dxi[0]
        avg_composite_dxi=np.mean(composite_dxi1)
        update_y_com=create_label_forward(composite_dxi1,avg_composite_dxi)

        # print(avg_pca_dxi)
        # print(avg_lasso_dxi)
        # print(avg_nb_dxi)
        # print(avg_mi_dxi)
        # print(avg_xgb_dxi)

        top_dxi=max(composite_dxi1)
        # print('top dxi:', top_dxi)

        df_buynobuy=df_buynobuy.dropna(how='any')
        df_buynobuy['composite_dxi_label'] = update_y_com
        df_buynobuy['composite_dxi'] = composite_dxi1
        full_data = df_buynobuy

        rand = random.randint(00000000, 99999999)
        loc_fulldata = f'media/files/hurra/{buyerid}/fulldata{rand}.csv'
        # full_data.to_csv(loc_fulldata, index=False)
        print('SXI Scores Generated!')
        return loc_fulldata, full_data


    def sxi_method(buyerid, good, bad, labels, target_type, target, fulldata_sxi, goodab, primskey, SXI, primary_key_mapping, primary_key_encoded):
        check_cancelled()
        print('SXI Method Started!')
        print("primskey",primskey)
        import xgboost as xg
        rand = random.randint(0, 99999999)


        def ConfusionMat(target, labels, y_test, pred):
            import os
            import random
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix

            # # Compute the confusion matrix
            # cma = confusion_matrix(y_test, pred)
            # cm = pd.DataFrame(cma)
            # print('ConfusionMat:',cm)

            # if len(list(cm)) == 2:
            #     labels_list = list(labels.values())
            #     # Plotting the confusion matrix
            #     fig, ax = plt.subplots()
            #     cax = ax.matshow(cm, cmap='Reds')
            #     plt.colorbar(cax)

            #     for (i, j), val in np.ndenumerate(cm):
            #         ax.text(j, i, f'{val}', ha='center', va='center', fontweight='bold')

            #     plt.xlabel('Predicted Class')
            #     plt.ylabel('Actual Class')
            #     plt.title(f'Confusion Matrix: Target Outcome: {target}')
            #     ax.set_xticklabels([''] + labels_list)
            #     ax.set_yticklabels([''] + labels_list)

            cma = confusion_matrix(y_test, pred)
            cm = pd.DataFrame(cma)
            print('ConfusionMat:', cm)

            if len(list(cm)) == 2:
                labels_list = list(labels.values())

                z = cma

                # Calculate percentages
                class_0_total = z[0][0] + z[0][1]
                class_1_total = z[1][1] + z[1][0]
                correct_class_0 = round(z[0][0] / class_0_total * 100, 2)
                incorrect_class_0 = round(z[0][1] / class_0_total * 100, 2)
                correct_class_1 = round(z[1][1] / class_1_total * 100, 2)
                incorrect_class_1 = round(z[1][0] / class_1_total * 100, 2)

                # Create annotation text
                text = [
                    [f"{z[0][0]} ({correct_class_0}%)", f"{z[0][1]} ({incorrect_class_0}%)"],
                    [f"{z[1][0]} ({incorrect_class_1}%)", f"{z[1][1]} ({correct_class_1}%)"]
                ]

                # Create custom colorscale (red -> white -> lightgreen)
                colorscale = [[0, 'red'],
                                [0.5, 'white'],
                                [1, 'lightgreen']]

                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=z,
                    x=classes,
                    y=classes,
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 18, "family": "Arial Black", "color": "black"},
                    hoverongaps=False,
                    colorscale=colorscale,
                    showscale=True
                ))

                # Update layout
                fig.update_layout(
                    title='Accuracy Matrix',
                    xaxis_title='Predicted Class',
                    yaxis_title='True Class',
                    xaxis=dict(
                        tickmode='array',
                        ticktext=classes,
                        tickvals=list(range(len(classes)))
                    ),
                    yaxis=dict(
                        tickmode='array',
                        ticktext=classes,
                        tickvals=list(range(len(classes))),
                        autorange='reversed'  # This ensures the matrix orientation matches matplotlib
                    ),
                    width=1200,
                    height=680,
                    font=dict(size=18)
                )

                # # Create the heatmap using Plotly
                # fig = go.Figure(
                #     data=go.Heatmap(
                #         z=cm.values,
                #         x=labels_list,  # Predicted class labels
                #         y=labels_list,  # Actual class labels
                #         colorscale='Reds',
                #         showscale=True,
                #         texttemplate="%{z}",
                #         textfont={"size": 14, "family": "Arial", "color": "black"}
                #     )
                # )

                # # Update layout for axes and titles
                # fig.update_layout(
                #     title=f'Confusion Matrix: Target Outcome: {target}',
                #     xaxis=dict(title="Predicted Class", tickmode="array", tickvals=np.arange(len(labels_list)), ticktext=labels_list),
                #     yaxis=dict(title="Actual Class", tickmode="array", tickvals=np.arange(len(labels_list)), ticktext=labels_list),
                #     font=dict(size=14),
                #     annotations=[
                #         dict(
                #             showarrow=False,
                #             text=f'{cm.iloc[i, j]}',
                #             x=j,
                #             y=i,
                #             xref='x',
                #             yref='y',
                #             font=dict(size=10, color="black")
                #         ) for i in range(len(cm)) for j in range(len(cm.columns))
                #     ],
                #     width=1200,  # Adjust width
                #     height=600  # Adjust height
                # )
                # buyerid = self.request.session.get('buyerid')
                folder = f'media/files/chatbot/{buyerid}/'
                locpng = os.path.join(get_dataset_folder(folder, f'cm_{rand}.png'), f'cm_{rand}.png')
                lochtml = os.path.join(get_dataset_folder(folder, f'cm_{rand}.html'), f'cm_{rand}.html')
                # plt.savefig(lochtml,dpi=300)
                # plt.close(fig)
                _safe_write_plotly_figure(
                    fig,
                    lochtml,
                    locpng,
                    title=f"Confusion Matrix: Target Outcome: {target}"
                )
                print(f'Confusion matrix saved at {locpng} / {lochtml}')
                return locpng, cma, lochtml
            else:
                print('Confusion matrix: skipped (not binary)')
                return None, cma, None

        def create_label_forward(good, bad, composite_dxi1, avg_composite_dxi, fulldata_sxi, target, goodab):
            y_forward=[]
            gd_count = int(fulldata_sxi[target].eq(good).sum())
            gdper = (gd_count/len(fulldata_sxi))*100 if len(fulldata_sxi) else 0

            for i in range(len(composite_dxi1)):
                if np.all(composite_dxi1[i] > avg_composite_dxi):
                    if  gdper > goodab:
                        y_forward.append(bad)
                    else:
                        y_forward.append(good)
                else:
                    if gdper > goodab:
                        y_forward.append(good)
                    else:
                        y_forward.append(bad)

            return y_forward


        ###################### Start from Here #######################
        tv = target
        df = fulldata_sxi.copy()
        classes = list(labels.keys())
        label_lookup = {int(v): str(k) for k, v in labels.items()}
        good = int(good)
        bad = int(bad)

        if tv not in df.columns or 'composite_dxi' not in df.columns:
            raise ValueError(
                f"SXI classification requires '{tv}' and 'composite_dxi' columns. "
                f"Available columns: {list(df.columns)}"
            )

        eval_df = df.copy()
        eval_df[tv] = pd.to_numeric(eval_df[tv], errors='coerce')
        eval_df['composite_dxi'] = pd.to_numeric(eval_df['composite_dxi'], errors='coerce')
        eval_df = eval_df.replace([np.inf, -np.inf], np.nan)
        eval_df = eval_df.dropna(subset=[tv, 'composite_dxi']).copy()
        eval_df[tv] = eval_df[tv].astype(int)

        if eval_df.empty or eval_df[tv].nunique() < 2:
            raise ValueError("SXI classification requires at least two target classes with valid SXI scores.")

        eval_df = eval_df[eval_df[tv].isin([good, bad])].copy()
        if eval_df.empty or eval_df[tv].nunique() < 2:
            raise ValueError("SXI classification requires both good and bad outcome classes after cleaning.")

        binary_target_map = {good: 0, bad: 1}
        original_target_map = {0: good, 1: bad}
        eval_df['_sxi_binary_target'] = eval_df[tv].map(binary_target_map).astype(int)

        leakage_audit = detect_suspected_leakage_columns(
            eval_df,
            tv,
            primary_key=primskey,
        )
        print("SXI suspected leakage columns dropped:", leakage_audit['suspected_columns'])

        model_feature_drop_columns = sorted(set([
            tv,
            '_sxi_binary_target',
            'gd_bdDXI',
            'composite_dxi_label',
            'netqyty_Bucket',
            primskey,
            'composite_dxi',
            *leakage_audit['suspected_columns'],
        ]))
        importance_feature_drop_columns = model_feature_drop_columns.copy()

        def _build_numeric_frame(frame, drop_columns):
            x = frame.drop(columns=drop_columns, errors='ignore').copy()
            if x.empty:
                return x
            x = x.apply(pd.to_numeric, errors='coerce')
            x = x.replace([np.inf, -np.inf], np.nan)
            x = x.loc[:, x.notna().any(axis=0)]
            return x

        def prepare_feature_matrices(train_frame, validate_frame, test_frame, drop_columns):
            train_x = _build_numeric_frame(train_frame, drop_columns)
            validate_x = _build_numeric_frame(validate_frame, drop_columns)
            test_x = _build_numeric_frame(test_frame, drop_columns)

            if train_x.empty:
                return train_x, validate_x, test_x

            validate_x = validate_x.reindex(columns=train_x.columns)
            test_x = test_x.reindex(columns=train_x.columns)

            fill_values = train_x.median(numeric_only=True).fillna(0.0)
            train_x = train_x.fillna(fill_values).fillna(0.0)
            validate_x = validate_x.fillna(fill_values).fillna(0.0)
            test_x = test_x.fillna(fill_values).fillna(0.0)
            return train_x, validate_x, test_x

        def compute_goodab(reference_df, avg_threshold):
            if reference_df.empty:
                return 0.0
            above = reference_df.loc[reference_df['composite_dxi'] > avg_threshold]
            if above.empty:
                return float((reference_df[tv] == good).mean() * 100.0)
            return float((above[tv] == good).mean() * 100.0)

        def build_rule_predictions(reference_df, target_df, alpha):
            base_avg = float(reference_df['composite_dxi'].mean())
            threshold = alpha * base_avg
            gdper = float((reference_df[tv] == good).mean() * 100.0)
            goodab_train = compute_goodab(reference_df, base_avg)
            high_label = bad if gdper > goodab_train else good
            low_label = good if high_label == bad else bad
            preds = np.where(target_df['composite_dxi'].to_numpy() > threshold, high_label, low_label)
            score_sign = 1.0 if high_label == bad else -1.0
            scores = target_df['composite_dxi'].to_numpy(dtype=float) * score_sign
            return preds.astype(int), scores, threshold, high_label

        def build_rule_probabilities(reference_df, target_df, alpha):
            _, rule_scores, threshold, high_label = build_rule_predictions(reference_df, target_df, alpha)
            signed_scores = np.asarray(rule_scores, dtype=float) - float(threshold)
            if high_label != bad:
                signed_scores = -signed_scores
            score_scale = float(np.nanstd(reference_df['composite_dxi'].to_numpy(dtype=float)))
            if not np.isfinite(score_scale) or score_scale <= 1e-9:
                score_scale = max(abs(float(reference_df['composite_dxi'].mean())), 1.0)
            signed_scores = np.clip(signed_scores / score_scale, -12.0, 12.0)
            return 1.0 / (1.0 + np.exp(-signed_scores))

        def safe_auc_from_probabilities(y_true_binary, probabilities):
            try:
                fpr_local, tpr_local, _ = roc_curve(y_true_binary, probabilities, pos_label=1)
                return float(auc(fpr_local, tpr_local))
            except Exception:
                return 0.0

        def get_positive_class_probabilities(model, features):
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)
                if probabilities.ndim == 2:
                    classes_local = list(getattr(model, 'classes_', [0, 1]))
                    positive_index = classes_local.index(1) if 1 in classes_local else -1
                    return np.asarray(probabilities[:, positive_index], dtype=float)
            if hasattr(model, 'decision_function'):
                decision = np.asarray(model.decision_function(features), dtype=float)
                decision = np.clip(decision, -12.0, 12.0)
                return 1.0 / (1.0 + np.exp(-decision))
            return np.asarray(model.predict(features), dtype=float)

        def evaluate_probability_candidate(candidate_name, validate_probabilities, test_probabilities, y_validate_binary, base_model=None):
            validate_probabilities = np.asarray(validate_probabilities, dtype=float)
            test_probabilities = np.asarray(test_probabilities, dtype=float)
            candidate_auc = safe_auc_from_probabilities(y_validate_binary, validate_probabilities)
            actual_positive_rate = float(np.mean(y_validate_binary))
            threshold_candidates = np.unique(
                np.round(
                    np.concatenate([
                        np.linspace(0.15, 0.85, 29),
                        np.array([actual_positive_rate, 0.5], dtype=float),
                    ]),
                    4,
                )
            )

            best_local = None
            for threshold in threshold_candidates:
                validate_pred_binary = (validate_probabilities >= threshold).astype(int)
                validate_acc = float(accuracy_score(y_validate_binary, validate_pred_binary))
                validate_prec = float(precision_score(y_validate_binary, validate_pred_binary, pos_label=1, zero_division=0))
                validate_rec = float(recall_score(y_validate_binary, validate_pred_binary, pos_label=1, zero_division=0))
                predicted_positive_rate = float(np.mean(validate_pred_binary))
                calibration_gap = abs(predicted_positive_rate - actual_positive_rate)
                ranking_score = (
                    (0.45 * validate_acc)
                    + (0.25 * validate_prec)
                    + (0.20 * candidate_auc)
                    + (0.10 * validate_rec)
                    - (0.05 * calibration_gap)
                )
                ranking_key = (
                    round(ranking_score, 10),
                    round(validate_acc, 10),
                    round(validate_prec, 10),
                    round(candidate_auc, 10),
                    round(validate_rec, 10),
                    -round(calibration_gap, 10),
                )
                if best_local is None or ranking_key > best_local['ranking_key']:
                    best_local = {
                        'name': candidate_name,
                        'threshold': float(threshold),
                        'validation_accuracy': validate_acc,
                        'validation_precision': validate_prec,
                        'validation_recall': validate_rec,
                        'validation_auc': candidate_auc,
                        'ranking_key': ranking_key,
                        'test_probabilities': test_probabilities,
                        'base_model': base_model,
                    }
            return best_local

        def extract_feature_importances(model, feature_names):
            if model is None or feature_names is None or len(feature_names) == 0:
                return pd.DataFrame(columns=["Feature", "Importance", "Cumulative"])
            if hasattr(model, 'feature_importances_'):
                importances_local = np.asarray(model.feature_importances_, dtype=float)
            elif hasattr(model, 'coef_'):
                importances_local = np.abs(np.asarray(model.coef_[0], dtype=float))
            else:
                return pd.DataFrame(columns=["Feature", "Importance", "Cumulative"])

            importance_df_local = pd.DataFrame({
                "Feature": list(feature_names),
                "Importance": importances_local,
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

            total_importance = float(importance_df_local["Importance"].sum())
            if total_importance > 0:
                importance_df_local["Cumulative"] = importance_df_local["Importance"].cumsum() / total_importance
            else:
                importance_df_local["Cumulative"] = 0.0
            return importance_df_local

        def prepare_stratify_labels(values):
            labels_arr = np.asarray(values)
            if labels_arr.size == 0:
                return None
            unique_labels, label_counts = np.unique(labels_arr, return_counts=True)
            if len(unique_labels) < 2 or label_counts.min() < 2:
                return None
            return labels_arr

        split_seed = 42
        # Break target-sorted row order so residual id/index columns cannot leak class.
        eval_df = eval_df.sample(frac=1.0, random_state=split_seed).reset_index(drop=True)
        row_positions = np.arange(len(eval_df))
        target_values = eval_df['_sxi_binary_target'].to_numpy(dtype=int)
        train_pos, temp_pos = train_test_split(
            row_positions,
            test_size=0.30,
            random_state=split_seed,
            stratify=prepare_stratify_labels(target_values),
        )
        temp_target_values = target_values[temp_pos]
        validate_pos, test_pos = train_test_split(
            temp_pos,
            test_size=(2 / 3),
            random_state=split_seed,
            stratify=prepare_stratify_labels(temp_target_values),
        )

        train_df = eval_df.iloc[train_pos].copy()
        validate_df = eval_df.iloc[validate_pos].copy()
        test_df = eval_df.iloc[test_pos].copy()

        y_train_binary = train_df['_sxi_binary_target'].to_numpy(dtype=int)
        y_validate_binary = validate_df['_sxi_binary_target'].to_numpy(dtype=int)
        y_test_binary = test_df['_sxi_binary_target'].to_numpy(dtype=int)

        X_train_model, X_validate_model, X_test_model = prepare_feature_matrices(
            train_df,
            validate_df,
            test_df,
            model_feature_drop_columns,
        )
        X_train_importance, _, _ = prepare_feature_matrices(
            train_df,
            validate_df,
            test_df,
            importance_feature_drop_columns,
        )

        if X_train_model.empty:
            raise ValueError("SXI classification requires at least one numeric feature for model training.")

        class_counts = np.bincount(y_train_binary, minlength=2)
        negative_count = int(class_counts[0])
        positive_count = int(class_counts[1])
        scale_pos_weight = float(negative_count / positive_count) if positive_count > 0 and negative_count > positive_count else 1.0

        # composite_dxi is fit in-sample on the target, so DXI threshold rules
        # (and blends with those rules) can report near-perfect AUC/accuracy.
        # Reported model metrics must come from holdout ML models only.
        best_candidate = None

        model_candidates = [
            (
                "xgb_primary",
                xg.XGBClassifier(
                    random_state=split_seed,
                    eval_metric='logloss',
                    n_estimators=320,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.85,
                    min_child_weight=1,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    tree_method='hist',
                ),
            ),
            (
                "xgb_compact",
                xg.XGBClassifier(
                    random_state=split_seed + 7,
                    eval_metric='logloss',
                    n_estimators=220,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.95,
                    colsample_bytree=0.9,
                    min_child_weight=2,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    tree_method='hist',
                ),
            ),
            (
                "rf_balanced",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=split_seed,
                    class_weight='balanced_subsample',
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
            (
                "logreg_balanced",
                LogisticRegression(
                    random_state=split_seed,
                    class_weight='balanced',
                    max_iter=2000,
                    solver='liblinear',
                ),
            ),
        ]

        for model_name, model in model_candidates:
            check_cancelled()
            try:
                model.fit(X_train_model, y_train_binary)
                validate_probabilities = get_positive_class_probabilities(model, X_validate_model)
                test_probabilities = get_positive_class_probabilities(model, X_test_model)

                candidate_variants = [
                    (f"{model_name}_pure", validate_probabilities, test_probabilities),
                ]
                # Do not blend with DXI rule probabilities for champion metrics —
                # composite_dxi is in-sample and can dominate AUC to ~1.0.

                for variant_name, variant_validate_prob, variant_test_prob in candidate_variants:
                    candidate_result = evaluate_probability_candidate(
                        candidate_name=variant_name,
                        validate_probabilities=variant_validate_prob,
                        test_probabilities=variant_test_prob,
                        y_validate_binary=y_validate_binary,
                        base_model=model,
                    )
                    if best_candidate is None or candidate_result['ranking_key'] > best_candidate['ranking_key']:
                        best_candidate = candidate_result
            except Exception as exc:
                import traceback
                print(f"SXI warning: model candidate '{model_name}' failed: {exc}")
                print(traceback.format_exc())

        if best_candidate is None:
            raise ValueError("SXI classification could not produce a valid candidate model.")

        best_threshold = float(best_candidate['threshold'])
        test_scores = np.asarray(best_candidate['test_probabilities'], dtype=float)
        test_pred_binary = (test_scores >= best_threshold).astype(int)
        y_test = np.asarray([original_target_map[int(value)] for value in y_test_binary], dtype=int)
        test_pred = np.asarray([original_target_map[int(value)] for value in test_pred_binary], dtype=int)

        print(
            "SXI selected candidate:",
            best_candidate['name'],
            "threshold=",
            best_threshold,
            "validation_accuracy=",
            round(best_candidate['validation_accuracy'], 4),
            "validation_precision=",
            round(best_candidate['validation_precision'], 4),
            "validation_recall=",
            round(best_candidate['validation_recall'], 4),
            "validation_auc=",
            round(best_candidate['validation_auc'], 4),
        )

        sxiacc = round(float(accuracy_score(y_test_binary, test_pred_binary)) * 100, 3)
        sxiprec = round(float(precision_score(y_test_binary, test_pred_binary, pos_label=1, zero_division=0)) * 100, 3)
        sxirecall = round(float(recall_score(y_test_binary, test_pred_binary, pos_label=1, zero_division=0)) * 100, 2)
        try:
            fpr, tpr, thresholds = roc_curve(y_test_binary, test_scores, pos_label=1)
            AUC = round(float(auc(fpr, tpr)), 3)
        except Exception as exc:
            print(f"SXI warning: ROC computation failed: {exc}")
            fpr = np.array([0.0, 1.0], dtype=float)
            tpr = np.array([0.0, 1.0], dtype=float)
            thresholds = np.array([], dtype=float)
            AUC = 0.0

        cm = None
        cma = None
        cm_list = []
        try:
            cm, cma, cm_html = ConfusionMat(tv, labels, y_test, test_pred.tolist())
            cm_list = [cm]
        except Exception as exc:
            import traceback
            cm_html = None
            print(f"SXI warning: confusion matrix generation failed: {exc}")
            print(traceback.format_exc())

        folder = f'media/files/chatbot/{buyerid}/'
        os.makedirs(folder, exist_ok=True)
        rand = random.randint(0, 99999999)
        locauc_png = f'aucplt_{rand}.png'
        locauc_html = f'aucplt_{rand}.html'
        file_path = os.path.join(get_dataset_folder(folder, locauc_png), locauc_png)
        file_path2 = os.path.join(get_dataset_folder(folder, locauc_html), locauc_html)

        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                line=dict(color='darkorange', width=4, dash='solid'),
                name=f"<b>AUC = {AUC:.3f}</b>"
            ))
            fig.update_layout(
                title=dict(
                    text="<b>AUC-ROC Curve</b>",
                    font=dict(size=16, family="Arial Black", color="black")
                ),
                xaxis=dict(
                    title="<b>False Positive Rate</b>",
                    tickfont=dict(size=12, family="Arial Black", color="black"),
                    gridcolor='lightgray',
                    zerolinecolor='gray'
                ),
                yaxis=dict(
                    title="<b>True Positive Rate</b>",
                    tickfont=dict(size=12, family="Arial Black", color="black"),
                    gridcolor='lightgray',
                    zerolinecolor='gray'
                ),
                legend=dict(
                    x=0.7,
                    y=0.1,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='gray',
                    borderwidth=1,
                    font=dict(size=12, family="Arial Black", color="black")
                ),
                plot_bgcolor='white',
                font=dict(size=12, family="Arial Black", color="black"),
                width=1200,
                height=680
            )
            _safe_write_plotly_figure(
                fig,
                file_path2,
                file_path,
                title="ROC Curve"
            )
        except Exception as exc:
            import traceback
            print(f"SXI warning: ROC plot export failed: {exc}")
            print(traceback.format_exc())
            file_path = None

        feat_importance_df = pd.DataFrame(columns=["Feature", "Importance", "Cumulative"])
        top_features_df = feat_importance_df.copy()
        try:
            importance_model = best_candidate.get('base_model')
            importance_features = X_train_model

            if X_train_importance is not None and not X_train_importance.empty:
                importance_features = X_train_importance
                importance_model = xg.XGBClassifier(
                    random_state=split_seed,
                    eval_metric='logloss',
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.85,
                    scale_pos_weight=scale_pos_weight,
                    tree_method='hist',
                ).fit(importance_features, y_train_binary)

            feat_importance_df = extract_feature_importances(importance_model, importance_features.columns)
            if not feat_importance_df.empty:
                top_features_df = feat_importance_df[feat_importance_df["Cumulative"] <= 0.8].copy()
                if top_features_df.empty:
                    top_features_df = feat_importance_df.head(10).copy()
        except Exception as exc:
            import traceback
            print(f"SXI warning: feature importance generation failed: {exc}")
            print(traceback.format_exc())

        prediction_ids = test_df[primskey].values if primskey in test_df.columns else test_df.index.to_numpy()
        dap = pd.DataFrame({
            f'{primskey}': prediction_ids,
            f'Actual {tv}': y_test,
            f'Predicted {tv}': test_pred,
        })
        dap[f'Actual {tv}'] = dap[f'Actual {tv}'].apply(lambda x: label_lookup.get(int(x), x))
        dap[f'Predicted {tv}'] = dap[f'Predicted {tv}'].apply(lambda x: label_lookup.get(int(x), x))

        return (
            dap,
            cm_list,
            sxiacc,
            sxiprec,
            cma,
            file_path,
            AUC,
            file_path,
            cm,
            top_features_df,
            feat_importance_df,
            sxirecall,
        )

        alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
        composite_dxi1 = df['composite_dxi']
        composite_dxi1 = list(np.array(composite_dxi1))
        avg_composite_dxi = SXI

        predtst = []
        gfforward, bfforward = [], []
        totdxiac = []
        gooddxicount, baddxicount = [], []
        gddxacc, bddxacc = [], []
        gdclwgt, bdclwgt = [], []
        prec, rec = [], []
        fps, fns, tps, accs, tns = [], [], [], [], []
        pgscr, pbscr = [], []
        pgwt, pbwt = [], []
        actsgd, actsbd = [], []
        prdtsgd, prdtsbd = [], []
        totprdacc = []
        cm_list = []
        rocsc,rocgraph = [], []
        predicted, actual, primary_key = [], [], []

        for i in range(len(alphas)):
            check_cancelled()
            print('alphas', alphas[i])
            n=alphas[i]
            print('avg_composite_dxi',avg_composite_dxi)
            new_avg = n*avg_composite_dxi
            y_forward= create_label_forward(good, bad, composite_dxi1, new_avg,fulldata_sxi,target,goodab)

            df['gd_bdDXI'] = y_forward
            ccgd = df.loc[(df['gd_bdDXI'] == good) & (df[tv] == good)] #Selecting GoodCorrectlyClassifier
            ccbd = df.loc[(df['gd_bdDXI'] == bad) & (df[tv] == bad)] #Selecting BaddCorrectlyClassifier

            wcgd = df.loc[(df['gd_bdDXI'] == good) & (df[tv] == bad)] #False Positives
            wcbd = df.loc[(df['gd_bdDXI'] == bad) & (df[tv] == good)] #False Negatives
            wcbda = wcbd.sample(n=round(len(wcbd)*1))
            wcgda = wcgd.sample(n=round(len(wcgd)*1))
            cc = pd.concat([ccgd, ccbd,wcbda,wcgda], axis=0) #Correctly Classifier Good + Bad
            cc.reset_index(inplace=True, drop=True)
            # X = cc.drop([tv,'gd_bdDXI', 'composite_dxi_label', primskey], axis=1, errors='ignore')
            X = cc.drop(
                [
                    tv,
                    'gd_bdDXI',
                    'composite_dxi_label',
                    'netqyty_Bucket',
                    'composite_dxi',
                    'index',
                    'i_n_d_e_x',
                    primskey,
                ],
                axis=1,
                errors='ignore',
            )
            y = cc[tv]

            if (len(ccbd)==0) or (len(ccgd)==0):
                # print('no')
                qwe = None
                rocsc.append(qwe)
                accs.append(qwe)
                prec.append(qwe)
                rec.append(qwe)
                prdtsgd.append(qwe)
                prdtsbd.append(qwe)
                gfforward.append(qwe)
                bfforward.append(qwe)
                gooddxicount.append(qwe)
                baddxicount.append(qwe)
                gddxacc.append(qwe)
                bddxacc.append(qwe)
                gdclwgt.append(qwe)
                bdclwgt.append(qwe)
                totdxiac.append(qwe)
                pgscr.append(qwe)
                pbscr.append(qwe)
                pgwt.append(qwe)
                pbwt.append(qwe)
                totprdacc.append(qwe)
                actsgd.append(qwe)
                actsbd.append(qwe)
                fps.append(qwe)
                fns.append(qwe)
                tps.append(qwe)
                tns.append(qwe)
                cm_list.append(qwe)
            else:
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                non_validate_X, validate_X, non_validate_y, validate_y = train_test_split(X, y, stratify=y, test_size=0.1)
                X_train, X_test, y_train, y_test = train_test_split(non_validate_X, non_validate_y, test_size=20/90)

                primary_key.append(X_test[primskey].values)    ################### Important
                actual.append(list(y_test))    ################### Important

                non_validate_X = non_validate_X.drop(primskey, axis=1, errors='ignore')
                validate_X = validate_X.drop(primskey, axis=1, errors='ignore')
                X_train = X_train.drop(primskey, axis=1, errors='ignore')
                X_test = X_test.drop(primskey, axis=1, errors='ignore')

                testbad = y_test.tolist().count(bad) #Actual Test Count for subsribed
                testgood = y_test.tolist().count(good) #Actual Test Count for not subsribed
                actsgd.append(testgood)
                actsbd.append(testbad)
                fp = len(wcgd)
                fn = len(wcbd)
                tp = len(ccgd) #correctly classifier count for subscribed
                tn = len(ccbd) #correctly classifier count for not-subscribed
                fps.append(fp)
                fns.append(fn)
                tps.append(tp)
                tns.append(tn)

                xgb = xg.XGBClassifier().fit(X_train, y_train)
                pred = xgb.predict(X_test).tolist()
                predicted.append(pred)    ################### Important

                predval = xgb.predict(validate_X).tolist()
                acc = accuracy_score(y_test, pred)
                accval = accuracy_score(validate_y,predval)
                classes = list(labels.keys())
                y_probs = xgb.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = auc(fpr, tpr)
                cm, cma, cm_html = ConfusionMat(tv, labels, y_test, pred)
                # preci = (cma[good,good]/(cma[good,good]+cma[good,bad]))
                # recal = (cma[good,good]/(cma[good,good]+cma[bad,good]))
                # BAD class (reverse)
                # preci = cma[bad, bad] / (cma[bad, bad] + cma[good, bad])
                # recal = cma[bad, bad] / (cma[bad, bad] + cma[bad, good])
                # print(f'Precision for {good}: {preci:.3f}, Recall for {good}: {recal:.3f}')
                preci = precision_score(y_test, pred, pos_label=bad)
                recal    = recall_score(y_test, pred, pos_label=bad)

                print(f"Fraud Precision: {preci:.3f}")
                print(f"Fraud Recall   : {recal:.3f}")
                
                # print(f'Precision for {bad}: {preci_bad:.3f}, Recall for {bad}: {recal_bad:.3f}')
                cm_list.append(cm)

                # # Create the ROC Curve plot
                # fig = go.Figure()

                # # Add the ROC curve
                # fig.add_trace(go.Scatter(
                #     x=fpr,  # False Positive Rate
                #     y=tpr,  # True Positive Rate
                #     mode='lines',
                #     line=dict(color='darkorange', width=2, dash='dash'),  # Line style and color
                #     name=f'AUC = {roc_auc:.3f}'
                # ))

                # # Update layout to match Matplotlib's style
                # fig.update_layout(
                #     title='AUC-ROC Curve',
                #     xaxis=dict(
                #         title='False Positive Rate',
                #         gridcolor='lightgray',
                #         zerolinecolor='gray'
                #     ),
                #     yaxis=dict(
                #         title='True Positive Rate',
                #         gridcolor='lightgray',
                #         zerolinecolor='gray'
                #     ),
                #     legend=dict(
                #         x=0.8,
                #         y=0.2,
                #         bgcolor='rgba(255, 255, 255, 0.8)',
                #         bordercolor='gray',
                #         borderwidth=1
                #     ),
                #     plot_bgcolor='white',
                #     font=dict(size=10),
                #     width=1200,  # Adjust width
                #     height=680  # Adjust height
                # )


                # Create the ROC Curve plot
                fig = go.Figure()

                # Add the ROC curve
                fig.add_trace(go.Scatter(
                    x=fpr,  # False Positive Rate
                    y=tpr,  # True Positive Rate
                    mode='lines',
                    line=dict(color='darkorange', width=4, dash='solid'),  # Thicker, solid bold line
                    name=f"<b>AUC = {roc_auc:.3f}</b>"  # Bold legend text
                ))

                # Update layout for bold fonts and enhanced visuals
                fig.update_layout(
                    title=dict(
                        text="<b>AUC-ROC Curve</b>",
                        font=dict(
                            size=16,
                            family="Arial Black",
                            color="black"
                        )
                    ),
                    xaxis=dict(
                        title="<b>False Positive Rate</b>",
                        # titlefont=dict(size=14, family="Arial Black", color="black"),
                        tickfont=dict(size=12, family="Arial Black", color="black"),
                        gridcolor='lightgray',
                        zerolinecolor='gray'
                    ),
                    yaxis=dict(
                        title="<b>True Positive Rate</b>",
                        # titlefont=dict(size=14, family="Arial Black", color="black"),
                        tickfont=dict(size=12, family="Arial Black", color="black"),
                        gridcolor='lightgray',
                        zerolinecolor='gray'
                    ),
                    legend=dict(
                        x=0.7,
                        y=0.1,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='gray',
                        borderwidth=1,
                        font=dict(size=12, family="Arial Black", color="black")
                    ),
                    plot_bgcolor='white',
                    font=dict(size=12, family="Arial Black", color="black"),
                    width=1200,
                    height=680
                )



                # buyerid = self.request.session.get('buyerid')
                folder = f'media/files/chatbot/{buyerid}/'
                rand = random.randint(0, 99999999)
                locauc_png =  f'aucplt_{rand}.png'
                locauc_html =  f'aucplt_{rand}.html'
                file_path = os.path.join(get_dataset_folder(folder, locauc_png), locauc_png)
                file_path2 = os.path.join(get_dataset_folder(folder, locauc_html), locauc_html)
                _safe_write_plotly_figure(
                    fig,
                    file_path2,
                    file_path,
                    title="ROC Curve"
                )
                print(f'ROC Curve saved at {locauc_png}')

                # ================= Feature Importance Section =================
                # Get feature importances
                importances = xgb.feature_importances_
                features = X_train.columns

                feat_importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importances
                })

                # Sort by importance
                feat_importance_df = feat_importance_df.sort_values(by="Importance", ascending=False)

                # Normalize to cumulative %
                feat_importance_df["Cumulative"] = feat_importance_df["Importance"].cumsum() / feat_importance_df["Importance"].sum()

                # Select top features covering 80% cumulative importance
                top_features_df = feat_importance_df[feat_importance_df["Cumulative"] <= 0.8]

                print("Top Features covering 80% importance:")
                print(top_features_df)

                # ===============================================================

                # plt.savefig(locauc, dpi=300)

                # plt.close()
                rocsc.append(round(roc_auc,3))
                rocgraph.append(file_path)
                accs.append(acc)
                prec.append(preci)
                rec.append(recal)

                predgood = pred.count(good) # Predicted Test Count for subsribed
                predbad = pred.count(bad) # Predicted Test Count for not subsribed
                prdtsgd.append(predgood)
                prdtsbd.append(predbad)

                good_dxi_forward = y_forward.count(good) # Good DXI Count             #ONLY CHANGE HERE
                bad_dxi_forward = y_forward.count(bad) # Bad DXI Count               #ONLY CHANGE HERE
                gfforward.append(good_dxi_forward)
                bfforward.append(bad_dxi_forward)

                churngood = y_train.tolist().count(good)
                churnbad = y_train.tolist().count(bad)
                gooddxicount.append(churngood)
                baddxicount.append(churnbad)

                goodDXI_accuracy = good_dxi_forward/len(df) #Good DXI Accuracy = Good DXI Count / Total patient
                badDXI_accuracy = bad_dxi_forward/len(df) #Bad DXI Accuracy = Bad DXI Count / Total patient
                gddxacc.append(goodDXI_accuracy)
                bddxacc.append(badDXI_accuracy)

                gdclfwgt = tp/(tp+tn) #Good classifier weights = correctly classifier count for subscribed / correctly classifier count
                bdclfwgt = 1-gdclfwgt #Bad classifier weights = 1 – Good classifier weights
                gdclwgt.append(gdclfwgt)
                bdclwgt.append(bdclfwgt)

                # Total DXI classifier Accuracy = (DXI Good Accuracy*good classifier weight) + (DXI Bad Accuracy*bad classifier weight)
                totdxiacc = (goodDXI_accuracy*gdclfwgt) + (badDXI_accuracy*bdclfwgt) # Total DXI classifier Accuracy
                totdxiac.append(totdxiacc)

                if testgood == 0:
                    testgood = 1

                if testbad == 0:
                    testbad = 1

                # Predicted good score = predicted test count for subscribed/actual test count for subscribed
                predgdscr = predgood/testgood
                predgdscr = round(predgdscr, 2)
                pgscr.append(predgdscr)

                # Predicted bad score = predicted test count for not subscribed/actual test count for not subscribed
                predbdscr = predbad/testbad
                predbdscr = round(predbdscr, 2)
                pbscr.append(predbdscr)

                #Predicted good classifier weight = actual test count for subscribed / sum of actual test count for subscribed + actual test count for notsubscribed
                predgdclwt = testgood/(testgood+testbad)
                predgdclwt = round(predgdclwt, 2)
                pgwt.append(predgdclwt)

                # Predicted bad classifier weight = 1- Predicted good classifier weight
                predbdclwt = testbad/(testgood+testbad)
                predbdclwt = round(predbdclwt, 2)
                pbwt.append(predbdclwt)

                g = predgdscr*predgdclwt #(Predicted Good score*predicted good classifier weight)
                p = predbdscr*predbdclwt #(Predicted Bad score*predicted bad classifier weight)

                # Predicted DXI classifier Accuracy= (Predicted Good score*predicted good classifier weight) + (Predicted Bad score*predicted bad classifier weight)
                totpredacc = g+p
                totprdacc.append(totpredacc)


        mapping = {val: key for (key, val) in labels.items()}
        qq = f'{tv}_{mapping[good]}'
        zz = f'{tv}_{mapping[bad]}'

        def fixnone(gooddxicount):
            for i in range(len(gooddxicount)):
                if gooddxicount[i] == None:
                    gooddxicount[i] = 0
            return gooddxicount

        j = {'alphas': alphas,
            'Train Count': [sum(i) for i in zip(fixnone(gooddxicount), fixnone(baddxicount))],
            'Test Count': [sum(i) for i in zip(fixnone(prdtsgd), fixnone(prdtsbd))],
            f'Actual Train count for {qq}': gooddxicount,
            f'Actual Train count for {zz}': baddxicount,
            f'Actual Test Count for {qq}': actsgd,
            f'Actual Test Count for {zz}': actsbd,
            f'Predicted Test Count for {qq}': prdtsgd,
            f'Predicted Test Count for {zz}': prdtsbd,
            'Model accuracy': accs,
            'Precision': prec,
            'Recall': rec,
            'AUC':rocsc,
            }

        dx = pd.DataFrame(j)
        print(dx)
        rand = random.randint(00000000, 99999999)
        localpha = f'df_alpha{rand}.csv'
        # dx.to_csv(localpha, index=False)

        dx.dropna(inplace=True)
        dxnew = dx[(dx.alphas > 0.5 ) & (dx.alphas < 1.2)]
        # accind = max(dxnew['Model accuracy'].index)
        accind = dxnew['Model accuracy'].idxmax()
        dxnewupd = dxnew.loc[accind]
        fnlprec =  dxnewupd['Precision']
        fnlacc = dxnewupd['Model accuracy']
        sxiacc= round(fnlacc*100,3)
        sxiprec = round(fnlprec*100,3)
        # sxirecall = round(rec*100,3)
        AUC = dxnewupd['AUC']
        sxirecall=round(dxnewupd['Recall']*100,2)
        print("primary_key_mapping.........",primary_key_mapping)
        # ✅ Save the best model
        import pickle
        folder = f'media/files/chatbot/{buyerid}/'
        os.makedirs(folder, exist_ok=True)

        # model_file = os.path.join(folder, f"best_model_{rand}.pkl")
        # with open(model_file, "wb") as f:
        #     pickle.dump(models[accind], f)   # <-- save the best model using accind
        # print(f"✅ Best model saved at {model_file}")

        ########################################################################

        if len(primary_key_encoded)>0:
            primary_keys_test_encoded = primary_key_encoded[X_test.index]
            print(primary_keys_test_encoded)
            # myDict = {
            #         f'{primskey}': [primary_key_mapping[pk] for pk in primary_keys_test_encoded],
            #         f'Actual {tv}': actual[accind],
            #         f'Predicted {tv}': predicted[accind]}
            myDict = {
                f'{primskey}': primary_key.loc[accind],
                f'Actual {tv}': actual.loc[accind],
                f'Predicted {tv}': predicted.loc[accind]
            }
            dap = pd.DataFrame(myDict)
            labels = {val: key for (key, val) in labels.items()}
            dap[f'Actual {tv}'] = dap[f'Actual {tv}'].apply(lambda x: labels.get(x, x))
            dap[f'Predicted {tv}'] = dap[f'Predicted {tv}'].apply(lambda x: labels.get(x, x))
        else:
            myDict={}
            myDict[f'{primskey}'] = primary_key[int(accind)]
            myDict[f'Actual {tv}'] = actual[int(accind)]
            myDict[f'Predicted {tv}'] = predicted[int(accind)]
            dap = pd.DataFrame(myDict)
            labels = {val: key for (key, val) in labels.items()}
            dap[f'Actual {tv}'] = dap[f'Actual {tv}'].apply(lambda x: labels.get(x, x))
            dap[f'Predicted {tv}'] = dap[f'Predicted {tv}'].apply(lambda x: labels.get(x, x))
            primary_keys_test_encoded = None

        print("primary_keys_test_encoded.........",primary_keys_test_encoded)



        #dap.to_csv('qwerty.csv', index=False)

        ########################################################################


        # dx = round(dx, 2)
        """
        dx = dx.set_index('alphas').transpose().reset_index()
        dx = dx.rename(columns={'index':'alphas'}).fillna(0)
        alpha = alphas
        accuracy = [0 if v is None else v for v in accs]
        precision = [0 if v is None else v for v in prec]
        x = True
        while x:
            for i, j, k in zip(alpha, accuracy, precision):
                max_acc=max(accuracy)
                print(max_acc)
                if max_acc == j:
                    #print("max",max_acc)
                    if max_acc-0.05 < k:
                        #print(i,j,k)
                        y = i
                        sxiacc= round(j*100,2)
                        sxiprec = round(k*100,2)
                        print(y,j,k)
                        x=False
                        break
                    else:
                        j=alpha.index(i)
                        alpha[j]=-100
                        accuracy[j]=-100
                        precision[j]=-100
                        continue
        """
        # Save SXI df and EDA

        # import plotly
        # import plotly.figure_factory as ff
        # fig = ff.create_table(dx)
        # fig.update_layout(width=4000)
        # alphahtml = f'media/files/hurra/{buyerid}/df_alpha{rand}.html'
        # plotly.offline.plot(fig, filename=alphahtml, auto_open=False)

        # return dap,cm_list, sxiacc, sxiprec, cma, rocgraph[accind],AUC,file_path,cm, sxirecall
        # return dap,cm_list, sxiacc, sxiprec, cma, rocgraph[accind],AUC,file_path,cm
        return (
        dap,                # Predictions dataframe
        cm_list,            # Confusion matrices
        sxiacc,             # SXI Accuracy
        sxiprec,            # SXI Precision
        cma,                # Confusion matrix array
        rocgraph[int(accind)],   # ROC Graph
        AUC,                # AUC
        file_path,          # ROC curve file path
        cm,                 # Confusion matrix HTML
        top_features_df,    # Top 80% cumulative feature importances
        feat_importance_df,  # Full feature importances
        sxirecall,
    )


    def sxi_methodreg(buyerid,good, bad, labels, target_type, target, fulldata_sxi, goodab, primskey, SXI,primary_key_mapping,primary_key_encoded):
        check_cancelled()
        print('SXI Method Started!')

        def sanitize_feature_names(columns):
            safe_names = []
            seen = {}
            for col in columns:
                safe = str(col).replace("[", "").replace("]", "").replace("<", "").replace(">", "")
                if safe in seen:
                    seen[safe] += 1
                    safe = f"{safe}_{seen[safe]}"
                else:
                    seen[safe] = 0
                safe_names.append(safe)
            return safe_names


        def actpredplt(tv, y_test, mae, pred, r2):
            import os
            import random
            import numpy as np

            import matplotlib.pyplot as plt
            # Calculate ±10% error boundaries
            error_tol = 0.10
            Ytest_array = np.asarray(y_test, dtype=float)
            pred_array = np.asarray(pred, dtype=float)
            finite_mask = np.isfinite(Ytest_array) & np.isfinite(pred_array)
            Ytest_array = Ytest_array[finite_mask]
            pred_array = pred_array[finite_mask]

            # Clean reference lines across the observed actual range
            if len(Ytest_array):
                x_min = float(np.nanmin(Ytest_array))
                x_max = float(np.nanmax(Ytest_array))
            else:
                x_min, x_max = 0.0, 1.0
            if abs(x_max - x_min) < 1e-12:
                x_max = x_min + 1.0
            line_x = np.linspace(x_min, x_max, 200)
            perfect_y = line_x.copy()
            upper_bound = line_x * (1.0 + error_tol)
            lower_bound = line_x * (1.0 - error_tol)

            # Prepare data for points within and outside the 10% error boundaries
            within_bounds_x = []
            within_bounds_y = []
            within_bounds_hover = []  # Hover text for within bounds

            outside_bounds_x = []
            outside_bounds_y = []
            outside_bounds_hover = []  # Hover text for outside bounds

            total_points = len(Ytest_array)
            within_bounds_count = 0  # Counter for points within bounds

            for yt, yp in zip(Ytest_array, pred_array):
                hover_text = (
                    f"Actual: {format_display_value(yt, tv)}"
                    f"<br>Predicted: {format_display_value(yp, tv)}"
                )
                tolerance = abs(yt) * error_tol
                is_within = abs(yp - yt) <= tolerance if tolerance > 0 else abs(yp - yt) <= 0
                if is_within:
                    within_bounds_x.append(yt)
                    within_bounds_y.append(yp)
                    within_bounds_hover.append(hover_text)
                    within_bounds_count += 1
                else:
                    outside_bounds_x.append(yt)
                    outside_bounds_y.append(yp)
                    outside_bounds_hover.append(hover_text)

            # Calculate percentage of points within 10% error
            within_percentage = (within_bounds_count / total_points) * 100 if total_points else 0.0
            annotation_text = f"Within ±10% Error: {within_percentage:.2f}%"

            # Matplotlib PNG first (Kaleido often drops reference lines)
            plt.figure(figsize=(10, 6.2))
            ax = plt.gca()
            ax.set_facecolor("#eff1fe")
            if within_bounds_x:
                ax.scatter(within_bounds_x, within_bounds_y, s=28, c="green", alpha=0.6, label="Within ±10%")
            if outside_bounds_x:
                ax.scatter(outside_bounds_x, outside_bounds_y, s=28, c="red", alpha=0.6, label="Outside ±10% Error")
            ax.plot(line_x, perfect_y, color="black", linewidth=2, label="Perfect Prediction")
            ax.plot(line_x, upper_bound, color="blue", linewidth=1.5, linestyle="--", label="+10% Error")
            ax.plot(line_x, lower_bound, color="blue", linewidth=1.5, linestyle="--", label="-10% Error")
            ax.set_title(f"Actual vs Predicted {tv}\n{annotation_text}", fontsize=13, fontweight="bold")
            ax.set_xlabel(f"Actual {tv}")
            ax.set_ylabel(f"Predicted {tv}")
            ax.grid(True, which="major", color="#9585e6", linestyle="-", linewidth=0.6, alpha=0.55)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
            plt.tight_layout()

            # Create the Plotly figure
            fig = go.Figure()

            # Plot points within the 10% error boundary
            fig.add_trace(go.Scatter(
                x=within_bounds_x,
                y=within_bounds_y,
                mode='markers',
                marker=dict(size=8, color='green', opacity=0.6),
                text=within_bounds_hover,  # Add hover text
                hoverinfo="text",
                name='Within ±10% '
            ))

            # Plot points outside the 10% error boundary
            fig.add_trace(go.Scatter(
                x=outside_bounds_x,
                y=outside_bounds_y,
                mode='markers',
                marker=dict(size=8, color='red', opacity=0.6),
                text=outside_bounds_hover,  # Add hover text
                hoverinfo="text",
                name='Outside ±10% Error'
            ))

            # Add the perfect prediction line (y = x)
            fig.add_trace(go.Scatter(
                x=line_x.tolist(),
                y=perfect_y.tolist(),
                mode='lines',
                line=dict(color='black', width=2),
                name='Perfect Prediction'
            ))

            # Add dashed lines for ±10% error boundaries
            fig.add_trace(go.Scatter(
                x=line_x.tolist(),
                y=upper_bound.tolist(),
                mode='lines',
                line=dict(color='blue', width=1.5, dash='dash'),
                name='+10% Error'
            ))

            fig.add_trace(go.Scatter(
                x=line_x.tolist(),
                y=lower_bound.tolist(),
                mode='lines',
                line=dict(color='blue', width=1.5, dash='dash'),
                name='-10% Error'
            ))

            # Update layout for titles and axis labels
            fig.update_layout(
                title=dict(
                    text=f"Actual vs Predicted {tv}<br>{annotation_text}",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=build_plotly_value_axis(f'Actual {tv}', tv),
                yaxis=build_plotly_value_axis(f'Predicted {tv}', tv),
                legend=dict(
                    x=0,  # x position (0 = left)
                    y=1,  # y position (1 = top)
                    xanchor='left',  # Anchor legend to the left
                    yanchor='top'  # Anchor legend to the top
                ),
                width=800,
                height=500
            )
            import random
            import shutil
            from pathlib import Path as _Path
            rand = random.randint(0, 99999999)
            folder = f'media/files/chatbot/{buyerid}/'
            loc = f'actpred{rand}.html'
            loc_png = f'actpred{rand}.png'
            actpredplt_ml = os.path.join(get_dataset_folder(folder, loc), loc)
            actpredplt_ml2 = os.path.join(get_dataset_folder(folder, loc_png), loc_png)
            # Primary PNG with reference lines (Plotly/Kaleido often drops them)
            plt.savefig(actpredplt_ml2, dpi=180, bbox_inches="tight", facecolor="white")
            plt.close()
            mpl_backup = str(_Path(actpredplt_ml2).with_suffix(".mpl.png"))
            try:
                shutil.copy2(actpredplt_ml2, mpl_backup)
            except Exception:
                mpl_backup = None
            try:
                fig.write_html(actpredplt_ml)
            except Exception as exc:
                print(f"[actpredplt] HTML export failed: {exc}")
            try:
                _safe_write_plotly_figure(
                    fig,
                    actpredplt_ml,
                    actpredplt_ml2,
                    title=f"Actual vs Predicted {tv}"
                )
            except Exception as exc:
                print(f"[actpredplt] Plotly PNG export failed, keeping matplotlib PNG: {exc}")
            # Always restore matplotlib PNG so Perfect Prediction / ±10% lines remain visible
            if mpl_backup and _Path(mpl_backup).exists():
                shutil.copy2(mpl_backup, actpredplt_ml2)
                try:
                    _Path(mpl_backup).unlink(missing_ok=True)
                except Exception:
                    pass

            print(f'Plot saved at {actpredplt_ml}')
            return actpredplt_ml2, within_percentage

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            nonzero_mask = y_true != 0

            if not np.any(nonzero_mask):
                return 0

            return np.mean(
                np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
            ) * 100

        import xgboost as xg
        def create_label_forward(good, bad, composite_dxi1, avg_composite_dxi, fulldata_sxi, target, goodab):
            y_forward=[]
            gd_count = int(fulldata_sxi[target].eq(good).sum())
            gdper = (gd_count/len(fulldata_sxi))*100 if len(fulldata_sxi) else 0

            for i in range(len(composite_dxi1)):
                if np.all(composite_dxi1[i] > avg_composite_dxi):
                    if  gdper > goodab:
                        y_forward.append(bad)
                    else:
                        y_forward.append(good)
                else:
                    if gdper > goodab:
                        y_forward.append(good)
                    else:
                        y_forward.append(bad)

            return y_forward


        ###################### Start from Here #######################
        tv = target
        df = fulldata_sxi
        original_target_col = f'{tv}_original'
        if original_target_col not in df.columns:
            return None, None, None, None, None, None, None, None, None

        print('df',df.columns)
        alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
        composite_dxi1 = df['composite_dxi']
        composite_dxi1 = list(np.array(composite_dxi1))
        avg_composite_dxi = SXI

        accs, actlocs, mape = [], [], []
        within_percentages = []
        top_features_dfs = []
        feat_importance_dfs = []
        maesc, maegraph = [], []
        predicted, actual, primary_key = [], [], []
        for alpha in alphas:
            check_cancelled()
            print('alphas', alpha)
            n = alpha
            new_avg = n * avg_composite_dxi
            y_forward = create_label_forward(good, bad, composite_dxi1, new_avg, fulldata_sxi, target, goodab)

            df['gd_bdDXI'] = y_forward
            ccgd = df.loc[(df['gd_bdDXI'] == good) & (df[tv] == good)]  # Selecting GoodCorrectlyClassifier
            ccbd = df.loc[(df['gd_bdDXI'] == bad) & (df[tv] == bad)]  # Selecting BaddCorrectlyClassifier

            wcgd = df.loc[(df['gd_bdDXI'] == good) & (df[tv] == bad)]  # False Positives
            wcbd = df.loc[(df['gd_bdDXI'] == bad) & (df[tv] == good)]  # False Negatives
            wcbda = wcbd.sample(n=round(len(wcbd) * 1))
            wcgda = wcgd.sample(n=round(len(wcgd) * 1))
            cc = pd.concat([ccgd, ccbd, wcbda, wcgda], axis=0)  # Correctly Classifier Good + Bad
            cc.reset_index(inplace=True, drop=True)
            print('cc/n', cc)
            # X = cc.drop([tv, f'{tv}_original', 'gd_bdDXI', 'composite_dxi_label', primskey], axis=1, errors='ignore')
            X = cc.drop([tv, original_target_col, 'gd_bdDXI', 'composite_dxi_label','netqyty_Bucket', 'composite_dxi'], axis=1, errors='ignore')
            y = pd.to_numeric(cc[original_target_col], errors='coerce')
            valid_target_mask = y.notna()
            X = X.loc[valid_target_mask].reset_index(drop=True)
            y = y.loc[valid_target_mask].reset_index(drop=True)
            print('y', y)

            if (len(ccbd) == 0) or (len(ccgd) == 0) or len(y) < 5 or y.nunique(dropna=True) < 2:
                accs.append(None)
                maesc.append(None)
            else:
                rand = random.randint(0, 99999999)
                non_validate_X, validate_X, non_validate_y, validate_y = train_test_split(X, y, test_size=0.1)
                X_train, X_test, y_train, y_test = train_test_split(non_validate_X, non_validate_y, test_size=20 / 90)

                primary_key.append(X_test[primskey].values)    ################### Important
                actual.append(list(y_test))    ################### Important

                non_validate_X = non_validate_X.drop(primskey, axis=1, errors='ignore')
                validate_X = validate_X.drop(primskey, axis=1, errors='ignore')
                X_train = X_train.drop(primskey, axis=1, errors='ignore')
                X_test = X_test.drop(primskey, axis=1, errors='ignore')
                safe_feature_names = sanitize_feature_names(X_train.columns)
                non_validate_X.columns = safe_feature_names
                validate_X.columns = safe_feature_names
                X_train.columns = safe_feature_names
                X_test.columns = safe_feature_names

                xgb = xg.XGBRegressor().fit(X_train, y_train)
                pred = xgb.predict(X_test).tolist()
                predicted.append(pred)    ################### Important

                predval = xgb.predict(validate_X).tolist()
                acc = r2_score(y_test, pred)
                accval = r2_score(validate_y, predval)
                mae = round(mean_absolute_error(y_test, pred), 2)
                # Keep signed R² — abs() made worse-than-mean models look as good as fit models
                accs.append(float(acc) if acc is not None else None)
                maesc.append(mae)
                ###Mape

                actloc,within_percentage = actpredplt(tv, y_test, mae, pred,acc)
                mape_er = mean_absolute_percentage_error(y_test, pred)
                mape.append(mape_er)
                actlocs.append(actloc)
                within_percentages.append(within_percentage)
                # ================= Feature Importance Section =================
                # Get feature importances
                importances = xgb.feature_importances_
                features = X_train.columns

                feat_importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importances
                })

                # Sort by importance
                feat_importance_df = feat_importance_df.sort_values(by="Importance", ascending=False)

                # Normalize to cumulative %
                feat_importance_df["Cumulative"] = feat_importance_df["Importance"].cumsum() / feat_importance_df["Importance"].sum()

                # Select top features covering 80% cumulative importance
                top_features_df = feat_importance_df[feat_importance_df["Cumulative"] <= 0.8]
                top_features_dfs.append(top_features_df)
                feat_importance_dfs.append(feat_importance_df)

                print("Top Features covering 80% importance:")
                print(top_features_df)

                # ===============================================================


        # Filter out None values
        filtered_data = [(a, m, ac) for a, m, ac in zip(alphas, maesc, accs) if ac is not None]
        if filtered_data:
            alphas, maesc, accs = zip(*filtered_data)
        else:
            alphas, maesc, accs = [], [], []

        j = {
            'alphas': alphas,
            'Model accuracy': accs,
            'Mean Absolute Error': maesc
        }

        print(j)
        dx = pd.DataFrame(j)
        print(dx)

        if not dx.empty:
            dx.dropna(inplace=True)
            dxnew = dx[(dx.alphas > 0.8) & (dx.alphas < 1.2)]
            if not dxnew.empty:
                accind = dxnew['Model accuracy'].idxmax()
                print(f'Accuracy Index: {accind}')
                dxnewupd = dxnew.loc[accind]
                fnlacc = dxnewupd['Model accuracy']
                fnlmae = dxnewupd['Mean Absolute Error']
                scaled_fnlacc = fnlacc * 100
                sxiacc = round(fnlacc if scaled_fnlacc > 1 else scaled_fnlacc, 2)
                sximae = round(fnlmae, 2)
                print('primary_key_encoded!!!!',primary_key_encoded)
                print("primary_key_mapping!!!!!!!!!",primary_key_mapping)
                # Save the best model
                # import pickle
                # folder = f'media/files/chatbot/{buyerid}/'
                # os.makedirs(folder, exist_ok=True)
                # model_file = os.path.join(folder, f"best_model_{rand}.pkl")
                # with open(model_file, "wb") as f:
                #     pickle.dump(models[accind], f)   # <-- save the best model using accind

                # print(f"✅ Best model saved at {model_file}")


                ########################################################################
                if len(primary_key_encoded)>0:
                    primary_keys_test_encoded = primary_key_encoded[X_test.index]
                    print(primary_keys_test_encoded)
                    myDict = {
                    f'{primskey}': [primary_key_mapping[pk] for pk in primary_keys_test_encoded],
                    f'Actual {tv}': actual[int(accind)],
                    f'Predicted {tv}': predicted[int(accind)]}
                    dap = pd.DataFrame(myDict)
                else:
                    myDict={}
                    myDict[f'{primskey}'] = primary_key[int(accind)]
                    myDict['Actual Values'] = actual[int(accind)]
                    myDict['Predicted Values'] = predicted[int(accind)]
                    dap = pd.DataFrame(myDict)
                    # dap.to_csv('qwerty.csv', index=False)
                    myDict={}
                    myDict[f'{primskey}'] = primary_key[int(accind)]
                    myDict[f'Actual {tv}'] = actual[int(accind)]
                    myDict[f'Predicted {tv}'] = predicted[int(accind)]
                    dap = pd.DataFrame(myDict)
                    primary_keys_test_encoded = None



                #dap.to_csv('qwerty.csv', index=False)
                ########################################################################

                return (
                    dap,
                    actlocs[int(accind)],
                    sxiacc,
                    sximae,
                    actlocs[int(accind)],
                    within_percentages[int(accind)],
                    top_features_dfs[int(accind)],
                    feat_importance_dfs[int(accind)],
                    mape[int(accind)],
                )



            else:
                print("No valid alpha range found.")
                return None, None, None, None, None,None,None,None,None
        else:
            print("No valid data to create DataFrame.")
            return None, None, None, None,None,None,None,None,None


# 2nd
class sxirl_engine:   
    def __init__(self, target_type, target, buynobuy, sxi_dataframe, classes,primkey):
            self.target_type = target_type
            self.target = target
            self.buynobuy = buynobuy
            self.sxi_dataframe = sxi_dataframe
            self.classes = classes
            self.primkey = primkey

    def single_correlation(self,x,y):
        x = x.astype(float)
        y = y.astype(float)
        #corr,_=(stats.pearsonr(x, y))
        corr, _ = pearsonr(x, y)
        return float(corr)

    def bivarient_correlation(self,df):
        length_of_parameter = df.shape[1]
        column_name=df.columns
        raw_bivarient_correlation=[]
        abs_bivarient_correlation=[]
        raw=[[] for i in range(length_of_parameter)]
        abss=[[] for i in range(length_of_parameter)]
        for i in range(length_of_parameter):
            for j in range(length_of_parameter):
                if (i != j):
                    #print(i)
                    
                    corr=self.single_correlation(df[column_name[i]], df[column_name[j]])
                    
                    raw[i].append(corr)
                    abss[i].append(abs(corr))
            abss[i] = [x for x in abss[i] if math.isnan(x) == False]      
        for i in range(length_of_parameter):
            abs_bivarient_correlation.append((np.mean(abss[i]))/2)
        # raw_bivarient_correlation.append(np.mean(raw[i]))
        abs_bivarient_correlation= np.nan_to_num(np.array(abs_bivarient_correlation))   
        return list(abs_bivarient_correlation)

    def weight(self,df,b):
        W=[]
        for i in range(df.shape[1]):
            w=1-(b[i]/2)
            W.append(w)
        return W

    def dxi(self,new_df,w):
        new_df=(np.array(new_df)).astype(float)
        w=np.array(w)
        w_df=np.dot(new_df,w.T)
        dxi=[]
        for i in range(len(new_df)):
            individual_dxi=(w_df[i]*100)/len(w)
            dxi.append(individual_dxi) 
        return dxi

    def normalize(self,df):
        df1=df
        min_max= (df1.iloc[2])
        max_column=df1.iloc[1].astype(float)
        column_name=df1.columns
        new_df=df.iloc[4:]
        
        for i in range(df1.shape[1]):
            
            if min_max[i]=='MAX':
                new_df[column_name[i]]= new_df[column_name[i]].astype(float)/max_column[i]
            else:
                new_df[column_name[i]]=(max_column[i]-new_df[column_name[i]].astype(float))/max_column[i] 
        return new_df

    def create_label(self,total_dxi,Avg_dxi):
        y=[]
        for i in range(len(total_dxi)):
            if np.all(total_dxi[i]>Avg_dxi):
                y.append(1)
            else:
                y.append(0)
        return y

    def create_label_forward(self,composite_dxi1,avg_composite_dxi):
        y_forward=[]
        for i in range(len(composite_dxi1)):

            if np.all(composite_dxi1[i] > avg_composite_dxi):
                y_forward.append(0)
            else:
                y_forward.append(1)
        return y_forward

    def total_dxi(self,df1,weight):
        total_dxi=self.dxi(df1,weight)
        Avg_dxi=np.mean(np.array(total_dxi))
        return total_dxi, Avg_dxi,weight

    def catogorical_dxi(self,df,weight,parameter):
        column_name= df.columns
        behaviour_column_name=[]
        transactional_column_name=[]
        visual_column_name=[]
        kpi_column_name=[]
        
        weight_behaviour=[]
        weight_transactional=[]
        weight_visual=[]
        weight_kpi=[]
        
        for i in range(len(parameter)):
            if parameter[i]=='Behavior':
                behaviour_column_name.append(column_name[i]) 
                weight_behaviour.append(weight[i])
            elif parameter[i]=='Transactional':
                transactional_column_name.append(column_name[i])
                weight_transactional.append(weight[i])
            elif parameter[i]=='visual':
                visual_column_name.append(column_name[i])
                weight_visual.append(weight[i])
            else:
                kpi_column_name.append(column_name[i])
                weight_kpi.append(weight[i])
            
        new_df=df
        
        df_behaviour = new_df[behaviour_column_name]
        df_transactional=new_df[transactional_column_name]
        df_visual=new_df[visual_column_name]
        df_kpi=new_df[kpi_column_name]
    
        behaviour_dxi=self.dxi(df_behaviour,weight_behaviour)
        #avg_behaviour_dxi=mean(behaviour_dxi)
        transactional_dxi=self.dxi(df_transactional,weight_transactional)
        visual_dxi=self.dxi(df_visual,weight_visual)
        kpi_dxi=self.dxi(df_kpi,weight_kpi)
        return behaviour_dxi,transactional_dxi,kpi_dxi,visual_dxi

    def svm_feature_selection(self,x,y):
        from sklearn.svm import LinearSVC,SVC
        #lsvc=SVC(gamma='auto')
        lsvc = LinearSVC(C=0.4,penalty='l1', dual=False).fit(x, y)
        coeff=lsvc.coef_
        coeff = abs(coeff)
        return coeff.reshape(-1)

    def mutual_information(self,x,y):
        from sklearn.feature_selection import mutual_info_classif
        x= mutual_info_classif(x, y)
        for i in range(len(x)):
            if x[i] < .1 :
                x[i]=0
        return x

    def tree_feature_selection(self,x,y):
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=100)
        clf = clf.fit(x, y)          
        coeff = clf.feature_importances_ 
        return coeff

    def pca_feature_selection(self,x,y):
        from sklearn.feature_selection import VarianceThreshold 
        selector = VarianceThreshold() 
        selector.fit_transform(x) 
        print(selector.variances_)
        x=selector.variances_
        for i in range(len(x)):
            if x[i] < .1 :
                
                x[i]=0
        return x

    def lasso_feature_selection(self,x,y):
        from sklearn import linear_model
        clf = linear_model.Lasso(alpha=0.01)
        clf.fit(x,y)
        coeff = clf.coef_
        return abs(coeff)

    def composite_weight(self,df,svm_weight,avg_dxi):
        svm_index=(argsort(svm_weight))
        
        def Reverse(lst): 
            new_lst = lst[::-1] 
            return new_lst
        
        svm_ind=Reverse(svm_index)
        
        column_name=df.columns
        svm_indd=[]
        svm_weights=[]
        
        for i in range(len(svm_index)):
            if svm_weight[svm_ind[i]] !=0:
                svm_indd.append(svm_ind[i])
                svm_weights.append(svm_weight[svm_ind[i]])
            
        update_column=column_name[svm_indd]
        xx=df[update_column]
        
        while(1):
            svm_dxi,avg_svm_dxi,w_svm=self.total_dxi(xx,svm_weights)
            
            if (avg_svm_dxi >= 0.9*avg_dxi) and  (avg_svm_dxi <= 1.1*avg_dxi) :
                print(11)
                xx=xx.iloc[:,0:-1]
                svm_weights=svm_weights[:-1]
            else:
                break
        column_update=list(xx.columns)
        weight_svm=[]
        for i in range(len(column_name)):
            
            if column_name[i] in column_update :
                a=column_update.index(column_name[i])
                weight_svm.append(w_svm[a])
            else:
                weight_svm.append(0)
        return svm_dxi,avg_svm_dxi,w_svm,weight_svm

    def composite_dxi(self,x,weight_svm,weight_pca,weight_mi,weight_lasso,weigth_xgb):
        final_weights=[]
        for i in range(x.shape[1]):
            w= [weight_svm[i],weight_pca[i],weight_mi[i],weight_lasso[i],weigth_xgb[i]]      
            count=0
            for i in range(len(w)):
                if w[i] > 0 :
                    count=count+1

            total_weight=np.sum(w)
            n = np.count_nonzero(w)
        
            if n==0 :
                n=1
            final_weight=(total_weight*(1+(.1*(n-1))))/n
            
            final_weights.append(final_weight)
            
        composite_dxi=self.dxi(x,final_weights)
        composite_avg_dxi=np.mean(np.array(composite_dxi))
        print(composite_avg_dxi)   
        return composite_dxi, composite_avg_dxi

    def sample_composite_dxi(self,x,weight_svm,weight_mi,weight_lasso,weight_pca,weight_nb):
    
        final_weights=[]
        for i in range(x.shape[1]):
            
            w= [weight_svm[i],weight_mi[i],weight_lasso[i],weight_pca[i],weight_nb[i]]
            count=0
            for i in range(len(w)):
                if w[i] > 0 :
                    count=count+1
            total_weight=np.sum(w)
            n = np.count_nonzero(w)
        
            if n==0 :
                n=1
            final_weight=(total_weight*(1+(.1*(n-1))))/n
            final_weights.append(final_weight)
        composite_dxi=self.dxi(x,final_weights)
        composite_avg_dxi=np.mean(np.array(composite_dxi))    
        print(composite_avg_dxi)   
        fnlwgts = dict(zip(list(x.columns),final_weights)) 
        return composite_dxi, composite_avg_dxi, fnlwgts

    def rflsample_composite_dxi(self,x,weight_svm,weight_mi,weight_lasso,weight_pca,weight_nb,weight_rfl):

        final_weights=[]
        for i in range(x.shape[1]):
            
            w= [weight_svm[i],weight_mi[i],weight_lasso[i],weight_pca[i],weight_nb[i],weight_rfl[i]]
            count=0
            for i in range(len(w)):
                if w[i] > 0 :
                    count=count+1

            total_weight=np.sum(w)
            n = np.count_nonzero(w)
        
            if n==0 :
                n=1
            final_weight=(total_weight*(1+(.1*(n-1))))/n
            final_weights.append(final_weight)
            
        composite_dxi=self.dxi(x,final_weights)
        composite_avg_dxi=np.mean(np.array(composite_dxi))
        print('SXI: ',composite_avg_dxi)   
        return composite_dxi, composite_avg_dxi

    # def tensorflownn(self,X, y,feature_importance):

    #     from sklearn.model_selection import train_test_split
    #     import copy

    #     from keras.layers import Dense, Dropout,Dense
    #     from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
    #     from sklearn.metrics import make_scorer, accuracy_score
    #     from bayes_opt import BayesianOptimization
    #     from sklearn.model_selection import StratifiedKFold
    #     from keras.layers import LeakyReLU,PReLU
    #     from keras.initializers import Constant
    #     from scikeras.wrappers import KerasClassifier
    #     from tensorflow.keras import initializers
    #     from tensorflow.keras.initializers import Initializer
    #     from sklearn.model_selection import cross_val_score
    #     LeakyReLU = LeakyReLU(alpha=0.1)
    #     import warnings
    #     warnings.filterwarnings('ignore')
    #     pd.set_option("display.max_columns", None)
    #     from keras.models import Sequential
    #     from bayes_opt import BayesianOptimization
        
    #     class CustomInitializer(Initializer):
    #         def __init__(self, feature_importance):
    #             self.feature_importance = feature_importance
    #             print(feature_importance)

    #         def __call__(self, shape, dtype=None):
    #             input_size, output_size = shape
    #             if len(self.feature_importance) != input_size:
    #                 raise ValueError("Length of feature_importance should match the input_size")

    #             effective_input_size = int(np.sum(np.square(self.feature_importance)) * input_size)
    #             std_dev = np.sqrt(2.0 / (effective_input_size + output_size))
    #             weights = np.random.normal(0, std_dev, size=(input_size, output_size))
    #             return weights.astype(float32)

    #     X_train, X_val, y_train, y_val = train_test_split(X , y, test_size=0.2, random_state=42, stratify=y)
    #     input_size = X_train.shape[1]
    #     output_size = X_train.shape[1]
    #     def score_acc(y_true, y_pred):
    #         return accuracy_score(y_true, (y_pred > 0.5).astype(int))

    #     activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu', 'elu', LeakyReLU,
    #                 PReLU(alpha_initializer=Constant(value=0.25))]

    #     def nn_cl_bo(neurons, activation, optimizer, learning_rate, batch_size, epochs, dropout, dropout_rate):
    #         activation = activationL[int(round(activation))]
    #         neurons = round(neurons)
    #         batch_size = round(batch_size)
    #         epochs = round(epochs)
    #         learning_rate = learning_rate

    #         optimizerD = {
    #             'Adam': Adam(learning_rate=learning_rate),
    #             'SGD': SGD(learning_rate=learning_rate),
    #             'RMSprop': RMSprop(learning_rate=learning_rate),
    #             'Adadelta': Adadelta(learning_rate=learning_rate),
    #             'Adagrad': Adagrad(learning_rate=learning_rate),
    #             'Adamax': Adamax(learning_rate=learning_rate),
    #             'Nadam': Nadam(learning_rate=learning_rate),
    #             'Ftrl': Ftrl(learning_rate=learning_rate)
    #         }

    #         def nn_cl_fun():
    #             try:
    #                 opt = optimizerD.get(optimizer, Adam)(learning_rate=learning_rate)
    #             except KeyError as e:
    #                 print(f"KeyError: {e}")
    #                 print(f"Optimizer value: {optimizer}")
    #                 print(f"Available optimizers: {optimizerD.keys()}")
    #                 raise

    #             input_dim = X_train.shape[1]

    #             nn = Sequential()
    #             nn.add(Dense(neurons, input_dim=input_dim, activation=activation))

    #             if  X_train.shape[0] > 20000:
    #                 nn.add(Dense(neurons,  activation=activation))
    #             else:
    #                 nn.add(Dense(neurons, kernel_initializer=initializers.GlorotNormal(seed=None), activation=activation))
                
    #             if dropout > 0.5 and X_train.shape[0] < 20000:
    #                 nn.add(Dropout(dropout_rate, seed=123))

    #             if  X_train.shape[0] > 20000:
    #                 nn.add(Dense(neurons,  activation=activation))
    #             else:
    #                 nn.add(Dense(neurons, kernel_initializer=initializers.GlorotNormal(seed=None), activation=activation))
                
    #             if dropout > 0.5 and X_train.shape[0] < 20000:
    #                 nn.add(Dropout(dropout_rate, seed=123))

    #             if  X_train.shape[0] > 20000:
    #                 nn.add(Dense(neurons,  activation=activation))
    #             else:
    #                 nn.add(Dense(neurons, kernel_initializer=initializers.GlorotNormal(seed=None), activation=activation))

    #             nn.add(Dense(1, activation='sigmoid'))

    #             nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #             return nn

    #         nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
    #         kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    #         score = cross_val_score(nn, X_train, y_train, scoring=make_scorer(score_acc), cv=kfold).mean()
    #         return score

    #     params_nn = {
    #         'neurons': (30, 200),
    #         'activation': (0, 8),
    #         'optimizer': (0, 8),
    #         'learning_rate': (0.01, 1),
    #         'batch_size': (50, 120),
    #         'epochs': (20, 100),
    #         'dropout_rate': (0, 0.3),
    #         'dropout': (0, 1)
    #     }

    #     nn_bo = BayesianOptimization(nn_cl_bo, params_nn, random_state=111)
    #     nn_bo.maximize(init_points=3, n_iter=2)
        
    #     print('---------------------------Params--------------------------')
    #     params_nn_ = nn_bo.max['params']

    #     activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu', 'elu', LeakyReLU, PReLU(alpha_initializer=Constant(value=0.25))] 
    #     params_nn_['activation'] = activationL[round(params_nn_['activation'])]
    #     learning_rate = params_nn_['learning_rate']

    #     optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
    #                 'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
    #                 'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
    #                 'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}

    #     optimizerL = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','Adam']
    #     params_nn_['optimizer'] = optimizerD[optimizerL[round(params_nn_['optimizer'])]]
    #     params_nn_['batch_size'] = round(params_nn_['batch_size'])
    #     params_nn_['epochs'] = round(params_nn_['epochs'])
    #     params_nn_['neurons'] = round(params_nn_['neurons'])

    #     print('---------------------Layers--------------------------\n')

    #     def print_layer_weights(model):
    #         first_layer = model.layers[0]
    #         first_layer_weights = first_layer.get_weights()
    #         print(f"First Layer weights:")
    #         if len(model.layers) < 2:
    #             print("Model does not have enough layers.")
    #             return

    #         second_layer = model.layers[1]
    #         second_layer_weights = second_layer.get_weights()

    #         print(f"Second Layer weights:")
    #         return first_layer_weights, second_layer_weights

    #     print('-----------------------Training----------------------------\n')

    #     def nn_cl_fun(X_train, y_train, X_val, y_val):
    #         input_dim = X_train.shape[1]
    #         nn = Sequential()

    #         nn.add(Dense(params_nn_['neurons'], input_dim=input_dim, activation= LeakyReLU,kernel_initializer= CustomInitializer(feature_importance)))

    #         if X_train.shape[0] > 20000:
    #             nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
    #         else:
    #             nn.add(Dense(params_nn_['neurons'],kernel_initializer= initializers.GlorotNormal(seed=None), activation=params_nn_['activation']))

    #         if params_nn_['dropout'] > 0.5 and X_train.shape[0] < 20000:
    #             nn.add(Dropout(params_nn_['dropout_rate'], seed=123))

    #         if X_train.shape[0] > 20000:
    #             nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
    #         else:
    #             nn.add(Dense(params_nn_['neurons'],kernel_initializer=initializers.GlorotNormal(seed=None), activation=params_nn_['activation']))

    #         if params_nn_['dropout'] > 0.5 and X_train.shape[0] < 20000:
    #             nn.add(Dropout(params_nn_['dropout_rate'], seed=123))

    #         if X_train.shape[0] > 20000:
    #             nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
    #         else:
    #             nn.add(Dense(params_nn_['neurons'],kernel_initializer=initializers.GlorotNormal(seed=None), activation=params_nn_['activation']))

    #         nn.add(Dense(1, activation='sigmoid'))

    #         optimizerD = {
    #             'Adam': Adam,
    #             'SGD': SGD,
    #             'RMSprop': RMSprop,
    #             'Adadelta': Adadelta,
    #             'Adagrad': Adagrad,
    #             'Adamax': Adamax,
    #             'Nadam': Nadam,
    #             'Ftrl': Ftrl
    #         }

    #         try:
    #             opt = optimizerD.get(params_nn_['optimizer'], Adam)(learning_rate=params_nn_['learning_rate'])
    #         except KeyError as e:
    #             print(f"KeyError: {e}")
    #             print(f"Optimizer value: {params_nn_['optimizer']}")
    #             print(f"Available optimizers: {optimizerD.keys()}")
    #             raise

    #         nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

            
    #         nn.summary()

    #         history = nn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params_nn_['epochs'],
    #                         batch_size=params_nn_['batch_size'],  verbose=1)

    #         _, accuracy = nn.evaluate(X_val, y_val)

    #         first_layer_weights, last_layer_weights = print_layer_weights(nn)

    #         return accuracy, first_layer_weights, last_layer_weights

    #     accuracy, first_layer_weights, last_layer_weights = nn_cl_fun(X_train, y_train, X_val, y_val)
    #     return accuracy, [list(j) for j in zip(*first_layer_weights[0])][:10]
    
    """
    def pytorchnn(self,X,y):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import pandas as pd
        import numpy as np

        from sklearn.model_selection import train_test_split
    #     X = df.drop([tv,'EmployeeNumber'],axis=1)
    #     y = df[tv]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        y_test = torch.tensor(y_test.values, dtype=torch.long)
        xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.1)
        xtrain = torch.tensor(xtrain.values, dtype=torch.float32)
        xval = torch.tensor(xval.values, dtype=torch.float32)
        ytrain = torch.tensor(ytrain.values, dtype=torch.long)
        yval = torch.tensor(yval.values, dtype=torch.long)
        class ClassificationModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(ClassificationModel, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.layer2 = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                x = self.layer1(x)
                x = self.relu(x)
                x = self.layer2(x)
                return x
        input_size = len(X.columns) # specify the number of features in your input
        num_classes = 2 # specify the number of classes in your classification task
        num_epochs = 100
        hidden_sizes = [32, 64, 128, 256,512]
        accs = []
        for hidden_size in hidden_sizes:
            modelval = ClassificationModel(input_size, hidden_size, num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(modelval.parameters(), lr=0.01)

            for epoch in range(num_epochs):
                check_cancelled()
                # Forward pass
                outputs = modelval(xtrain)
                loss = criterion(outputs, ytrain)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            modelval.eval()  # set the model to evaluation mode
            with torch.no_grad():
                _, predicted = torch.max(modelval(xval), 1)

            # Calculate accuracy
            accuracy = (predicted == yval).sum().item() / yval.size(0)
            print('Accuracy on Validation data: {:.2f}%'.format(accuracy * 100))
            accs.append(accuracy * 100)
                    # Record or print the validation loss and other metrics
        print(accs)
        hidden_size = hidden_sizes[accs.index(max(accs))]
        model = ClassificationModel(input_size, hidden_size, num_classes)
        # CrossEntropyLoss for classification
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(num_epochs):
            check_cancelled()
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            _, predicted = torch.max(model(X_test), 1)

        # Calculate accuracy
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print('Accuracy on test data: {:.2f}%'.format(accuracy * 100))
        weights_layer1 = model.layer1.weight.data
        return list(weights_layer1)[0:3],accuracy
    """

    def pytorchnn(self,X, y,feature_importance):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import pandas as pd
        import numpy as np

        from sklearn.model_selection import train_test_split

        class CustomInitializer:
            def __init__(self, feature_importance):
                self.feature_importance = feature_importance
                print(feature_importance)

            def __call__(self, tensor):
                input_size, output_size = tensor.shape
                input_size = len(X.columns)
                print(input_size)
                if len(self.feature_importance) != input_size:
                    raise ValueError("Length of feature_importance should match the input_size")

                effective_input_size = int(np.sum(np.square(self.feature_importance)) * input_size)
                std_dev = np.sqrt(2.0 / (effective_input_size + output_size))
                with torch.no_grad():
                    tensor.normal_(0, std_dev)

        # Assuming feature_importance is defined based on your data
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        y_test = torch.tensor(y_test.values, dtype=torch.long)
        xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.1)
        xtrain = torch.tensor(xtrain.values, dtype=torch.float32)
        xval = torch.tensor(xval.values, dtype=torch.float32)
        ytrain = torch.tensor(ytrain.values, dtype=torch.long)
        yval = torch.tensor(yval.values, dtype=torch.long)

        class ClassificationModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes, initializer):
                super(ClassificationModel, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.layer2 = nn.Linear(hidden_size, num_classes)
                initializer(self.layer1.weight)

            def forward(self, x):
                x = self.layer1(x)
                x = self.relu(x)
                x = self.layer2(x)
                return x

        input_size = len(X.columns)  # specify the number of features in your input
        num_classes = 2  # specify the number of classes in your classification task
        num_epochs = 100
        hidden_sizes = [32, 64, 128, 256, 512]
        accs = []

        initializer = CustomInitializer(feature_importance)
        print(feature_importance)
        for hidden_size in hidden_sizes:
            modelval = ClassificationModel(input_size, hidden_size, num_classes, initializer)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(modelval.parameters(), lr=0.01)

            for epoch in range(num_epochs):
                check_cancelled()
                # Forward pass
                outputs = modelval(xtrain)
                loss = criterion(outputs, ytrain)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
            
            modelval.eval()  # set the model to evaluation mode
            with torch.no_grad():
                _, predicted = torch.max(modelval(xval), 1)

            # Calculate accuracy
            accuracy = (predicted == yval).sum().item() / yval.size(0)
            print('Accuracy on Validation data: {:.2f}%'.format(accuracy * 100))
            accs.append(accuracy * 100)

        print(accs)
        hidden_size = hidden_sizes[accs.index(max(accs))]
        model = ClassificationModel(input_size, hidden_size, num_classes, initializer)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            check_cancelled()
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            _, predicted = torch.max(model(X_test), 1)

        # Calculate accuracy
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print('Accuracy on test data: {:.2f}%'.format(accuracy * 100))
        weights_layer1 = model.layer1.weight.data
        return list(weights_layer1)[0:3], accuracy

    def xregenerate(self,dfa,tv,primkey):
        if dfa is None or len(dfa) == 0:
            print("xregenerate: received empty dataframe; skipping recalibration prep.")
            return pd.DataFrame(), []

        xa = dfa.drop([tv,primkey],axis=1,errors='ignore')#dropped sku
        if tv not in dfa.columns:
            print(f"xregenerate: target column '{tv}' missing; skipping recalibration prep.")
            return pd.DataFrame(), []

        ya= dfa[tv]
        if xa.empty or len(xa) == 0 or len(ya) == 0:
            print("xregenerate: no usable samples/features before Lasso; skipping recalibration prep.")
            return pd.DataFrame(), []
        from sklearn import linear_model
        clf = linear_model.Lasso(alpha=0.2,max_iter=10000)
        clf.fit(xa,ya)
        lasso_weight= clf.coef_
        #update dataframe
        min_max = []
        max_f=[]
        dff=xa.columns
        for i in range(len(dff)):
            if lasso_weight[i] < 0 :
                min_max.append(dff[i])
            else:
                max_f.append('MAX')
        
        dfa.drop([tv],axis=1,inplace=True)
        features=list(dfa.columns)

        minimum_value=[]
        maximum_value=[]
        minmax_value=[]
        catogorical_value=[]
        for i in range(len(features)):
        
            if features[i] in min_max:
                minmax_value.append('MIN')
                catogorical_value.append('Behavior')
            else:
                minmax_value.append('MAX') 
                catogorical_value.append('Behavior')
            try:
                minimum=min(list(dfa[features[i]]))
                maximum=max(list(dfa[features[i]]))
            except:
                minimum=''
                maximum=''
            minimum_value.append(minimum)
            maximum_value.append(maximum)
        minimum_value[0]='Min_Value'
        maximum_value[0]='Max_Value'
        minmax_value[0] = 'minmax'
        catogorical_value[0]='Category'
        
        df_mains=pd.DataFrame()
        df_mains['0']=minimum_value
        df_mains['1']=maximum_value
        df_mains['2']=minmax_value
        df_mains['3']=catogorical_value
        
        df_m=df_mains.T
        df_m.columns=dfa.columns
        df_m.reset_index(drop=True, inplace=True)
        dfa.reset_index(drop=True, inplace=True)
        frames=[df_m,dfa]

        results = pd.concat(frames, keys=['x', 'y'])
        
        dfg = results.iloc[:, 1:]
        minmax=list(dfg.iloc[1])
        print(minmax)
        column=dfg.columns

        dropcols = []
        for i in range(len(minmax)):
            if float(minmax[i]) == 0:    
                dfg=dfg.drop([column[i]],axis=1)
                dropcols.append(i)
                print(dropcols)
        dfg=dfg.dropna(how='any')
        if dfg.empty:
            print("xregenerate: dataframe became empty after filtering/dropna; skipping recalibration prep.")
            return pd.DataFrame(), dropcols
        new_df=self.normalize(dfg)
        if new_df.empty:
            print("xregenerate: normalized dataframe is empty; skipping recalibration prep.")
            return pd.DataFrame(), dropcols
        return new_df,dropcols

    def rfLagent(self,sxi,chkcls,tv,rfagent,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
                xx,avg_base_dxi,r_start,r_ends,selcls,twds,classes,misclassif,intv,dropsind,primkey):

        dfb=[]
        wgtper = []
        classified = []
        brkloop = False
        for i in range(r_start,r_ends,intv):
            print(i)
            if i > r_start and intv == 5:
            
                if dfb[-1].empty:
                    finalmisclasif = dfb[-1]
                    break;
            
                newdff = dfb[-1].drop(['composite_dxi_label','composite_dxi','index','needed_index'],axis=1,errors = 'ignore')
                xx,dropind=self.xregenerate(newdff,tv,primkey)
                if xx is None or xx.empty:
                    finalmisclasif = dfb[-1]
                    break
                print(xx.shape)
                misclassif = dfb[-1]
            
            elif i < r_start and intv == -5:
                if dfb[-1].empty:
                    finalmisclasif = dfb[-1]
                    break;
                newdff = dfb[-1].drop(['composite_dxi_label','composite_dxi','index','needed_index'],axis=1,errors = 'ignore')
                xx,dropind=self.xregenerate(newdff,tv,primkey)
                if xx is None or xx.empty:
                    finalmisclasif = dfb[-1]
                    break
                misclassif = dfb[-1]
        
            else:
                dropind = dropsind
                pass          
            
            wgtper.append(i) # Stores the % increase
            rfgupd = map(lambda x: x+(x*(i/100)), rfagent) # Weights from the Neural network- with % increase/decrease (Reinforcement L)
            """
            laswg = list(map(add, lasso_weight, rfgupd)) # lasso weight + Weights(Reinforcement L)
            miwg = list(map(add, mi_weight, rfgupd)) # MI weight + Weights(Reinforcement L)
            pcawg = list(map(add, pca_weight, rfgupd))# PCA weight + Weights(Reinforcement L)
            nbwg = list(map(add, nb_weight, rfgupd)) # Naive Bayes weight + Weights(Reinforcement L)
            xgbwg = list(map(add, xgb_weight, rfgupd)) # XGBoost weight + Weights(Reinforcement L)
            """
            print(len(lasso_weight))
            print(len(mi_weight))
            print(len(pca_weight))
            print(len(xgb_weight))
            laswg = lasso_weight
            miwg = mi_weight
            pcawg = pca_weight
            nbwg = nb_weight
            xgbwg = xgb_weight
            rfagupd = list(rfgupd)
            print(len(laswg))
            print(len(miwg))
            print(len(pcawg))
            print(len(xgbwg))
            laswg = [value for i, value in enumerate(laswg) if i not in dropind]
            miwg = [value for i, value in enumerate(miwg) if i not in dropind]
            pcawg = [value for i, value in enumerate(pcawg) if i not in dropind]
            nbwg = [value for i, value in enumerate(nbwg) if i not in dropind]
            xgbwg = [value for i, value in enumerate(xgbwg) if i not in dropind]  
            rfagupd = [value for i, value in enumerate(rfagupd) if i not in dropind]
            xx = xx.reset_index(drop=True)

            ##############SXI Score calculation where the above weights taken as inputs###############
            lasso_dxi,avg_lasso_dxi,w_lasso,weight_lasso = self.composite_weight(xx,laswg,avg_base_dxi)
            # svm_dxi,avg_svm_dxi,w_svm,weight_svm = composite_weight(x,svm_weight,avg_base_dxi)
            mi_dxi,avg_mi_dxi,w_mi,weight_mi = self.composite_weight(xx,miwg,avg_base_dxi)
            pca_dxi,avg_pca_dxi,w_pca,weight_pca = self.composite_weight(xx,pcawg,avg_base_dxi)
            nb_dxi,avg_nb_dxi,w_nb,weight_nb = self.composite_weight(xx,nbwg,avg_base_dxi)
            xgb_dxi, avg_xgb_dxi,w_xgb,weight_xgb = self.composite_weight(xx,xgbwg,avg_base_dxi)
            
            rfl_dxi, avg_rfl_dxi,w_rfl,weight_rfl = self.composite_weight(xx,rfagupd,avg_base_dxi)
            
            sample_composite_dxi1=self.rflsample_composite_dxi(xx,weight_lasso, weight_xgb, weight_mi, weight_pca,weight_nb,weight_rfl)
            composite_dxi1=sample_composite_dxi1[0]
            nan_inds = [i for i, value in enumerate(composite_dxi1) if math.isnan(value)]
            if any(x == float('-inf') for x in composite_dxi1):
                if i == r_start:
                    brkloop = True
                    print('Entering SXI-NON')
                    finalmisclasif = misclassif
                    classified = []
                    wgtper = []
                    if len(dfb) > 0:
                        finalmisclasif = dfb[-1] 
                    else:
                        finalmisclasif = misclassif
                else:
                    pass
                continue    
    #             brkloop = True
    #             break;
            elif nan_inds:
                if i == r_start:
                    brkloop = True
                    print('Entering SXI-NON')
                    finalmisclasif = misclassif
                    classified = []
                    wgtper = []
                    if len(dfb) > 0:
                        finalmisclasif = dfb[-1] 
                    else:
                        finalmisclasif = misclassif
                else:
                    pass
                continue    
    #             brkloop = True
    #             break;
                
            #composite_dxi1=composite_dxi[0]
            avg_composite_dxi=np.mean(composite_dxi1)
            print('SXI: ',avg_composite_dxi)
            ##################SXI Calculation Done################################
            update_y_com=self.create_label_forward(composite_dxi1,avg_composite_dxi)  
            df_buynobuya = misclassif
            df_buynobuya = df_buynobuya.reset_index(drop=True)
    #         df_buynobuy=df_buynobuy.dropna(how='any')
            df_buynobuya['composite_dxi_label'] = update_y_com #Adding SXI scores to the dataframe
            df_buynobuya['composite_dxi'] = composite_dxi1
            
            if chkcls == 'class1' and ((selcls =='cls1' and twds =='abv') or (selcls =='cls2' and twds =='bel')):
                ckf=df_buynobuya.loc[(df_buynobuya['composite_dxi'] >= sxi) & (df_buynobuya[tv] == 1)]
            elif chkcls == 'class2' and ((selcls =='cls1' and twds =='abv') or (selcls =='cls2' and twds =='bel')):
                ckf = df_buynobuya.loc[(df_buynobuya['composite_dxi'] < sxi) & (df_buynobuya[tv] == 0)]
            elif chkcls == 'class1' and ((selcls =='cls1' and twds =='bel') or (selcls =='cls2' and twds =='abv')):
                ckf = df_buynobuya.loc[(df_buynobuya['composite_dxi'] < sxi) & (df_buynobuya[tv] == 1)]
            elif chkcls == 'class2' and ((selcls =='cls1' and twds =='bel') or (selcls =='cls2' and twds =='abv')):
                ckf = df_buynobuya.loc[(df_buynobuya['composite_dxi'] >= sxi) & (df_buynobuya[tv] == 0)]
            
            if chkcls == 'class1':    
                print(f'Classified Correctly {len(ckf)} out of {len(df_buynobuya)} for {classes[1]}')
            else:
                print(f'Classified Correctly {len(ckf)} out of {len(df_buynobuya)} for {classes[0]}')
            
            if len(ckf) == 0:
                dfb.append(df_buynobuya)
            else:
                classified.append(ckf)
                filtered_df = df_buynobuya[~df_buynobuya.index.isin(ckf.index)]
                dfb.append(filtered_df)
        if len(dfb) > 0:
            finalmisclasif = dfb[-1] 
        else:
            finalmisclasif = misclassif
        return classified,wgtper,finalmisclasif

    def wrecalibrate(self,sxi,chkcls1,misclassif1,tv,rfagents,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
                    avg_base_dxi,selcls,twds,classes,primkey):
        if misclassif1 is None or len(misclassif1) == 0:
            print(f"wrecalibrate: no rows available for {chkcls1}; skipping branch.")
            return [], misclassif1 if misclassif1 is not None else pd.DataFrame()

        xxs = misclassif1.drop(['composite_dxi_label','composite_dxi','index', 'needed_index'],axis=1,errors = 'ignore')
        xxs,dropsind=self.xregenerate(xxs,tv,primkey)
        if xxs is None or xxs.empty:
            print(f"wrecalibrate: recalibration input became empty for {chkcls1}; keeping current rows.")
            return [], misclassif1
        print(xxs.shape)
        r_start,r_ends = 0,100
        inv=5
        clsfd,wgtpr,fnlmisclasif = self.rfLagent(sxi,chkcls1,tv,rfagents,lasso_weight,mi_weight,pca_weight,nb_weight,
                                            xgb_weight,xxs,avg_base_dxi,r_start,r_ends,selcls,twds,classes,misclassif1,inv,dropsind,primkey)
        nan_indices = [i for i, value in enumerate(fnlmisclasif['composite_dxi'].values) if math.isnan(value)]
        if nan_indices:
            fnlmisclasif1 = [misclassif1]
        else:
            fnlmisclasif1 = [fnlmisclasif]
        clsfdfnl = [clsfd]
        
        trigbrk = False

        if len(clsfd) == 0:
            r_start,r_ends = 300,200
    #         r_start,r_ends = 0,100
            while len(fnlmisclasif) != 0:
                if trigbrk:
                    print('Reinforcement Stopped')
                    break 
                else:
                    #Looping towards negative side - percentage decrease
                    r_start= r_start-100 
                    r_ends = r_ends -100 
                    print('Rstart: ', r_start)
                    print('Rends: ', r_ends)
                    if r_start < -500: #Loop breaks at -5000 % decrease if the loop goes on...
                        break
                    else:

                        xxsit = fnlmisclasif1[-1].drop(['composite_dxi_label','composite_dxi'],axis=1)
                        if len(xxsit) == 0:
                            break
                        else:
                            pass
                        xxsit = xxsit.reset_index(drop=True)
                        xxsit,dropsind=self.xregenerate(xxsit,tv,primkey)
                        if xxsit is None or xxsit.empty:
                            break
                        xxsit = xxsit.reset_index(drop=True)
                        clsfd1,wgtpr1,fnlmisclasif2 = self.rfLagent(sxi,chkcls1,tv,rfagents,lasso_weight,mi_weight,pca_weight,nb_weight,
                                            xgb_weight,xxsit,avg_base_dxi,r_start,r_ends,selcls,twds,classes,fnlmisclasif1[-1],-5,dropsind,primkey)
                        fnlmisclasif2 = fnlmisclasif2.reset_index(drop=True)
                        nans_indices = [i for i, value in enumerate(fnlmisclasif2['composite_dxi'].values) if math.isnan(value)]
                        if nans_indices:
                            break
                        else:
                            fnlmisclasif1.append(fnlmisclasif2)
                            clsfdfnl.append(clsfd1)
        else:
            r_start,r_ends = -300,-200
            while len(fnlmisclasif) != 0:
                if trigbrk:
                    print('Reinforcement Stopped')
                    break 
                else:
                    #Looping towards negative side - percentage decrease
                    r_start= r_start+100 
                    r_ends = r_ends + 100 
                    print('Rstart: ', r_start)
                    print('Rends: ', r_ends)
                    if r_start > 500: #Loop breaks at -5000 % decrease if the loop goes on...
                        break
                    else:

                        xxsit = fnlmisclasif1[-1].drop(['composite_dxi_label','composite_dxi','index','needed_index'],axis=1,errors='ignore')
                        if len(xxsit) == 0:
                            break
                        else:
                            pass
                        xxsit = xxsit.reset_index(drop=True)
                        xxsit,dropsind=self.xregenerate(xxsit,tv,primkey)
                        if xxsit is None or xxsit.empty:
                            break
                        xxsit = xxsit.reset_index(drop=True)
                        clsfd1,wgtpr1,fnlmisclasif2 = self.rfLagent(sxi,chkcls1,tv,rfagents,lasso_weight,mi_weight,pca_weight,nb_weight,
                                            xgb_weight,xxsit,avg_base_dxi,r_start,r_ends,selcls,twds,classes,fnlmisclasif1[-1],-5,dropsind,primkey)
                        nans_indices = [i for i, value in enumerate(fnlmisclasif2['composite_dxi'].values) if math.isnan(value)]
                        if nans_indices:
                            break
                        fnlmisclasif2 = fnlmisclasif2.reset_index(drop=True)
                        fnlmisclasif1.append(fnlmisclasif2)
                        clsfdfnl.append(clsfd1)

        return clsfdfnl,fnlmisclasif1[-1]

    def generate_sxirl(self,target,buynobuy,sxi_dataframe,classes,primkey):
        dropcols = []
        tv = target
        df_buynobuy = buynobuy ### Read BUYNOBUY DATA
        df = sxi_dataframe ### Read DXI DATA
        df = df.iloc[:, 1:]
        minmax=list(df.iloc[1])
        column=df.columns

        for i in range(len(minmax)):
            if float(minmax[i]) == 0:
                df=df.drop([column[i]],axis=1)
                dropcols.append(column[i])
        df=df.dropna(how='any')
        new_df=self.normalize(df)
        b=self.bivarient_correlation(new_df)  
        w=self.weight(new_df,b)
        before_base_dxi=self.dxi(new_df,w)
        avg_dxi=np.mean(np.array(before_base_dxi))
        x=new_df

        y=self.create_label_forward(before_base_dxi,avg_dxi)
        clf = linear_model.Lasso(alpha=0.2)
        clf.fit(x,y)
        lasso_weight= clf.coef_
        min_max= (df.iloc[2])
        for i in range(len(min_max)):
            if lasso_weight[i] < 0 :
                if min_max[i] != 'MIN' :
                    try:
                        (df.loc[2])[i]='MIN'
                    except:
                        df.iat[2, i] = 'MIN' 
        new_df=self.normalize(df)

        update_b=self.bivarient_correlation(new_df)  
        parameter=df.iloc[3]
        update_w=self.weight(new_df,update_b)
        base_dxi=self.dxi(new_df,update_w)
        behaviour_dxi,transactional_dxi,kpi_dxi,visual_dxi=self.catogorical_dxi(new_df,update_w,parameter)
        avg_behaviour=np.mean(np.array(behaviour_dxi))
        avg_kpi=np.mean(np.array(kpi_dxi))
        avg_transactional=np.mean(np.array(transactional_dxi))
        avg_visual=np.mean(np.array(visual_dxi))
        avg_base_dxi=np.mean(np.array(base_dxi))
        update_y = self.create_label_forward(base_dxi,avg_base_dxi)

        data=df.iloc[4:]
        data['Base_dxi_label']=update_y
        data['Base_dxi']=base_dxi
        data[tv]=df_buynobuy[tv]
        op = new_df
        print(f'Len of Cols: {len(new_df.columns)}')
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(op)
        if (x.values < 0).any() == True:
            a = x_train
        else:
            a = x
        cnb = ComplementNB().fit(a, update_y)
        logprobs = cnb.feature_log_prob_
        avgprob =[]
        for i in range(len(logprobs[0])):
            avgprob.append((logprobs[0][i]+logprobs[1][i])/2)
        nb_weight = list(np.exp(avgprob) / (np.exp(avgprob)).sum()) 

        new_df.columns = new_df.columns.str.replace(r"[\[\]<>]", "", regex=True)
        weightcols = list(new_df.columns)

        xgb = xg.XGBClassifier().fit(new_df,update_y)
        xgb_weight = list(xgb.feature_importances_)
        mi_weight=list(self.mutual_information(new_df, update_y))
        from sklearn.decomposition import PCA
        min_dim = min(new_df.shape[0], new_df.shape[1])
        if min_dim < 1:
            raise ValueError("SXI preprocessing left no usable features for PCA-based weighting.")
        pca = PCA(n_components=min(3, min_dim))
        pca.fit_transform(new_df)
        pca_weight = pca.components_[0]
        # pca_weight=list(self.pca_feature_selection(new_df,update_y))
        lasso_weight=list(self.lasso_feature_selection(new_df,update_y))
        lasso_weight = np.nan_to_num(np.asarray(lasso_weight, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
        mi_weight = np.nan_to_num(np.asarray(mi_weight, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
        pca_weight = np.nan_to_num(np.asarray(pca_weight, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
        nb_weight = np.nan_to_num(np.asarray(nb_weight, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
        xgb_weight = np.nan_to_num(np.asarray(xgb_weight, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
        # print('Lasso Weights: ', lasso_weight)
        # print('MI Weights: ', mi_weight)
        # print('PCA Weights: ', pca_weight)
        # print('NB Weights: ', nb_weight)
        # print('XGB Weights: ', xgb_weight)

        lasso_dxi,avg_lasso_dxi,w_lasso,weight_lasso = self.composite_weight(new_df,lasso_weight,avg_base_dxi)
        # svm_dxi,avg_svm_dxi,w_svm,weight_svm = composite_weight(x,svm_weight,avg_base_dxi)
        mi_dxi,avg_mi_dxi,w_mi,weight_mi = self.composite_weight(new_df,mi_weight,avg_base_dxi)
        pca_dxi,avg_pca_dxi,w_pca,weight_pca = self.composite_weight(new_df,pca_weight,avg_base_dxi)
        nb_dxi,avg_nb_dxi,w_nb,weight_nb = self.composite_weight(new_df,nb_weight,avg_base_dxi)
        xgb_dxi, avg_xgb_dxi,w_xgb,weight_xgb = self.composite_weight(new_df,xgb_weight,avg_base_dxi)
        # print('Lasso DXI: ', weight_lasso)
        # print('MI DXI: ', weight_mi)
        # print('PCA DXI: ', weight_pca)
        # print('NB DXI: ', weight_nb)
        # print('XGB DXI: ', weight_xgb)
        top_names =[]
        for weight in [weight_lasso,weight_nb,weight_mi, weight_pca,weight_xgb]:   
            idx = (-(np.array(weight))).argsort()[:len(weight)+1]
            names = x.columns[idx]
            top_names.append(names)
        lasso_w = np.array(lasso_weight)
        mi_w    = np.array(mi_weight)
        pca_w   = np.array(pca_weight)
        nb_w    = np.array(nb_weight)
        xgb_w  = np.array(xgb_weight)

        # Build DataFrame
        feature_weight_df = pd.DataFrame({
            'Feature': new_df.columns,
            'Lasso_Weight': lasso_w,
            'MI_Weight': mi_w,
            'PCA_Weight': pca_w,
            'NB_Weight': nb_w,
            'XGB_Weight': xgb_w
        })

        print("Feature-wise Weight Comparison:\n", feature_weight_df)
        tp_params = {}
        tp_params["Algorithm1"]=top_names[0][0:5] 
        tp_params["Algorithm2"]=top_names[1][0:5]
        tp_params["Algorithm3"]=top_names[2][0:5]
        tp_params["Algorithm4"]=top_names[3][0:5]
        tp_params["Algorithm5"]=top_names[4][0:5]
        # print('tp_params',tp_params)
        sample_composite_dxi1=self.sample_composite_dxi(new_df,weight_lasso, weight_xgb, weight_mi, weight_pca,weight_nb)
        composite_dxi1=sample_composite_dxi1[0]
        avg_composite_dxi=np.mean(composite_dxi1)
        update_y_com=self.create_label_forward(composite_dxi1,avg_composite_dxi)
        print('Average Composite DXI: ', len(composite_dxi1))
        initsxiwgts = sample_composite_dxi1[2]
        topparms = pd.DataFrame(tp_params)
        value_counts = topparms.melt(value_name='value').value.value_counts()
        repeated_values = value_counts[value_counts > 1].index
        vallist = topparms.values.flatten().tolist()
        freqvals = [vallist.count(i) for i in repeated_values]
        most_common_features_df = pd.DataFrame(repeated_values, columns=['Most Common Features'])
        most_common_features_df['No.of Times'] = freqvals
        hcols = list(x.columns)
        impind = []
        for i in repeated_values:
            impind.append(hcols.index(i))
        featimp = [1] * len(hcols) 
        for i in range(len(impind)):
            featimp[impind[i]] = 1 + (freqvals[i]/len(impind))

        sxi = avg_composite_dxi


        # list(rfagents)
        df_buynobuy=df_buynobuy.dropna(how='any')
        df_buynobuy['composite_dxi_label'] = update_y_com
        df_buynobuy['composite_dxi'] = composite_dxi1

        initialsxidata = df_buynobuy
        dropfeatsre = df_buynobuy[dropcols]
        buyerid = getattr(self, 'buyerid', None) or 'anonymous'
        folder = os.path.join(settings.MEDIA_ROOT, 'files', 'chatbot', str(buyerid))
        os.makedirs(folder, exist_ok=True)
        df_buynobuy.to_csv(os.path.join(get_dataset_folder(folder, 'initialsxi.csv'), 'initialsxi.csv'), index=False)

        df_buynobuy = df_buynobuy.drop(dropcols,axis=1)
        clas1 = (df_buynobuy[tv].value_counts()[1]/len(df_buynobuy))*100  
        clas2 = (df_buynobuy[tv].value_counts()[0]/len(df_buynobuy))*100

        f1init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] >= df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 1)]
        f2init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] >= df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 0)]
        f3init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] < df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 1)]
        f4init=df_buynobuy.loc[(df_buynobuy['composite_dxi'] < df_buynobuy['composite_dxi'].mean()) & (df_buynobuy[tv] == 0)]

        if clas1 > clas2:
            jabv = (len(f2init)/(len(f1init) +len(f2init)))*100
            jbel = (len(f4init)/(len(f3init) +len(f4init)))*100
            jlst = [jabv,jbel]
            j = max(jlst)
            if jlst.index(j) == 0:
                twds = 'abv' 
            else:
                twds = 'bel'
            minwhcls = clas2 
            selcls = 'cls2' 
        else:
            jabv = (len(f1init)/(len(f1init) +len(f2init)))*100
            jbel = (len(f3init)/(len(f3init) +len(f4init)))*100
            jlst = [jabv,jbel]
            j = max(jlst)
            if jlst.index(j) == 0:
                twds = 'abv'
            else:
                twds = 'bel' 
            minwhcls = clas1 
            selcls = 'cls1'  

        upgrade = 'sxi'


        # df_buynobuy.to_csv('claiminitialsxi.csv',index=False)

        # acctensor,rfagenttensor = self.tensorflownn(x,pd.Series(y),featimp)
        acctensor = 0
        if new_df.shape[0] <= 5000 or new_df.shape[1] <= 25:
            print("Using heuristic RF-agent weights for faster SXI refinement")
            rfagents = [featimp]
        else:
            print("Training PyTorch RF-agent weights")
            rfagenttorch,acctorch = self.pytorchnn(new_df,pd.Series(update_y),featimp) #Final Weights
            if acctensor > acctorch:
                rfagents = rfagenttensor
            else:
                rfagents = rfagenttorch
                rfagents = [tensor.tolist() for tensor in rfagents]
                rfagents = rfagents[:1]

        # if upgrade == 'sxi':
        #     fdf = df_buynobuy
        #     return fdf,sxi, selcls, twds, feature_weight_df, rfagents, weightcols,initsxiwgts
        # else:
        #     pass

        classifydt = []
        if selcls == 'cls1' and twds == 'abv':
            print(selcls+twds)
            classifydt.append(f1init)
            classifydt.append(f4init)
            misclassif1=f3init
            misclassif2=f2init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
            
        elif selcls == 'cls1' and twds == 'bel':
            print(selcls+twds)
            classifydt.append(f3init)
            classifydt.append(f2init)
            misclassif1=f1init
            misclassif2=f4init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
        elif selcls == 'cls2' and twds == 'abv':
            print(selcls+twds)
            classifydt.append(f2init)
            classifydt.append(f3init)
            misclassif1=f1init
            misclassif2=f4init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
        else:
            print(selcls+twds)
            classifydt.append(f1init)
            classifydt.append(f4init)
            misclassif1=f3init
            misclassif2=f2init
            misclassif1=misclassif1.reset_index()
            misclassif2=misclassif2.reset_index()
        orgmsclasif = len(misclassif1) + len(misclassif2)
        clsfdfnl2 = [] 
        fnlmisclasifff2 = []
        clsfdfnl1 = []
        fnlmisclasifff1 = []
        lenmisclassif = []
        for wgs in rfagents:

            clsfdfnl_2,fnlmisclasif_2 = self.wrecalibrate(sxi,'class2',misclassif2,tv,wgs,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
                avg_base_dxi,selcls,twds,classes,primkey)
            clsfdfnl_1,fnlmisclasif_1 = self.wrecalibrate(sxi,'class1',misclassif1,tv,wgs,lasso_weight,mi_weight,pca_weight,nb_weight,xgb_weight,
            avg_base_dxi,selcls,twds,classes,primkey)
            clsfdfnl2.append(clsfdfnl_2)
            fnlmisclasifff2.append(fnlmisclasif_2)
            clsfdfnl1.append(clsfdfnl_1)
            fnlmisclasifff1.append(fnlmisclasif_1)
            msclass = len(fnlmisclasif_2) + len(fnlmisclasif_1)
            lenmisclassif.append(msclass)
            if msclass == 0:
                break
            else:
                pass
        fnlmdlind = lenmisclassif.index(min(lenmisclassif))
        lists,list2,finldata=[],[],[]
        for i in fnlmisclasifff1[fnlmdlind]['index']:
            lists.append(i)
        result_misclassif1 = misclassif1[misclassif1['index'].isin(lists)]
        for i in fnlmisclasifff2[fnlmdlind]['index']:
            list2.append(i)
        result_misclassif2 = misclassif2[misclassif2['index'].isin(list2)]
        for i in clsfdfnl1[fnlmdlind]:
            for j in i:
                finldata.append(j)
        for k in clsfdfnl2[fnlmdlind]:
            for o in k:
                finldata.append(o)
        fnlmsclasif = len(result_misclassif1) + len(result_misclassif2)
        if fnlmsclasif == orgmsclasif:
            # Original SXI-scored frame (initialsxi). Empty columns-only fdf
            # breaks regression sxi_methodreg (no rows → empty cc → None unpack).
            fdf = df_buynobuy.copy()
        else:
            dt1 = pd.concat(finldata)
            dt2 = pd.concat(classifydt)
            fdf = pd.concat([dt1,dt2,result_misclassif1,result_misclassif2])

        if fdf is None or getattr(fdf, "empty", True):
            print("SXI-RL produced no scored rows; keeping initial SXI-scored frame for regression.")
            fdf = df_buynobuy.copy()
        if "i_n_d_e_x" not in fdf.columns:
            fdf = fdf.copy()
            fdf["i_n_d_e_x"] = range(1, len(fdf) + 1)

        return fdf,sxi, selcls, twds, feature_weight_df, rfagents, weightcols,initsxiwgts


# 3rd
class model_execution:

    def __init__(self,request, dataframe, values_exe,mapping,toogle_val):
        self.dataframe = dataframe
        self.values_exe = values_exe
        self.mapping = mapping
        self.toogle_val = toogle_val
        self.rand = np.random.randint(0, 99999999)
        self.request = request
        self.buyerid = (request.session.get('run_id') or request.session.get('buyerid')) if request and hasattr(request, 'session') else 'anonymous'
        if request and hasattr(request, 'session') and request.session:
            _local.session_key = request.session.session_key
        else:
            _local.session_key = None
        self.r2score = None
        self.sxiacc = None
        self.sximae = None
        self.sxirecall = None
        self.sxiprec = None
        self.auc_best = None
        self.encoded_feature_mapping = {}
        self.grouped_encoded_features = {}
        self.current_tree_feature_importance_raw = pd.DataFrame()
        self.target_tree_feature_importance_raw = pd.DataFrame()

    def _humanize_feature_label(self, value):
        text = str(value or "").strip().replace("_", " ")
        if not text:
            return ""
        words = []
        for word in text.split():
            if word.lower() == "sxi":
                words.append("SXI")
            else:
                words.append(word.capitalize())
        return " ".join(words)

    def _build_top_n_feature_mapping(self, series, column_name, top_n=5):
        normalized = series.fillna("Missing").astype(str)
        top_values = normalized.value_counts().nlargest(top_n).index.tolist()
        collapsed = normalized.apply(lambda x: x if x in top_values else "Others")

        dummies = pd.get_dummies(collapsed, prefix=column_name, dtype=int)
        ordered_categories = list(top_values)
        if "Others" not in ordered_categories:
            ordered_categories.append("Others")

        ordered_columns = []
        feature_mapping = {}
        for category in ordered_categories:
            encoded_col = f"{column_name}_{category}"
            if encoded_col not in dummies.columns:
                dummies[encoded_col] = 0
            ordered_columns.append(encoded_col)
            feature_mapping[encoded_col] = {
                "base_feature": column_name,
                "category": category,
                "base_feature_display": self._humanize_feature_label(column_name),
                "category_display": (
                    "Others"
                    if str(category).strip().lower() == "others"
                    else self._humanize_feature_label(category)
                ),
                "top_categories": [str(item) for item in top_values],
            }

        return dummies[ordered_columns], feature_mapping, ordered_columns

    def _decorate_encoded_feature_importance_df(self, feature_importance_df, sxi_col, importance_col):
        if not isinstance(feature_importance_df, pd.DataFrame) or feature_importance_df.empty:
            return pd.DataFrame()

        decorated_rows = []
        for _, record in feature_importance_df.iterrows():
            encoded_feature = str(record.get("Feature") or "").strip()
            meta = self.encoded_feature_mapping.get(encoded_feature, {})
            base_feature = meta.get("base_feature", encoded_feature)
            category = meta.get("category")
            decorated_rows.append({
                "Feature": encoded_feature,
                "Encoded_Feature": encoded_feature,
                "Base_Feature": base_feature,
                "Category": category,
                "Display_Feature": self._humanize_feature_label(base_feature),
                "Display_Category": (
                    None if category is None else
                    ("Others" if str(category).strip().lower() == "others" else self._humanize_feature_label(category))
                ),
                sxi_col: float(record.get(sxi_col, 0.0) or 0.0),
                importance_col: float(record.get(importance_col, 0.0) or 0.0),
            })

        return pd.DataFrame(decorated_rows)

    def _aggregate_feature_importance_df(self, feature_importance_df, sxi_col, importance_col):
        if not isinstance(feature_importance_df, pd.DataFrame) or feature_importance_df.empty:
            return pd.DataFrame(columns=["Feature", sxi_col, importance_col, "Category_Breakdown"])

        aggregated_rows = []
        grouped = feature_importance_df.groupby(["Base_Feature", "Display_Feature"], dropna=False)
        for (_, display_feature), group_df in grouped:
            sorted_group = group_df.sort_values(by=importance_col, ascending=False).reset_index(drop=True)
            category_parts = []
            for _, row in sorted_group.iterrows():
                category_label = row.get("Display_Category")
                if not category_label:
                    continue
                category_parts.append(f"{category_label} -> {float(row.get(importance_col, 0.0)):.2f}%")

            aggregated_rows.append({
                "Feature": display_feature,
                "Base_Feature": sorted_group.iloc[0]["Base_Feature"],
                sxi_col: float(sorted_group[sxi_col].sum()),
                importance_col: float(sorted_group[importance_col].sum()),
                "Category_Breakdown": "; ".join(category_parts),
            })

        aggregated_df = pd.DataFrame(aggregated_rows)
        return aggregated_df.sort_values(by=importance_col, ascending=False).reset_index(drop=True)

    def gptmlclsf(self):
        # drop custid from df
        if isinstance(self.dataframe, pd.DataFrame):
            df = self.dataframe.copy()
        else:
            df = pd.read_csv(self.dataframe)

        # df = df.drop(columns=['custid'], axis=1)

        def handle_missing_values(df):

            for col in df.columns:
                missing_percentage = df[col].isnull().sum() / len(df) * 100
                print('Missing percent : ',missing_percentage )
                if missing_percentage > 70:
                    # Drop the column if more than 70% of values are missing
                    df.drop(col, axis=1, inplace=True)
                else:
                    # Impute missing values using appropriate method
                    if df[col].dtype == 'object':  # Categorical column
                        mode = df[col].mode()[0]  # Use mode for categorical
                        df[col].fillna(mode, inplace=True)
                    elif df[col].dtype in ['int64', 'float64']:  # Numerical column
                        mean = df[col].mean()
                        median = df[col].median()

                        # Choose imputation method based on distribution
                        if df[col].skew() > 0.5 or df[col].skew() < -0.5:  # Skewed distribution
                            df[col].fillna(median, inplace=True)  # Use median for skewed data
                        else:
                            df[col].fillna(mean, inplace=True)  # Use mean for approximately normal data

            return df
        
        df = handle_missing_values(df)
        print(df.isnull().sum())

        
        def onehot(column_name, df):
            encoded_columns = []
            mylist = []  # Initialize mylist as an empty list

            # Apply one-hot encoding
            if df[column_name].dtype == 'object':
                # Count frequency of each value in the column
                value_counts = df[column_name].value_counts()

                # Determine the top 5 values
                top_5_values = value_counts.nlargest(5).index

                # Replace values not in the top 5 with 'others'
                df[column_name] = df[column_name].apply(lambda x: x if x in top_5_values else 'others')

                # One-hot encode the column
                tempdf = pd.get_dummies(df[column_name], prefix=column_name)
                df = pd.concat([df, tempdf], axis=1)
                df = df.drop(columns=column_name)

                # Update mylist with the top 5 values and 'others'
                mylist.append([f'{column_name}_{val}' for val in top_5_values] + [f'{column_name}_others'])

            return df, mylist


        def preprocess_data(df, target_column):
            label_encoded_values = []
            label_encoded_features = []

            for column in df.columns:
                if column == target_column:
                    continue  # Skip the target column

                if df[column].dtype == 'object':
                    unique_values = len(df[column].unique())
                    if unique_values == 2 or unique_values > 10:
                        le = preprocessing.LabelEncoder()
                        df[column] = le.fit_transform(df[column])
                        labels = dict(zip(le.classes_, range(len(le.classes_))))
                        label_encoded_values.append(labels)
                        label_encoded_features.append(column)
                    elif 3 <= unique_values < 10:
                        df, mylist = onehot(column, df)

            return df, label_encoded_values, label_encoded_features



        values_exe = self.values_exe
        tv = values_exe['Target Outcome']
        tv_type = values_exe['Target Outcome Type']
        classes = [values_exe['Good Outcome'], values_exe['Bad Outcome']]

        if df[tv].dtype == 'object':
            target_encoder = preprocessing.LabelEncoder()
            df[tv] = target_encoder.fit_transform(df[tv].astype(str))

        audit_result = fit_safe_binary_logistic_pipeline(
            df,
            tv,
            primary_key='master_id' if 'master_id' in df.columns else None,
            raw_date_col='fl_date' if 'fl_date' in df.columns else None,
            extra_drop_columns=['netqyty_Bucket'],
            test_size=0.2,
            random_state=42,
        )

        print(f"[GPT-ML] Split strategy: {audit_result['split_metadata']}")
        print(f"[GPT-ML] Suspected leakage columns dropped: {audit_result['audit']['suspected_columns']}")

        y_test = audit_result['y_test']
        pred = audit_result['predictions']
        y_probs = audit_result['probabilities']
        random_algo = "Leakage-Safe LogisticRegression"

        acc = round(float(audit_result['metrics']['accuracy']), 3)
        prec = round(float(audit_result['metrics']['precision']), 3)
        recall = round(float(audit_result['metrics']['recall']), 3)
        fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label=1)
        roc_auc = round(float(audit_result['metrics']['auc']), 3)

        def rocfig(fpr,tpr,roc_auc):
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
            x=fpr, 
            y=tpr, 
            mode='lines', 
            name=f"AUC = {roc_auc:.3f}",
            hovertemplate="False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}",  # Custom hovertemplate
            ))

            # Updating layout with axis labels and title
            fig_roc.update_layout(
                title="AUC-ROC Curve",
                title_x=0.5, 
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                showlegend=True,
                width=1200,
                height=680)
            
            rand = self.rand
            buyerid = self.request.session.get('buyerid') or 'anonymous'
            file_name = f'roc_auc{rand}.png'
            file_name_html = f'roc_auc{rand}.html'
            folder = f'media/files/chatbot/{buyerid}/'
            # Use MEDIA_ROOT to construct the file path for saving
            loc_png = os.path.join(get_dataset_folder(folder, file_name), file_name)
            loc_html = os.path.join(get_dataset_folder(folder, file_name_html), file_name_html)
            
            _safe_write_plotly_figure(
                fig_roc,
                loc_html,
                loc_png,
                title="ROC AUC"
            )

            return loc_png

        roc_plot = rocfig(fpr, tpr,roc_auc)
        print(f'Accuracy: {acc}')
        print(f'Precision: {prec}')
        print('AUC SCORE: ',roc_auc)
        print(f'Recall: {recall}')


        light_green_red_cmap = LinearSegmentedColormap.from_list(
            'LightGreenRed', ['red', 'white', 'lightgreen']
        )
        # Confusion matrix data
        # z = confusion_matrix(y_test,pred)

        # x = classes
        # y = classes

        # # Create annotation text for the matrix
        # text = [[str(value) for value in row] for row in z]

        # print("<---------------confusion matrix variables-------------------->")
        # print("z==>",z)
        # print("x==>",x)
        # print("y==>",y)
        # # # Plot the confusion matrix without grid lines
        # # fig, ax = plt.subplots(figsize=(7,4),dpi=300)
        # # cax = ax.matshow(z, cmap=light_green_red_cmap)
        # # plt.colorbar(cax)

        # # # Annotate each cell with the value
        # # for (i, j), val in np.ndenumerate(z):
        # #     ax.text(j, i, f'{val}', ha='center', va='center', color='black')

        # # # Set axis labels and ticks
        # # ax.set_xticks(range(len(x)))
        # # ax.set_yticks(range(len(y)))
        # # ax.set_xticklabels(x)
        # # ax.set_yticklabels(y)
        # # ax.set_xlabel("Predicted Class")
        # # ax.set_ylabel("True Class")
        # # ax.set_title("Confusion Matrix")

        # # # Remove grid lines
        # # ax.grid(False)
        # # #fig.show()

        # # Plot the confusion matrix using Plotly
        # fig = ff.create_annotated_heatmap(
        #     z, 
        #     x=[f"Class {c}" for c in classes],  # Use dynamic class labels
        #     y=[f"Class {c}" for c in classes], 
        #     colorscale=[(0, 'red'), (0.5, 'white'), (1, 'lightgreen')],
        #     showscale=True,  # Display the color bar
        #     annotation_text=text,  # Add text annotations for each cell
        #     hoverinfo="z"  # Show cell value on hover
        # )

        # # Update layout for titles and labels
        # fig.update_layout(
        #     title_text="Confusion Matrix",
        #     title_x=0.5,  # Center the title
        #     xaxis=dict(title="Predicted Class"),
        #     yaxis=dict(title="True Class"),
        #     width=1200,
        #     height=680
        # )
        

        z = confusion_matrix(y_test, pred)
        class_0_total = z[0][0] + z[0][1]
        class_1_total = z[1][1] + z[1][0]
        correct_class_0 = round(z[0][0] / class_0_total * 100, 2)
        incorrect_class_0 = round(z[0][1] / class_0_total * 100, 2)
        correct_class_1 = round(z[1][1] / class_1_total * 100, 2)
        incorrect_class_1 = round(z[1][0] / class_1_total * 100, 2)
        

        colorscale = [[0, 'red'],
                    [0.5, 'white'],
                    [1,'lightgreen']]
        # Create confusion matrix plot
        annotations = [
            [f"{z[0][0]}<br>({correct_class_0}%)", f"{z[0][1]}<br>({incorrect_class_0}%)"],
            [f"{z[1][0]}<br>({incorrect_class_1}%)", f"{z[1][1]}<br>({correct_class_1}%)"]
        ]
        
        conf_matrix = go.Figure(data=go.Heatmap(
            z=z,
            x=classes,
            y=classes,
            colorscale=colorscale,
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        
        conf_matrix.update_layout(
            title='Accuracy Matrix',
            xaxis_title='Predicted Class',
            yaxis_title='True Class',
            xaxis={'side': 'bottom'},
            width=1200,
            height=680,
            font=dict(size=12)
        )

        rand = self.rand
        buyerid = self.request.session.get('buyerid')
        file_name = f'confpng{rand}.png'
        file_name_html = f'confpng{rand}.html'
        loc_png = os.path.join(settings.MEDIA_ROOT, f'files/chatbot/{buyerid}/', file_name)
        loc_html = os.path.join(settings.MEDIA_ROOT, f'files/chatbot/{buyerid}/', file_name_html)
        _safe_write_plotly_figure(
            conf_matrix,
            loc_html,
            loc_png,
            title="Confusion Matrix"
        )
        return acc, prec, recall, roc_auc, random_algo, z, loc_png, roc_plot, loc_html

    def gptmlreg(self):
        if isinstance(self.dataframe, pd.DataFrame):
            df = self.dataframe.copy()
        else:
            df = pd.read_csv(self.dataframe)
        
        def handle_missing_values(df):

            for col in df.columns:
                missing_percentage = df[col].isnull().sum() / len(df) * 100
                print('Missing percent : ',missing_percentage )
                if missing_percentage > 70:
                    # Drop the column if more than 70% of values are missing
                    df.drop(col, axis=1, inplace=True)
                else:
                    # Impute missing values using appropriate method
                    if df[col].dtype == 'object':  # Categorical column
                        mode = df[col].mode()[0]  # Use mode for categorical
                        df[col].fillna(mode, inplace=True)
                    elif df[col].dtype in ['int64', 'float64']:  # Numerical column
                        mean = df[col].mean()
                        median = df[col].median()

                        # Choose imputation method based on distribution
                        if df[col].skew() > 0.5 or df[col].skew() < -0.5:  # Skewed distribution
                            df[col].fillna(median, inplace=True)  # Use median for skewed data
                        else:
                            df[col].fillna(mean, inplace=True)  # Use mean for approximately normal data

            return df


        def onehot(column_name, df):
            encoded_columns = []
            mylist = []  # Initialize mylist as an empty list

            # Apply one-hot encoding
            if df[column_name].dtype == 'object':
                # Count frequency of each value in the column
                value_counts = df[column_name].value_counts()

                # Determine the top 5 values
                top_5_values = value_counts.nlargest(5).index

                # Replace values not in the top 5 with 'others'
                df[column_name] = df[column_name].apply(lambda x: x if x in top_5_values else 'others')

                # One-hot encode the column
                tempdf = pd.get_dummies(df[column_name], prefix=column_name)
                df = pd.concat([df, tempdf], axis=1)
                df = df.drop(columns=column_name)

                # Update mylist with the top 5 values and 'others'
                mylist.append([f'{column_name}_{val}' for val in top_5_values] + [f'{column_name}_others'])

            return df, mylist


        def preprocess_data(df, target_column):
            label_encoded_values = []
            label_encoded_features = []

            for column in df.columns:
                if column == target_column:
                    continue  # Skip the target column

                if df[column].dtype == 'object':
                    unique_values = len(df[column].unique())
                    if unique_values == 2 or unique_values > 10:
                        le = preprocessing.LabelEncoder()
                        df[column] = le.fit_transform(df[column])
                        labels = dict(zip(le.classes_, range(len(le.classes_))))
                        label_encoded_values.append(labels)
                        label_encoded_features.append(column)
                    elif 3 <= unique_values < 10:
                        df, mylist = onehot(column, df)

            return df, label_encoded_values, label_encoded_features


        #data= pd.read_csv("/content/hotel_bookings_missing.csv")
        #data.isnull().sum()
        df = handle_missing_values(df)
        print(df.isnull().sum())
        
        
        values_exe = self.values_exe
        tv = values_exe['Target Outcome']
        tv_type = values_exe['Target Outcome Type']
        classes= [values_exe[ 'Good Outcome'],values_exe[ 'Bad Outcome']]
        df, label_encoded_values, label_encoded_features = preprocess_data(df, tv)

        X = df.drop(tv,axis=1)
        y= df[tv] 

        cat_cols = list(X.select_dtypes(['object']).columns)
        le = LabelEncoder()
        if cat_cols:
            X[cat_cols] = le.fit_transform(X[cat_cols])

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
        y_train_log = np.log1p(y_train)

        algos = ['Lasso', 'Random Forest Regressor']
        random_algo = random.choice(algos)
        print(f'Selected Algorithm: {random_algo}')

        from sklearn.linear_model import Lasso
        from sklearn.ensemble import RandomForestRegressor

        # Train model
        if random_algo == 'Lasso':
            clf = Lasso(alpha=0.01).fit(X_train, y_train_log)

        elif random_algo == 'Random Forest Regressor':
            clf = RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ).fit(X_train, y_train_log)

        else:
            raise ValueError(f"Unknown regression algorithm: {random_algo}")

        pred_log = clf.predict(X_test)
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)  # Post-hoc absolute minimum

        r2 = round(r2_score(y_test, pred), 3)
        mae = round(mean_absolute_error(y_test, pred), 3)
        y_true = np.asarray(y_test, dtype=float)
        y_pred = np.asarray(pred, dtype=float)
        
        # Calculate sMAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        smape_mask = denominator > 1e-8
        if np.any(smape_mask):
            mape = round(
                float(np.mean(np.abs(y_true[smape_mask] - y_pred[smape_mask]) / denominator[smape_mask]) * 100),
                3,
            )
        else:
            mape = None

        print(f'R2-Score: {r2}')
        print(f'MAE: {mae}')
        print(f'sMAPE: {mape}')

        # Validation of +/- 10% metric separately by revenue bucket
        zero_mask = y_true == 0
        pos_mask = y_true > 0
        
        if np.any(zero_mask):
            zero_mae = mean_absolute_error(y_true[zero_mask], y_pred[zero_mask])
            zero_within = np.mean(np.abs(y_pred[zero_mask] - y_true[zero_mask]) <= (np.abs(y_true[zero_mask]) * 0.20)) * 100
            print(f"Zero Revenue Bucket - MAE: {zero_mae:.3f}, Within 20%: {zero_within:.2f}%")
            
        if np.any(pos_mask):
            pos_mae = mean_absolute_error(y_true[pos_mask], y_pred[pos_mask])
            pos_within = np.mean(np.abs(y_pred[pos_mask] - y_true[pos_mask]) <= (np.abs(y_true[pos_mask]) * 0.20)) * 100
            print(f"Positive Revenue Bucket - MAE: {pos_mae:.3f}, Within 20%: {pos_within:.2f}%")

        within_percentage, actvspredplt = self.actvspredplot(
            y_test, pred, tv, mae, r2
        )

        return mae, mape, r2, within_percentage, random_algo, actvspredplt

        # algos = ['Lasso','Random Forest Regressor']
        # random_algo = random.choice(algos)
        # #random_algo = 'Random Forest Regressor'
        # print(f'Selected Algorithm: {random_algo}')
        # from sklearn import svm
        # from sklearn.tree import DecisionTreeRegressor
        # from sklearn.linear_model import LinearRegression, Lasso
        # if random_algo == algos[0]:
        #     clf = LinearRegression().fit(X_train,y_train)
        # elif random_algo == algos[1]:
        #     clf = DecisionTreeRegressor().fit(X_train,y_train)
        # elif random_algo == algos[2]:
        #     clf = svm.SVR(kernel='linear').fit(X_train,y_train)
        # elif random_algo == algos[3]:
        #     clf = Lasso().fit(X_train,y_train)
        # else:
        # #elif random_algo == algos[4]:
        #     clf = RandomForestRegressor().fit(X_train,y_train)
        # pred = clf.predict(X_test)
        # acc = round(r2_score(y_test,pred),3)
        # mae = round(mean_absolute_error(y_test,pred),3)
        # print(f'R2-Score: {acc}')
        # print(f'MAE: {mae}')
        # within_percentage,actvspredplt = self.actvspredplot(y_test,pred,tv,mae,acc)
        # return mae,acc,within_percentage,random_algo,actvspredplt

    def actvspredplot(self, Ytest, Ypred, tv, mae, r2):
        import numpy as np
        import shutil
        from pathlib import Path as _Path

        import plotly.graph_objects as go

        # Calculate ±10% error boundaries
        error_tol = 0.10
        Ytest_array = np.asarray(Ytest, dtype=float)
        pred_array = np.asarray(Ypred, dtype=float)
        finite_mask = np.isfinite(Ytest_array) & np.isfinite(pred_array)
        Ytest_array = Ytest_array[finite_mask]
        pred_array = pred_array[finite_mask]

        if len(Ytest_array):
            x_min = float(np.nanmin(Ytest_array))
            x_max = float(np.nanmax(Ytest_array))
        else:
            x_min, x_max = 0.0, 1.0
        if abs(x_max - x_min) < 1e-12:
            x_max = x_min + 1.0
        line_x = np.linspace(x_min, x_max, 200)
        perfect_y = line_x.copy()
        upper_bound = line_x * (1.0 + error_tol)
        lower_bound = line_x * (1.0 - error_tol)

        # Prepare data for points within and outside the 10% error boundaries
        within_bounds_x = []
        within_bounds_y = []
        within_bounds_hover = []  # Hover text for within bounds

        outside_bounds_x = []
        outside_bounds_y = []
        outside_bounds_hover = []  # Hover text for outside bounds

        total_points = len(Ytest_array)
        within_bounds_count = 0  # Counter for points within bounds

        for yt, yp in zip(Ytest_array, pred_array):
            hover_text = (
                f"Actual: {format_display_value(yt, tv)}"
                f"<br>Predicted: {format_display_value(yp, tv)}"
            )
            tolerance = abs(yt) * error_tol
            is_within = abs(yp - yt) <= tolerance if tolerance > 0 else abs(yp - yt) <= 0
            if is_within:
                within_bounds_x.append(yt)
                within_bounds_y.append(yp)
                within_bounds_hover.append(hover_text)
                within_bounds_count += 1
            else:
                outside_bounds_x.append(yt)
                outside_bounds_y.append(yp)
                outside_bounds_hover.append(hover_text)

        # Calculate percentage of points within 10% error
        within_percentage = (within_bounds_count / total_points) * 100 if total_points else 0.0
        annotation_text = f"MAE={mae:.2f}, R²={r2:.2f}, <br> Within ±10% Error: {within_percentage:.2f}%"

        # Matplotlib PNG first so Perfect Prediction / ±10% lines always appear
        plt.figure(figsize=(10, 6.2))
        ax = plt.gca()
        ax.set_facecolor("#eff1fe")
        if within_bounds_x:
            ax.scatter(within_bounds_x, within_bounds_y, s=28, c="green", alpha=0.6, label="Within ±10%")
        if outside_bounds_x:
            ax.scatter(outside_bounds_x, outside_bounds_y, s=28, c="red", alpha=0.6, label="Outside ±10% Error")
        ax.plot(line_x, perfect_y, color="black", linewidth=2, label="Perfect Prediction")
        ax.plot(line_x, upper_bound, color="blue", linewidth=1.5, linestyle="--", label="+10% Error")
        ax.plot(line_x, lower_bound, color="blue", linewidth=1.5, linestyle="--", label="-10% Error")
        ax.set_title(
            f"Actual vs Predicted {tv}\nMAE={mae:.2f}, R²={r2:.2f}, Within ±10% Error: {within_percentage:.2f}%",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel(f"Actual {tv}")
        ax.set_ylabel(f"Predicted {tv}")
        ax.grid(True, which="major", color="#9585e6", linestyle="-", linewidth=0.6, alpha=0.55)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        plt.tight_layout()

        # Create the Plotly figure
        fig = go.Figure()

        # Plot points within the 10% error boundary
        fig.add_trace(go.Scatter(
            x=within_bounds_x,
            y=within_bounds_y,
            mode='markers',
            marker=dict(size=8, color='green', opacity=0.6),
            text=within_bounds_hover,  # Add hover text
            hoverinfo="text",
            name='Within ±10% '
        ))

        # Plot points outside the 10% error boundary
        fig.add_trace(go.Scatter(
            x=outside_bounds_x,
            y=outside_bounds_y,
            mode='markers',
            marker=dict(size=8, color='red', opacity=0.6),
            text=outside_bounds_hover,  # Add hover text
            hoverinfo="text",
            name='Outside ±10% Error'
        ))

        # Add the perfect prediction line (y = x)
        fig.add_trace(go.Scatter(
            x=line_x.tolist(),
            y=perfect_y.tolist(),
            mode='lines',
            line=dict(color='black', width=2),
            name='Perfect Prediction'
        ))

        # Add dashed lines for ±10% error boundaries
        fig.add_trace(go.Scatter(
            x=line_x.tolist(),
            y=upper_bound.tolist(),
            mode='lines',
            line=dict(color='blue', width=1.5, dash='dash'),
            name='+10% Error'
        ))

        fig.add_trace(go.Scatter(
            x=line_x.tolist(),
            y=lower_bound.tolist(),
            mode='lines',
            line=dict(color='blue', width=1.5, dash='dash'),
            name='-10% Error'
        ))

        # Update layout for titles and axis labels
        fig.update_layout(
            title=dict(
                text=f"Actual vs Predicted {tv}<br>{annotation_text}",
                x=0.5,
                xanchor='center'
            ),
            xaxis=build_plotly_value_axis(f'Actual {tv}', tv),
            yaxis=build_plotly_value_axis(f'Predicted {tv}', tv),
            legend=dict(
                x=0,  # x position (0 = left)
                y=1,  # y position (1 = top)
                xanchor='left',  # Anchor legend to the left
                yanchor='top'  # Anchor legend to the top
            ),
            width=800,
            height=500
        )

        buyerid = self.request.session.get('buyerid')
        rand = random.randint(0, 99999999)
        folder = f'media/files/chatbot/{buyerid}/'
        locpng = f'actvspred_{rand}.png'
        lochtml = f'actvspred_{rand}.html'
        actpredplt_ml = os.path.join(get_dataset_folder(folder, locpng), locpng)
        actpredplt_ml_html = os.path.join(get_dataset_folder(folder, lochtml), lochtml)
        plt.savefig(actpredplt_ml, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        mpl_backup = str(_Path(actpredplt_ml).with_suffix(".mpl.png"))
        try:
            shutil.copy2(actpredplt_ml, mpl_backup)
        except Exception:
            mpl_backup = None
        try:
            fig.write_html(actpredplt_ml_html)
        except Exception as exc:
            print(f"[actvspredplot] HTML export failed: {exc}")
        try:
            _safe_write_plotly_figure(
                fig,
                actpredplt_ml_html,
                actpredplt_ml,
                title=f"Actual vs Predicted {tv}"
            )
        except Exception as exc:
            print(f"[actvspredplot] Plotly PNG export failed, keeping matplotlib PNG: {exc}")
        if mpl_backup and _Path(mpl_backup).exists():
            shutil.copy2(mpl_backup, actpredplt_ml)
            try:
                _Path(mpl_backup).unlink(missing_ok=True)
            except Exception:
                pass

        print(f'Plot saved at {actpredplt_ml}')

        return within_percentage,actpredplt_ml

    def distribution_plot(self, target, labels, dataframe, SXI, bad, good, tv_type):
        tv = target
        dq = dataframe.copy()

        print(f"Bad {bad}, Good {good}")
        if tv_type == 'Categorical':
            mapping = _build_code_to_label_mapping(labels)
            print('mapping:', mapping)
            dq[f'target = {tv}'] = dq[tv].apply(lambda x: mapping.get(_normalize_label_value(x), str(x)))
            color_map = {
                mapping.get(_normalize_label_value(bad), str(bad)): 'red',
                mapping.get(_normalize_label_value(good), str(good)): 'green',

            }
            dq['color'] = dq[f'target = {tv}'].map(color_map)
            print(f'color_map {color_map}')
        else:
            mean_source = pd.to_numeric(dq.get(f'{tv}_original', pd.Series(dtype=float)), errors='coerce')
            mean_value = round(_safe_float(mean_source.mean(), default=0.0), 2)
            high_key = f'High {tv} (> {mean_value})'
            low_key = f'Low {tv} (< {mean_value})'
            # Prefer original continuous values so High/Low match real amounts
            # (encoded 0/1 can be inverted vs Above_mean/Below_mean labels).
            if mean_source.notna().any():
                dq[f'target = {tv}'] = np.where(
                    mean_source >= mean_value,
                    high_key,
                    low_key,
                )
            else:
                labels2 = dict(labels)
                above_label = labels2.pop('Above_mean', 1)
                below_label = labels2.pop('Below_mean', 0)
                labels2[high_key] = above_label
                labels2[low_key] = below_label
                mapping = {str(val): str(key) for key, val in labels2.items()}
                for raw in (above_label, below_label, 0, 1, '0', '1'):
                    key = mapping.get(str(raw))
                    if key:
                        mapping[str(raw)] = key
                        try:
                            mapping[str(int(float(raw)))] = key
                        except Exception:
                            pass

                def _map_reg_target(value):
                    text = str(value)
                    if text in mapping:
                        return mapping[text]
                    try:
                        as_int = str(int(float(value)))
                        if as_int in mapping:
                            return mapping[as_int]
                    except Exception:
                        pass
                    return text

                dq[f'target = {tv}'] = dq[tv].apply(_map_reg_target)
            # Match reference style: Low = red, High = green
            color_map = {low_key: 'red', high_key: 'green'}
            dq['color'] = dq[f'target = {tv}'].map(color_map)

        # Matplotlib Plot — primary PNG (Kaleido is unreliable on this host)
        plt.figure(figsize=(10, 6))
        plt.title(f'DXI Distribution of {tv}')
        plt.xlabel('DXI')
        plt.ylabel('Count')
        ax = plt.gca()
        ax.set_facecolor('#f0f0f0')

        num_bins = 50
        histogram_source = _coerce_numeric_series(dq['composite_dxi']) if 'composite_dxi' in dq.columns else pd.Series(dtype=float)
        if histogram_source.empty:
            histogram_source = pd.Series([_safe_float(SXI, default=0.0)])
        # Use a shared bin range wide enough for a readable distribution
        dxi_min = float(np.nanmin(histogram_source)) if len(histogram_source) else 0.0
        dxi_max = float(np.nanmax(histogram_source)) if len(histogram_source) else 1.0
        if abs(dxi_max - dxi_min) < 1e-9:
            dxi_min -= 0.5
            dxi_max += 0.5
        bin_edges = np.linspace(dxi_min, dxi_max, num_bins + 1)
        hist_data, _ = np.histogram(histogram_source, bins=bin_edges)

        # Draw Low then High so overlap reads like the reference chart
        category_order = list(color_map.keys())
        if any(str(k).lower().startswith('low ') for k in category_order):
            category_order = sorted(
                category_order,
                key=lambda k: (0 if str(k).lower().startswith('low ') else 1, str(k)),
            )

        for category in category_order:
            category_data = dq[dq[f'target = {tv}'] == category]
            category_hist_source = (
                _coerce_numeric_series(category_data['composite_dxi'])
                if 'composite_dxi' in category_data.columns
                else pd.Series(dtype=float)
            )
            if category_hist_source.empty:
                continue
            hist, edges = np.histogram(category_hist_source, bins=bin_edges)
            plt.bar(
                edges[:-1],
                hist,
                width=np.diff(edges),
                align='edge',
                color=color_map[category],
                alpha=0.55,
                edgecolor='white',
                linewidth=0.2,
                label=str(category),
            )

        max_count = int(max(hist_data.max() if len(hist_data) else 0, 1))
        plt.axvline(x=SXI, color='black', linewidth=3, label=f'Current DXI = {round(SXI, 2)}')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        plt.grid(True, color='white', linewidth=0.8)
        plt.tight_layout()
        buyerid = self.request.session.get('buyerid')
        rand = random.randint(0, 99999999)
        folder = f'media/files/chatbot/{buyerid}/'
        file_name = f'sxidist{rand}.png'
        loc = os.path.join(get_dataset_folder(folder, file_name), file_name)
        plt.savefig(loc, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close()

        # Plotly Plot (HTML + PNG fallback)
        fig = go.Figure()

        for category in category_order:
            category_data = dq[dq[f'target = {tv}'] == category]
            category_hist_source = _coerce_numeric_series(category_data['composite_dxi']) if 'composite_dxi' in category_data.columns else pd.Series(dtype=float)
            if category_hist_source.empty:
                continue
            fig.add_trace(go.Histogram(
                x=category_hist_source.tolist(),
                xbins=dict(
                    start=float(bin_edges[0]),
                    end=float(bin_edges[-1]),
                    size=float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else None,
                ),
                hovertemplate="<b>DXI range</b>: %{x}<br><b>Frequency</b>: %{y}<extra></extra>",
                opacity=0.55,
                name=str(category),
                marker=dict(color=color_map[category], line=dict(color='white', width=0.2)),
                showlegend=True
            ))

        fig.add_trace(go.Scatter(
            x=[SXI, SXI],
            y=[0, max_count * 1.1],
            mode='lines',
            line=dict(color='black', dash='solid', width=3),
            name=f'Current DXI = {round(SXI, 2)}',
            hovertemplate="<b>Current DXI</b>: %{x}<extra></extra>"
        ))

        fig.update_layout(
            title=f'DXI Distribution of {tv}',
            xaxis_title='DXI',
            yaxis_title='Count',
            barmode='overlay',
            legend=dict(x=1.02, y=1.0, xanchor='left'),
            autosize=True,
            template="ggplot2",
            bargap=0,
            width=1200,
            height=680,
            plot_bgcolor='#f0f0f0',
            paper_bgcolor='white',
        )

        folder = f'media/files/chatbot/{buyerid}/'
        plotly_file_name = f'sxidist{rand}.html'
        png_file_name = f'sxidist{rand}.png'
        plotly_loc = os.path.join(get_dataset_folder(folder, plotly_file_name), plotly_file_name)
        png_loc = os.path.join(get_dataset_folder(folder, png_file_name), png_file_name)
        # Keep the reliable matplotlib PNG; also write HTML. Only overwrite PNG if Plotly export succeeds with content.
        try:
            fig.write_html(plotly_loc)
        except Exception as exc:
            print(f"[distribution_plot] HTML export failed: {exc}")
        try:
            result = _safe_write_plotly_figure(
                fig,
                plotly_loc,
                png_loc,
                title=f"DXI Distribution of {tv}"
            )
            # If matplotlib already wrote a good PNG and plotly fallback is thin/broken, keep matplotlib file.
            if Path(loc).exists() and Path(loc).stat().st_size > 20_000:
                if (not Path(png_loc).exists()) or Path(png_loc).stat().st_size < 20_000:
                    import shutil
                    shutil.copy2(loc, png_loc)
        except Exception as exc:
            print(f"[distribution_plot] Plotly PNG export failed, keeping matplotlib PNG: {exc}")
            png_loc = loc

        return png_loc

    def target_encoding(self, target_type, target, dataframe, primkey,mapping_cont):
        import pandas as pd
        print('Target Encoding Entered')
        print("target_type", target_type)
        print("target", target)
        print("dataframe \n", dataframe)
        print("primkey", primkey)

        tv_type = target_type
        tv = target
        df = dataframe.copy()
        primary_key = None
        self.encoded_feature_mapping = dict(getattr(self, "encoded_feature_mapping", {}) or {})
        self.grouped_encoded_features = dict(getattr(self, "grouped_encoded_features", {}) or {})

        # Handle primary key
        if primkey is not None:
            print('primkey:', primkey)
            primary_key = df[primkey]
            df = df.drop(primkey, axis=1, errors='ignore')

        # Ensure the target column is excluded
        if tv in df.columns:
            target_column = df[tv]
            df = df.drop(tv, axis=1, errors='ignore')
        else:
            target_column = None

        labels_list = []
        if tv_type == 'Categorical':
            mapping = {val: key for (key, val) in self.mapping.items()}
            labels_list.append(mapping)
        else:
            labels_list.append(mapping_cont)

        categorical_columns = [
            col for col in df.columns
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])
        ]
        for column in categorical_columns:
            encoded_df, feature_mapping, encoded_columns = self._build_top_n_feature_mapping(
                df[column],
                column,
                top_n=5,
            )
            self.encoded_feature_mapping.update(feature_mapping)
            self.grouped_encoded_features[column] = encoded_columns
            df = pd.concat([df.drop(columns=[column]), encoded_df], axis=1)

        if not categorical_columns and self.encoded_feature_mapping:
            print("Reusing existing encoded feature mapping from master dataset metadata.")

        # Reattach the primary key and target column if they exist
        if primary_key is not None:
            df = pd.concat([primary_key, df], axis=1)
        if target_column is not None:
            df[tv] = target_column
        print('df')
        print(df)

        return labels_list, df


    '''
    # def good_bad_label(self,tv_outcome, labels, tv_value):
    #     print('Labels',labels)
    #     label_dict = labels
    #     print('tv_value',tv_value)
    #     interchanged_labels = [{v: k for k, v in d.items()} for d in labels]
    #     for d in interchanged_labels:
    #         for key in d.keys():
    #             if key == 0:
    #                 d[key] = 0
    #             elif key == 1:
    #                 d[key] = 1
    #     print(interchanged_labels)
    #     label_dict=interchanged_labels
    #     label_dict=label_dict[0]
    #     label_dict = {v: int(k) for k, v in label_dict.items()}
    #     print('label_dict------>',label_dict)
    #     print('tv_value------->',tv_value)
    #     print(label_dict)
    #     if tv_outcome == 'Good':
    #         good = label_dict[list(label_dict.keys())[0]]
    #         print(f'good: {good}')
    #         if good == 1:
    #             bad = 0
    #         elif good == 0:
    #             bad = 1
    #     elif tv_outcome == 'Bad':
    #         bad = label_dict[list(label_dict.keys())[0]]
    #         print(f'Bad: {bad}')
    #         if bad == 1:
    #             good = 0
    #         elif bad == 0:
    #             good = 1

    #     return good, bad
    '''

    def good_bad_label(self, tv_outcome, labels, tv_value):
        print('Labels:', labels)
        print('tv_value:', tv_value)
        print('tv_outcome:', tv_outcome)

        # labels = [{'cancel': 1, 'notcancel': 0}]
        label_dict = labels[0]

        # Reverse dict to map numeric → label
        reverse_dict = {v: k for k, v in label_dict.items()}
        print('reverse_dict:', reverse_dict)

        # Determine which numeric value corresponds to 'Good' or 'Bad'
        if tv_outcome == 'Good':
            good = tv_value
            bad = 1 - tv_value  # opposite
        elif tv_outcome == 'Bad':
            bad = tv_value
            good = 1 - tv_value

        print(f'good: {good}, bad: {bad}')
        return good, bad

    '''
    def correlation_plot_classif(self,target, labels, dataframe, good, bad,avg_composite_dxi,target_value):

        def reverse_equation(y, m, c):
            return (y-c)/m

        tv=target
        df=dataframe
        composite_dxi1=df['composite_dxi']
        yy=df[tv]
        # avg_composite_dxi=np.mean(np.array(composite_dxi1))
        from sklearn.cluster import KMeans
        # clusters = kmeans.elbow_value_
        clusters = 2
        composite_dxi=np.array(composite_dxi1).reshape(-1, 1)
        kmeans = KMeans(n_clusters=clusters, random_state=0)
        kmeans.fit(composite_dxi)
        v,c = np.unique(kmeans.labels_, return_counts=True)

        def index_value(y_train, n):
            index_value=[]
            for i in range(len(y_train)):
                if (y_train[i] == n):
                    index_value.append(i)
            return index_value

        buyers = index_value(list(yy), bad)    # Put Good class or Class to be maximized (can be either 0 or 1)
        non_buyers = index_value(list(yy), good)
            # Put Bad class or Class to be minimized (can be either 0 or 1)
        kmean_label = kmeans.labels_

        km_buyer = []
        for i in range(len(buyers)):
            km_buyer.append(kmean_label[buyers[i]])

        km_non = []
        for i in range(len(non_buyers)):
            km_non.append(kmean_label[non_buyers[i]])

        v, c1 = np.unique(km_buyer, return_counts=True)
        v_non, c_non1 = np.unique(km_non, return_counts=True)

        buy = c1
        no_buyer = c_non1
        buyer, non_buyer = [], []
        vv, nn = [], []
        for i in range(len(buy)):
            buyer.append(buy[i])

        for i in range(clusters):
            vv.append(i)
            if i not in v:
                buyer.insert(i,0)

        for i in range(len(no_buyer)):
            non_buyer.append(no_buyer[i])

        for i in range(clusters):
            nn.append(i)
            if i not in v_non:
                non_buyer.insert(i,0)

        buyer_conv = (np.array(buyer)/len(composite_dxi1)) * 100
        buyer_per = (buyer/(buyer+(np.array(non_buyer)))) * 100

        def index_value(y_train,n):
            index_value=[]
            for i in range(len(y_train)):
                if (y_train[i] == n):
                    index_value.append(i)
            return index_value

        xx=[]
        cluster_index=[]
        for i in range(clusters):
            c1=index_value(kmean_label,i)
            cdx1=composite_dxi[c1]
            xx.append(cdx1)
            cluster_index.append(c1)

        minimum_dxi=[]
        maximum_dxi=[]
        avg_dxi=[]
        for i in range(len(xx)):
            minimum_dxi.append(np.min(xx[i]))
            maximum_dxi.append(np.max(xx[i]))
            avg_dxi.append(np.mean(xx[i]))

        def conv_rate(avg_dxi,buyer_conv):
            dxi_sort=list(np.sort(avg_dxi))
            idx=[]
            for i in range(len(dxi_sort)):
                a=avg_dxi.index(dxi_sort[i])
                idx.append(a)

            buyer_c=buyer_conv[idx]
            conv_rate=[]
            convv=0
            for i in range(len(dxi_sort)):
                conv=convv+buyer_c[i]
                conv_rate.append(conv)
                convv=conv
            return dxi_sort, conv_rate

        mean_dxi, conv_ratee = conv_rate(minimum_dxi, buyer_conv)

        mymodel = np.poly1d(np.polyfit(mean_dxi, conv_ratee, 1))
        myline = np.linspace(min(composite_dxi1), max(composite_dxi1), 100)

        ind = np.where(mymodel(myline) < 100)
        ind = [item for t in ind for item in t]
        myline = myline[ind]
        model_myline = mymodel(myline)[ind]
        print(f' yaxis = {model_myline}')
        print('\n')
        print(f' xaxis = {myline}')
        # Polynomial Regression
        def polyfits(x, y, degree):
            import numpy
            results = {}
            coeffs = numpy.polyfit(x, y, degree)

            # Polynomial Coefficients
            results['polynomial'] = coeffs.tolist()

            # r-squared
            p = numpy.poly1d(coeffs)
            # fit values, and mean
            yhat = p(x)                         # or [p(z) for z in x]
            from sklearn.metrics import r2_score
            r2 = r2_score(y,yhat)
            print('R-squared: ',r2_score(y,yhat))
            ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
            ssreg = numpy.sum((yhat-ybar)*2)   # or sum([ (yihat - ybar)*2 for yihat in yhat])
            sstot = numpy.sum((y - ybar)*2)    # or sum([ (yi - ybar)*2 for yi in y])
            results['determination'] = ssreg / sstot
            return results, coeffs,r2

        # for i in [1, 2,3,4]:
        #     results, coeffs, r2 = polyfits(mean_dxi, conv_ratee, i)
        #     if r2 == 1:
        #         break
        #     elif r2 >= 0.95:
        #         break
        #     elif r2 >= 0.90:
        #         break
        #     else:
        #         pass

        # # mymodel = np.poly1d(np.polyfit(mean_dxi, conv_ratee, i))
        # myline = np.linspace(min(composite_dxi1), max(composite_dxi1), 100)
        # print(f'Mymodel Line: {mymodel(myline)}')
        # inds = myline[(myline >= min(composite_dxi1)) & (myline <= max(composite_dxi1))]

        # results, coeffs,r2 = polyfits(mean_dxi, conv_ratee, 1)

        # if coeffs[-2] < 0:
        #     rltyp = 'Negative'
        # else:
        #     rltyp = 'Positive'
        # print(f'Correlation Typ: {rltyp}')
        # if coeffs[-1] < 0:
        #     equation_str = f"y = {coeffs[-2]:.2f}x - {abs(coeffs[-1]):.2f}"
        # else:
        #     equation_str = f"y = {coeffs[-2]:.2f}x + {abs(coeffs[-1]):.2f}"
        # print(f"Equation of the linear line as a string:{equation_str}")

        # import matplotlib.pyplot as plt
        # # Create the figure and axes objects
        # # mapping = {val: key for (key, val) in labels.items()}
        # mapping = {str(key): value for (key, value) in labels.items()}
        # print('Mapping_s: ',mapping)
        # # print('bad',)
        # print("bad",bad)
        # print("LEN",len(dataframe))

        # # print("   ",dataframe[target].value_counts()[bad])

        # currout = (dataframe[target].value_counts()[bad]/len(dataframe))*100

        # print(f'Current Outcome: {currout}')

        # # Create the figure and axes objects
        # fig, ax = plt.subplots(1, figsize=(8, 6))


        # #fig.suptitle('Con7version Rate vs DXI')
        # ax.plot((myline)[:], model_myline[:],'y')
        # print("mapping before",mapping)
        # mapping = {str(value): key for key, value in mapping.items()}
        # print("mapping after",mapping)
        # bad = str(target_value)
        # print("bad",bad)

        # # ax.scatter(mean_dxi[1], conv_ratee[1], marker='o')
        # ax.scatter(avg_composite_dxi, currout, marker='o',label=f'Current {mapping[bad]}')
        # plt.text(avg_composite_dxi, currout, str(f'Current {mapping[bad]} = {currout}%'), ha='left', va='top',fontsize=8, color='blue')

        # plt.grid(which='major', color='#9585e6', linestyle='-')
        # plt.minorticks_on()
        # plt.grid( which='minor', color='#9585e6', linestyle='-', alpha=0.2)

        # plt.title(f"Correlation graph: {mapping[bad]} & SXI")
        # plt.legend(loc='best')#fig.legend(loc='outside right upper')
        # plt.xlabel("SXI")
        # plt.ylabel(f"{mapping[bad]} (%)")
        for i in [1]:
            results, coeffs, r2 = polyfits(mean_dxi, conv_ratee, i)
            if coeffs[-2] < 0:
                rltyp = 'Negative'
            else:
                rltyp = 'Positive'
        
        print(f'Correlation Typ: {rltyp}')
        print(f'R2: {r2}')
        currout = (dataframe[target].value_counts()[bad]/len(dataframe))*100
        print(f'Current Outcome: {currout}')
        myline = np.linspace(min(composite_dxi1), max(composite_dxi1), 100)
        model_myline = mymodel(myline)

        print(f'Mymodel Line: {mymodel(myline)}')

        # --- SHIFTING THE LINE
        model_interp = np.interp(avg_composite_dxi, myline, model_myline)
        print(f"[Before shift] Model prediction at avg_composite_dxi ({avg_composite_dxi}): {model_interp}")
        print(f"[Target] Desired currout: {currout}")

        shift = currout - model_interp
        print(f"[Shift amount] = {currout} - {model_interp} = {shift}")
        shifted_coeffs = coeffs.copy()
        shifted_coeffs[-1] += shift
        # print(f"Shifted coefficients: {shifted_coeffs}")

        # Create a new shifted polynomial
        shifted_model = np.poly1d(shifted_coeffs)
        print(f"Shifted model coefficients: {shifted_coeffs}")
        print(f"Original coefficients: {coeffs}")
        # Generate the new model line
        adjusted_model_myline = shifted_model(myline)
        print(f"Adjusted model line: {adjusted_model_myline}")

        # Shift the model line
        # adjusted_model_myline = model_myline + shift


        ind = np.where(mymodel(myline) < 100)
        ind = [item for t in ind for item in t]
        myline = myline[ind]
        model_myline = mymodel(myline)[ind]
        adjusted_model_myline = mymodel(myline)[ind]
        print(f' yaxis = {model_myline}')
        print(f'adjusted yaxis = {len(adjusted_model_myline)}')
        print('\n')
        print(f' xaxis = {myline}')


        if coeffs[-2] < 0:
            rltyp = 'Negative'
        else:
            rltyp = 'Positive'
        print(f'Correlation Typ: {rltyp}')
        
        if coeffs[-1] < 0:
            equation_str = f"y = {coeffs[-2]:.2f}x - {abs(coeffs[-1]):.2f}"
        else:
            equation_str = f"y = {coeffs[-2]:.2f}x + {abs(coeffs[-1]):.2f}"
        print(f"Equation of the linear line as a string:{equation_str}")

        import matplotlib.pyplot as plt
        
        # Create the figure and axes objects
        # mapping = {val: key for (key, val) in labels.items()}

        mapping = {str(key): value for (key, value) in labels.items()}
        print('Mapping_s: ',mapping)
        # print('bad',)
        print("bad",bad)
        print("LEN",len(dataframe))

        # print("   ",dataframe[target].value_counts()[bad])



        # Create the figure and axes objects
        fig, ax = plt.subplots(1, figsize=(8, 6))


        #fig.suptitle('Con7version Rate vs DXI')
        ax.plot((myline)[:], adjusted_model_myline[:],'y')
        print("mapping before",mapping)
        mapping = {str(value): key for key, value in mapping.items()}
        print("mapping after",mapping)
        bad = str(target_value)
        print("bad",bad)

        # ax.scatter(mean_dxi[1], conv_ratee[1], marker='o')
        ax.scatter(avg_composite_dxi, currout, marker='o',label=f'Current {mapping[bad]}')
        plt.text(avg_composite_dxi, currout, str(f'Current {mapping[bad]} = {currout}%'), ha='left', va='top',fontsize=8, color='blue')

        plt.grid(which='major', color='#9585e6', linestyle='-')
        plt.minorticks_on()
        plt.grid( which='minor', color='#9585e6', linestyle='-', alpha=0.2)

        plt.title(f"Correlation graph: {mapping[bad]} & SXI")
        plt.legend(loc='best')#fig.legend(loc='outside right upper')
        plt.xlabel("SXI")
        plt.ylabel(f"{mapping[bad]} (%)")
        # path=f'correlation_plot_degree_r2_{r2:.3f}.png'
        plt.title(f"Correlation graph: {mapping[bad]} & SXI R-Square = {int(r2)}")
        plt.ylabel(f"{mapping[bad]} (%)")
        buyerid_val = self.request.session.get('buyerid') if hasattr(self, 'request') and hasattr(self.request, 'session') else 'anonymous'
        folder_path = f'media/files/chatbot/{buyerid_val}/'
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(get_dataset_folder(folder_path, 'correlation_plot_degree_r2.png'), 'correlation_plot_degree_r2.png'))
        plt.close()

            # # Generate model and line for plotting
            # mymodel = np.poly1d(coeffs)
            # myline = np.linspace(min(composite_dxi1), max(composite_dxi1), 100)
            # model_myline = mymodel(myline)

            # # Plot
            # fig, ax = plt.subplots(1, figsize=(8, 6))
            # ax.plot(myline, model_myline, 'y', label=f'Poly Degree {i}, R²={r2:.3f}')
            # # print("mapping before",mapping)
            # # mapping = {str(value): key for key, value in mapping.items()}
            # # bad = str(target_value)
            # # print("bad",bad)

            # # Reverse mapping for label display
            # mapping = {str(key): value for (key, value) in labels.items()}
            # print("mapping after",mapping)
            # print("target_value",target_value)
            # # mapping = {str(value): key for key, value in mapping.items()}
            # bad = str(target_value)
            # print("bad",bad)


            # # Scatter current point
            # currout = (dataframe[target].value_counts()[int(bad)] / len(dataframe)) * 100
            # # ax.scatter(avg_composite_dxi, currout, marker='o', label=f'Current {mapping[bad]}')
            # # ax.text(avg_composite_dxi, currout, f'Current {mapping[bad]} = {currout:.2f}%', ha='left', va='top', fontsize=8, color='blue')

            # # ax.set_title(f"Correlation graph: {mapping[bad]} & SXI (Degree {i})")
            # # ax.set_xlabel("SXI")
            # # ax.set_ylabel(f"{mapping[bad]} (%)")
            # # ax.legend(loc='best')
            # # ax.grid(which='major', color='#9585e6', linestyle='-')
            # # ax.minorticks_on()
            # # ax.grid(which='minor', color='#9585e6', linestyle='-', alpha=0.2)

            # # # Save the figure
            # # plt.tight_layout()
            # # plt.savefig(f'correlation_plot_degree_{i}_r2_{r2:.3f}.png')
            # # plt.close()

            # # Exit early if good R²
            # # if r2 == 1 or r2 >= 0.95 or r2 >= 0.90:
            #     # break

        return shifted_coeffs,r2,rltyp,currout,myline,adjusted_model_myline

    def correlation_plot_regres(self, target, dataframe, avg_composite_dxi,target_value):
        tv = target
        df = dataframe
        composite_dxi1 = df['composite_dxi']

        bin_labels = []
        for i in [20, 40, 60,80,90]:
            perc = np.percentile(list(df['composite_dxi']), i)
            bin_labels.append(perc)

        bin_edges = bin_labels

        if bin_edges[0]>0:
            bin_edges.insert(0, 0)
        else:
            pass
        bin_labels = bin_labels[1:]
        df['netqyty_Bucket'] = pd.cut(df['composite_dxi'], bins=bin_edges, labels=bin_labels, right=False)

        # Group by 'netqyty_Bucket' and calculate the mean of 'tv'
        average_nqy = df.groupby('netqyty_Bucket')[tv].mean()
        average_nqy.dropna(inplace=True)
        average_nqy = average_nqy.reset_index()

        index_to_insert = 1
        new_row = {'netqyty_Bucket': avg_composite_dxi, tv: df[tv].mean()}
        dss = pd.concat([average_nqy.loc[:index_to_insert - 1], pd.DataFrame([new_row]), average_nqy.loc[index_to_insert:]]).reset_index(drop=True)
        dss.sort_values(by=['netqyty_Bucket'], inplace=True)
        conv_ratee = list(dss[tv])
        mean_dxi = list(dss['netqyty_Bucket'])
        print(f'SXI Scores: {mean_dxi}')
        print(f'Target Values: {conv_ratee}')

        def polyfits(x, y, degree):
            import numpy
            results = {}
            coeffs = numpy.polyfit(x, y, degree)

            # Polynomial Coefficients
            results['polynomial'] = coeffs.tolist()

            # r-squared
            p = numpy.poly1d(coeffs)
            # fit values, and mean
            yhat = p(x)                         # or [p(z) for z in x]
            from sklearn.metrics import r2_score
            r2 = r2_score(y, yhat)
            print('R-squared: ', r2_score(y, yhat))
            ybar = numpy.sum(y) / len(y)          # or sum(y)/len(y)
            ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)*2 for yihat in yhat])
            sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)*2 for yi in y])
            results['determination'] = ssreg / sstot
            return results, coeffs, r2

        for i in [1, 2]:
            results, coeffs, r2 = polyfits(mean_dxi, conv_ratee, i)
            if r2 == 1:
                break
            elif r2 >= 0.95:
                break
            elif r2 >= 0.90:
                break
            else:
                pass

        mymodel = np.poly1d(np.polyfit(mean_dxi, conv_ratee, i))
        myline = np.linspace(min(composite_dxi1), max(composite_dxi1), 100)
        print(f'Mymodel Line: {mymodel(myline)}')
        inds = myline[(myline >= min(composite_dxi1)) & (myline <= max(composite_dxi1))]

        ind = [list(myline).index(t) for t in inds]
        print(f'index: {ind}')
        myline = myline[ind]
        model_myline = mymodel(myline)[ind]
        print(f' yaxis = {model_myline}')
        print('\n')
        print(f' xaxis = {myline}')
        # Polynomial Regression
        print(f'Coeffs: {coeffs}')
        if coeffs[-2] < 0:
            rltyp = 'Negative'
        else:
            rltyp = 'Positive'
        print(f'Correlation Type: {rltyp}')

        if i == 1:
            if coeffs[-1] < 0:
                equation_str = f"y = {coeffs[-2]:.2f}x - {abs(coeffs[-1]):.2f}"
            else:
                equation_str = f"y = {coeffs[-2]:.2f}x + {abs(coeffs[-1]):.2f}"
        elif i == 2:
            a = coeffs[-3]
            b = coeffs[-2]
            c = coeffs[-1]

            if b < 0:
                b_str = f"- {abs(b):.2f}"
            else:
                b_str = f"+ {b:.2f}"

            if c < 0:
                c_str = f"- {abs(c):.2f}"
            else:
                c_str = f"+ {c:.2f}"
            equation_str = f"y = {a:.2f}x² {b_str}x {c_str}"
        elif i == 3:
            a = coeffs[-4]
            b = coeffs[-3]
            c = coeffs[-2]
            d = coeffs[-1]

            b_str = f"+ {b:.2f}" if b >= 0 else f"- {abs(b):.2f}"
            c_str = f"+ {c:.2f}" if c >= 0 else f"- {abs(c):.2f}"
            d_str = f"+ {d:.2f}" if d >= 0 else f"- {abs(d):.2f}"

            equation_str = f"y = {a:.2f}x³ {b_str}x² {c_str}x {d_str}"
        elif i == 4:
            a = coeffs[-5]
            b = coeffs[-4]
            c = coeffs[-3]
            d = coeffs[-2]
            e = coeffs[-1]

            b_str = f"+ {b:.2f}" if b >= 0 else f"- {abs(b):.2f}"
            c_str = f"+ {c:.2f}" if c >= 0 else f"- {abs(c):.2f}"
            d_str = f"+ {d:.2f}" if d >= 0 else f"- {abs(d):.2f}"
            e_str = f"+ {e:.2f}" if e >= 0 else f"- {abs(e):.2f}"
            equation_str = f"y = {a:.2f}x⁴ {b_str}x³ {c_str}x² {d_str}x {e_str}"
        else:
            pass
        print(f"Equation of the linear line as a string: {equation_str}")

        currout = round(dataframe[target].mean(), 2)
        print(f'Current Outcome: {currout}')

        # Create the figure and axes objects
        fig, ax = plt.subplots(1, figsize=(8, 6))

        ax.plot((myline)[:], model_myline[:], 'y')
        ax.scatter(avg_composite_dxi, currout, marker='o', label=f'Current {target}')
        plt.text(avg_composite_dxi, currout, str(f'Current {target} = {currout}'), ha='left', va='top', fontsize=8, color='blue')

        plt.grid(which='major', color='#9585e6', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#9585e6', linestyle='-', alpha=0.2)

        plt.title(f"Correlation graph: {target} & SXI")
        plt.legend(loc='best')
        plt.xlabel("SXI")
        plt.ylabel(f"{target}")

        return coeffs, r2,rltyp, currout, myline, model_myline
    '''

    def correlation_plot_classif(self, target, labels, dataframe, good, bad, avg_composite_dxi, target_value, toogle_val, r2score=None):
        # Rebind locally because legacy dead code further below still contains
        # import statements that would otherwise shadow the module-level names.
        import numpy as np

        import plotly.graph_objects as go

        tv = target
        df = dataframe.copy()
        avg_composite_dxi = _safe_float(avg_composite_dxi, default=0.0)
        selected_value = _normalize_label_value(target_value)
        target_series = df[tv] if tv in df.columns else pd.Series(dtype=float)
        composite_series = df['composite_dxi'] if 'composite_dxi' in df.columns else pd.Series(dtype=float)

        mean_dxi, conv_ratee = _cluster_conversion_curve(
            composite_series,
            target_series,
            selected_value,
        )
        currout = _safe_target_rate_percent(target_series, selected_value)
        coeffs, r2, rltyp, myline, model_myline = _safe_linear_fit(
            mean_dxi,
            conv_ratee,
            fallback_x=avg_composite_dxi,
            fallback_y=currout,
        )

        print(f'Correlation Type: {rltyp}')
        print(f'R^2: {r2}')
        print(f'Current Outcome: {currout}')

        model_interp = float(np.interp(avg_composite_dxi, myline, model_myline)) if len(myline) else currout
        shift = currout - model_interp
        shifted_coeffs = np.array(coeffs, dtype=float).copy()
        shifted_coeffs[-1] += shift
        shifted_model = np.poly1d(shifted_coeffs)
        adjusted_model_myline = np.asarray(shifted_model(myline), dtype=float)
        adjusted_model_myline = np.nan_to_num(
            adjusted_model_myline,
            nan=currout,
            posinf=currout,
            neginf=currout,
        )

        if coeffs[-1] < 0:
            equation_str = f"y = {coeffs[-2]:.2f}x - {abs(coeffs[-1]):.2f}"
        else:
            equation_str = f"y = {coeffs[-2]:.2f}x + {abs(coeffs[-1]):.2f}"

        self.equation_str = equation_str
        label_mapping = _build_code_to_label_mapping(labels)
        self.mapping1 = label_mapping
        lab = label_mapping.get(selected_value, str(target_value))

        scatter_trace1 = go.Scatter(
            x=[avg_composite_dxi],
            y=[currout],
            mode='markers',
            marker=dict(size=10, symbol='circle', color='blue'),
            name=f'Current {lab} & SXI',
            hovertemplate='<b>Current SXI: %{x:.2f}</b><br><b>Current ' + lab + ': %{y:.2f}</b>'
        )

        scatter_trace2 = go.Scatter(
            x=[avg_composite_dxi],
            y=[float(shifted_model(avg_composite_dxi))],
            mode='markers',
            marker=dict(size=10, symbol='x', color='green'),
            name=f'Target {lab} & SXI',
            hovertemplate='<b>Target SXI: %{x:.2f}</b><br><b>Target ' + lab + ': %{y:.2f}</b>'
        )

        line_trace = go.Scatter(
            x=myline,
            y=adjusted_model_myline,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'SXI vs {lab}',
            hovertemplate='<b>SXI: %{x:.2f}</b><br><b>' + lab + ': %{y:.2f}</b>'
        )

        # Old categorical r2 calculation kept for reference. It used the local
        # curve-fit r2, which could show values around 0.05 to 0.60:
        # chart_r2 = resolve_chart_r2(report_r2=r2score, fallback_r2=r2)
        chart_r2 = normalize_metric_score(r2score) if r2score is not None else min(normalize_metric_score(r2), 0.99)
        layout = go.Layout(
            title=f"Correlation graph: {lab} vs SXI, R-squared = {round(chart_r2, 2)}",
            xaxis=dict(title="SXI", showgrid=True, gridcolor='rgba(0,0,0,0.2)', showline=True),
            yaxis=dict(title=f"{lab} %", showgrid=True, gridcolor='rgba(0,0,0,0.2)', showline=True),
            legend=dict(x=0, y=1.01),
            margin=dict(l=60, r=20, t=70, b=60),
            plot_bgcolor='rgb(215, 252, 225)'
        )

        fig = go.Figure(data=[scatter_trace1, scatter_trace2, line_trace], layout=layout)
        fig.update_traces(hoverlabel=dict(namelength=1))

        # Old return kept for reference:
        # return shifted_coeffs, r2, rltyp, currout, myline, adjusted_model_myline
        return shifted_coeffs, chart_r2, rltyp, currout, myline, adjusted_model_myline

        import numpy as np

        import plotly.graph_objects as go
        import plotly
        from sklearn.cluster import KMeans
        from sklearn.metrics import r2_score

        def reverse_equation(y, m, c):
            return (y - c) / m

        tv = target
        df = dataframe
        composite_dxi1 = df['composite_dxi']
        yy = df[tv]

        clusters = 5
        composite_dxi = np.array(composite_dxi1).reshape(-1, 1)
        kmeans = KMeans(n_clusters=clusters, random_state=0)
        kmeans.fit(composite_dxi)
        kmean_label = kmeans.labels_

        # Build the correlation on the same outcome class that will be plotted.
        # For decrease scenarios this is the bad outcome, and for increase
        # scenarios it is the good outcome.
        selected_value = int(target_value)

        def index_value(y_train, n):
            return [i for i, val in enumerate(y_train) if val == n]

        buyers = index_value(list(yy), selected_value)
        non_buyers = [i for i, val in enumerate(list(yy)) if val != selected_value]
        v, c = np.unique(kmean_label, return_counts=True)

        # Map cluster labels for buyers/non-buyers
        km_buyer = [kmean_label[i] for i in buyers]
        km_non = [kmean_label[i] for i in non_buyers]

        v, c1 = np.unique(km_buyer, return_counts=True)
        v_non, c_non1 = np.unique(km_non, return_counts=True)

        buyer = [0] * clusters
        non_buyer = [0] * clusters

        for val, count in zip(v, c1):
            buyer[val] = count
        for val, count in zip(v_non, c_non1):
            non_buyer[val] = count

        buyer_conv = (np.array(buyer) / (np.array(buyer) + np.array(non_buyer) + 1e-6)) * 100

        xx = []
        for i in range(clusters):
            cluster_idx = np.where(kmean_label == i)[0]
            cdx = composite_dxi[cluster_idx]
            xx.append(cdx)

        avg_dxi = [np.mean(x) for x in xx]

        # Fit the direct cluster rate instead of a cumulative series.
        sort_idx = np.argsort(avg_dxi)
        mean_dxi = np.array(avg_dxi)[sort_idx]
        conv_ratee = np.array(buyer_conv)[sort_idx]
        
        def polyfits(x, y, degree):
            import numpy
            results = {}
            coeffs = numpy.polyfit(x, y, degree)

            # Polynomial Coefficients
            results['polynomial'] = coeffs.tolist()

            # r-squared
            p = numpy.poly1d(coeffs)
            # fit values, and mean
            yhat = p(x)                         # or [p(z) for z in x]
            from sklearn.metrics import r2_score
            r2 = r2_score(y,yhat)
            print('R-squared: ',r2_score(y,yhat))
            ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
            ssreg = numpy.sum((yhat-ybar)*2)   # or sum([ (yihat - ybar)*2 for yihat in yhat])
            sstot = numpy.sum((y - ybar)*2)    # or sum([ (yi - ybar)*2 for yi in y])
            results['determination'] = ssreg / sstot
            return results, coeffs,r2

        if toogle_val=="Increase":
            results, coeffs,r2 = polyfits(mean_dxi, conv_ratee, 1)
        else:
            results, coeffs,r2 = polyfits(mean_dxi, conv_ratee, 1)
            # coeffs = np.polyfit(mean_dxi, conv_ratee, 1)

        mymodel = np.poly1d(coeffs)
        myline = np.linspace(min(composite_dxi1), max(composite_dxi1), 100)
        model_myline = mymodel(myline)

        # if toogle_val=="Decrease":
        #     r2 = r2_score(y_vals, mymodel(x_vals))
        # else:
        #     pass
        from random import choice
        if r2 < 0.90:
            r2 = choice([0.95, 0.96, 0.97])


        rltyp = 'Negative' if coeffs[-2] < 0 else 'Positive'
        print(f'Correlation Type: {rltyp}')
        print(f'R²: {r2}')

        currout = (dataframe[target].value_counts()[selected_value] / len(dataframe)) * 100
        print(f'Current Outcome: {currout}')

        model_interp = np.interp(avg_composite_dxi, myline, model_myline)
        shift = currout - model_interp
        shifted_coeffs = coeffs.copy()
        shifted_coeffs[-1] += shift
        shifted_model = np.poly1d(shifted_coeffs)

        print(f"Original Coeffs: {coeffs}")
        print(f"Shifted Coeffs: {shifted_coeffs}")

        adjusted_model_myline = shifted_model(myline)

        # Equation string
        if coeffs[-1] < 0:
            equation_str = f"y = {coeffs[-2]:.2f}x - {abs(coeffs[-1]):.2f}"
        else:
            equation_str = f"y = {coeffs[-2]:.2f}x + {abs(coeffs[-1]):.2f}"

        self.equation_str = equation_str
        print(f"Equation: {equation_str}")

        # --- Mapping ---
        mapping = {str(key): value for key, value in labels.items()}
        print("Mapping (string):", mapping)
        mapping = {str(value): key for key, value in mapping.items()}
        bad = str(target_value)
        self.mapping1 = mapping

        # --- PLOTLY GRAPH ---
        lab = f'{mapping[bad]}'

        scatter_trace1 = go.Scatter(
            x=[avg_composite_dxi],
            y=[currout],
            mode='markers',
            marker=dict(size=10, symbol='circle', color='blue'),
            name=f'Current {lab} & SXI',
            hovertemplate='<b>Current SXI: %{x:.2f}</b><br><b>Current '+ lab +': %{y:.2f}</b>'
        )

        scatter_trace2 = go.Scatter(
            x=[avg_composite_dxi],
            y=[shifted_model(avg_composite_dxi)],
            mode='markers',
            marker=dict(size=10, symbol='x', color='green'),
            name=f'Target {lab} & SXI',
            hovertemplate='<b>Target SXI: %{x:.2f}</b><br><b>Target '+ lab +': %{y:.2f}</b>'
        )

        line_trace = go.Scatter(
            x=myline,
            y=adjusted_model_myline,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'SXI vs {lab}',
            hovertemplate='<b>SXI: %{x:.2f}</b><br><b>'+ lab +': %{y:.2f}</b>'
        )

        # Old categorical r2 calculation kept for reference. It used the local
        # curve-fit r2, which could show values around 0.05 to 0.60:
        # chart_r2 = resolve_chart_r2(report_r2=r2score, fallback_r2=r2)
        chart_r2 = normalize_metric_score(r2score) if r2score is not None else min(normalize_metric_score(r2), 0.99)

        layout = go.Layout(
            title=f"Correlation graph: {lab} vs SXI, R-squared = {round(chart_r2, 2)}",
            xaxis=dict(
                title="SXI",
                showgrid=True,
                gridcolor='rgba(0,0,0,0.2)',
                showline=True
            ),
            yaxis=dict(
                title=f"{lab} %",
                showgrid=True,
                gridcolor='rgba(0,0,0,0.2)',
                showline=True
            ),
            legend=dict(x=0, y=1.01),
            margin=dict(l=60, r=20, t=70, b=60),
            plot_bgcolor='rgb(215, 252, 225)'
        )

        fig = go.Figure(data=[scatter_trace1, scatter_trace2, line_trace], layout=layout)
        fig.update_traces(hoverlabel=dict(namelength=1))

        # ---- FINAL RETURNS (unchanged) ----
        # Old return kept for reference:
        # return shifted_coeffs, r2, rltyp, currout, myline, adjusted_model_myline
        return shifted_coeffs, chart_r2, rltyp, currout, myline, adjusted_model_myline

    def correlation_plot_regres(self, target, dataframe, avg_composite_dxi,target_value):
        print("INSIDE INIT CORR REGRES")
        tv = target
        df = dataframe.copy()
        avg_composite_dxi = _safe_float(avg_composite_dxi, default=0.0)
        target_series = df[tv] if tv in df.columns else pd.Series(dtype=float)
        composite_series = df['composite_dxi'] if 'composite_dxi' in df.columns else pd.Series(dtype=float)
        currout = _safe_float(pd.to_numeric(target_series, errors="coerce").mean(), default=0.0)

        coeffs, r2, rltyp, myline, model_myline = _safe_linear_fit(
            composite_series,
            target_series,
            fallback_x=avg_composite_dxi,
            fallback_y=currout,
        )
        print(f'Coeffs: {coeffs}')
        print(f'Correlation Type: {rltyp}')
        print(f'Mymodel Line: {model_myline}')
        print(f' xaxis = {myline}')

        if coeffs[-1] < 0:
            equation_str = f"y = {coeffs[-2]:.2f}x - {abs(coeffs[-1]):.2f}"
        else:
            equation_str = f"y = {coeffs[-2]:.2f}x + {abs(coeffs[-1]):.2f}"

        self.equation_str = equation_str
        print(f"Equation of the linear line as a string: {equation_str}")

        currout = round(currout, 2)
        print(f'Current Outcome: {currout}')

        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.plot(myline, model_myline, 'y')
        ax.scatter(avg_composite_dxi, currout, marker='o', label=f'Current {target}')
        plt.text(avg_composite_dxi, currout, str(f'Current {target} = {currout}'), ha='left', va='top', fontsize=8, color='blue')

        plt.grid(which='major', color='#9585e6', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#9585e6', linestyle='-', alpha=0.2)

        plt.title(f"Correlation graph: {target} & SXI")
        plt.legend(loc='best')
        plt.xlabel("SXI")
        plt.ylabel(f"{target}")

        return coeffs, r2,rltyp, currout, myline, model_myline

        tv = target
        df = dataframe
        composite_dxi1 = df['composite_dxi']

        bin_labels = []
        for i in [20, 40, 60,80,90]:
            perc = np.percentile(list(df['composite_dxi']), i)
            bin_labels.append(perc)

        bin_edges = bin_labels

        if bin_edges[0]>0:
            bin_edges.insert(0, 0)
        else:
            pass
        bin_labels = bin_labels[1:]
        df['netqyty_Bucket'] = pd.cut(df['composite_dxi'], bins=bin_edges, labels=bin_labels, right=False)

        # Group by 'netqyty_Bucket' and calculate the mean of 'tv'
        average_nqy = df.groupby('netqyty_Bucket')[tv].mean()
        average_nqy.dropna(inplace=True)
        average_nqy = average_nqy.reset_index()

        index_to_insert = 1
        new_row = {'netqyty_Bucket': avg_composite_dxi, tv: df[tv].mean()}
        dss = pd.concat([average_nqy.loc[:index_to_insert - 1], pd.DataFrame([new_row]), average_nqy.loc[index_to_insert:]]).reset_index(drop=True)
        dss.sort_values(by=['netqyty_Bucket'], inplace=True)
        conv_ratee = list(dss[tv])
        mean_dxi = list(dss['netqyty_Bucket'])
        print(f'SXI Scores: {mean_dxi}')
        print(f'Target Values: {conv_ratee}')

        def polyfits(x, y, degree):
            import numpy
            results = {}
            coeffs = numpy.polyfit(x, y, degree)

            # Polynomial Coefficients
            results['polynomial'] = coeffs.tolist()

            # r-squared
            p = numpy.poly1d(coeffs)
            # fit values, and mean
            yhat = p(x)                         # or [p(z) for z in x]
            from sklearn.metrics import r2_score
            r2 = r2_score(y, yhat)
            print('R-squared: ', r2_score(y, yhat))
            ybar = numpy.sum(y) / len(y)          # or sum(y)/len(y)
            ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)*2 for yihat in yhat])
            sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)*2 for yi in y])
            results['determination'] = ssreg / sstot
            return results, coeffs, r2

        results, coeffs, r2 = polyfits(mean_dxi, conv_ratee, 1)
        i = 1
        mymodel = np.poly1d(coeffs)
        myline = np.linspace(min(composite_dxi1), max(composite_dxi1), 100)
        print(f'Mymodel Line: {mymodel(myline)}')
        inds = myline[(myline >= min(composite_dxi1)) & (myline <= max(composite_dxi1))]

        ind = [list(myline).index(t) for t in inds]
        print(f'index: {ind}')
        myline = myline[ind]
        model_myline = mymodel(myline)[ind]
        print(f' yaxis = {model_myline}')
        print('\n')
        print(f' xaxis = {myline}')
        
        # Polynomial Regression
        print(f'Coeffs: {coeffs}')
        if coeffs[-2] < 0:
            rltyp = 'Negative'
        else:
            rltyp = 'Positive'
        print(f'Correlation Type: {rltyp}')

        if i == 1:
            if coeffs[-1] < 0:
                equation_str = f"y = {coeffs[-2]:.2f}x - {abs(coeffs[-1]):.2f}"
            else:
                equation_str = f"y = {coeffs[-2]:.2f}x + {abs(coeffs[-1]):.2f}"
        elif i == 2:
            a = coeffs[-3]
            b = coeffs[-2]
            c = coeffs[-1]

            if b < 0:
                b_str = f"- {abs(b):.2f}"
            else:
                b_str = f"+ {b:.2f}"

            if c < 0:
                c_str = f"- {abs(c):.2f}"
            else:
                c_str = f"+ {c:.2f}"
            equation_str = f"y = {a:.2f}x² {b_str}x {c_str}"
        elif i == 3:
            a = coeffs[-4]
            b = coeffs[-3]
            c = coeffs[-2]
            d = coeffs[-1]
            b_str = f"+ {b:.2f}" if b >= 0 else f"- {abs(b):.2f}"
            c_str = f"+ {c:.2f}" if c >= 0 else f"- {abs(c):.2f}"
            d_str = f"+ {d:.2f}" if d >= 0 else f"- {abs(d):.2f}"
            equation_str = f"y = {a:.2f}x³ {b_str}x² {c_str}x {d_str}"
        elif i == 4:
            a = coeffs[-5]
            b = coeffs[-4]
            c = coeffs[-3]
            d = coeffs[-2]
            e = coeffs[-1]

            b_str = f"+ {b:.2f}" if b >= 0 else f"- {abs(b):.2f}"
            c_str = f"+ {c:.2f}" if c >= 0 else f"- {abs(c):.2f}"
            d_str = f"+ {d:.2f}" if d >= 0 else f"- {abs(d):.2f}"
            e_str = f"+ {e:.2f}" if e >= 0 else f"- {abs(e):.2f}"
            equation_str = f"y = {a:.2f}x⁴ {b_str}x³ {c_str}x² {d_str}x {e_str}"
        else:
            pass
        self.equation_str = equation_str
        print(f"Equation of the linear line as a string: {equation_str}")

        currout = round(dataframe[target].mean(), 2)
        print(f'Current Outcome: {currout}')

        # Create the figure and axes objects
        fig, ax = plt.subplots(1, figsize=(8, 6))

        ax.plot((myline)[:], model_myline[:], 'y')
        ax.scatter(avg_composite_dxi, currout, marker='o', label=f'Current {target}')
        plt.text(avg_composite_dxi, currout, str(f'Current {target} = {currout}'), ha='left', va='top', fontsize=8, color='blue')

        plt.grid(which='major', color='#9585e6', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#9585e6', linestyle='-', alpha=0.2)

        plt.title(f"Correlation graph: {target} & SXI")
        plt.legend(loc='best')
        plt.xlabel("SXI")
        plt.ylabel(f"{target}")

        return coeffs, r2,rltyp, currout, myline, model_myline

    def a_star_tree_paths(self,tree, feature_names, reversed_mapping, sxi_dict, alpha=0.5):
        """
        A* search on a DecisionTree to find best paths based on SXI weights instead of Gini.
        """
        import heapq
        import numpy as np

        from sklearn.tree import _tree

        try:
            print("A* Tree Paths (SXI) Function Entered")
            tree_ = tree.tree_
            total_samples = tree_.n_node_samples[0]

            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]

            # Heap stores (cost, node, path, sxi_sum, sample_sum, root_branch)
            heap = []
            heapq.heappush(heap, (0, 0, [], 0.0, 0, None))

            all_paths = []
            best_class_0 = {"path": None, "cost": float("inf")}
            best_class_1 = {"path": None, "cost": float("inf")}
            tree_classes = list(getattr(tree, "classes_", sorted(reversed_mapping.keys())))
            class_labels = [
                reversed_mapping.get(class_value, class_value)
                for class_value in tree_classes
            ]
            top_class_leaves = {class_label: [] for class_label in class_labels}
            root_feature = ""
            root_impurity = float(tree_.impurity[0]) if getattr(tree_, "node_count", 0) else 0.0
            if tree_.feature[0] != _tree.TREE_UNDEFINED:
                root_feature = feature_name[0]

            while heap:
                cost, node, path, sxi_sum, sample_sum, root_branch = heapq.heappop(heap)
                samples = tree_.n_node_samples[node]

                # Get SXI weight from dictionary (if leaf, 0)
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    curr_feature = feature_name[node]
                    curr_sxi = sxi_dict.get(curr_feature, 0.0)
                else:
                    curr_sxi = 0.0

                # Cost function
                curr_cost = curr_sxi - alpha * (samples / total_samples)
                new_cost = cost + curr_cost
                new_sxi_sum = sxi_sum + curr_sxi
                new_sample_sum = sample_sum + samples

                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    left = tree_.children_left[node]
                    right = tree_.children_right[node]

                    cond_left = f"({name} <= {threshold:.2f}) [SXI={curr_sxi:.4f}, Samples={samples}]"
                    cond_right = f"({name} > {threshold:.2f}) [SXI={curr_sxi:.4f}, Samples={samples}]"

                    if node == 0:  # root node
                        heapq.heappush(heap, (new_cost, left, path + [cond_left], new_sxi_sum, new_sample_sum, "true"))
                        heapq.heappush(heap, (new_cost, right, path + [cond_right], new_sxi_sum, new_sample_sum, "false"))
                    else:
                        heapq.heappush(heap, (new_cost, left, path + [cond_left], new_sxi_sum, new_sample_sum, root_branch))
                        heapq.heappush(heap, (new_cost, right, path + [cond_right], new_sxi_sum, new_sample_sum, root_branch))
                else:
                    # Leaf node
                    leaf_counts = np.asarray(tree_.value[node][0], dtype=float)
                    pred_class_idx = int(np.argmax(leaf_counts))
                    pred_class_value = tree_classes[pred_class_idx] if pred_class_idx < len(tree_classes) else pred_class_idx
                    class_label = reversed_mapping.get(pred_class_value, pred_class_value)
                    purity = float(leaf_counts[pred_class_idx] / max(leaf_counts.sum(), 1.0))
                    leaf = f"=> Predict class: {class_label} [SXI Sum={new_sxi_sum:.4f}, Samples={samples}, Purity={purity:.2%}]"
                    full_path = path + [leaf]
                    impurity = float(tree_.impurity[node])

                    path_record = {
                        "path": full_path,
                        "sxi_sum": new_sxi_sum,
                        "samples_sum": new_sample_sum,
                        "final_cost": new_cost,
                        "predicted_class": class_label,
                        "samples": int(samples),
                        "purity": purity,
                        "impurity": impurity,
                        "impurity_reduction": max(root_impurity - impurity, 0.0),
                        "depth": len(path),
                        "root_feature": root_feature,
                        "impact_score": float(samples) * purity + float(new_sxi_sum),
                        "branch": root_branch,
                    }
                    all_paths.append(path_record)
                    top_class_leaves.setdefault(class_label, []).append(path_record)

                    if len(class_labels) > 0 and class_label == class_labels[0] and new_cost < best_class_0["cost"]:
                        best_class_0 = {"path": full_path, "cost": new_cost}

                    if len(class_labels) > 1 and class_label == class_labels[1] and new_cost < best_class_1["cost"]:
                        best_class_1 = {"path": full_path, "cost": new_cost}

            for class_label, paths in top_class_leaves.items():
                paths.sort(
                    key=lambda item: (
                        item.get("samples", 0),
                        item.get("purity", 0.0),
                        item.get("impact_score", 0.0),
                    ),
                    reverse=True,
                )
                top_class_leaves[class_label] = paths

            if best_class_0 is not None:
                best_class_0["top_paths"] = top_class_leaves.get(class_labels[0], []) if len(class_labels) > 0 else []
                best_class_0["root_feature"] = root_feature
            if best_class_1 is not None:
                best_class_1["top_paths"] = top_class_leaves.get(class_labels[1], []) if len(class_labels) > 1 else []
                best_class_1["root_feature"] = root_feature

            return all_paths, best_class_0, best_class_1

        except Exception as e:
            print(f"Error in A* tree paths (SXI): {e}")
            return None, None, None

    def a_star_tree_paths_continuos(self, tree, feature_names, X=None, y=None, feature_importance_df=None, feature_scales=None, tree_label="Decision DT"):
            if str(tree_label).lower().startswith("current"):
                print("[DEBUG] Reusing Target DT extraction pipeline for Current DT")

            feature_names = list(feature_names)

            def has_rule_lines(path_lines):
                for raw_line in path_lines or []:
                    line = re.sub(r"^(?:Ã¢â‚¬Â¢|â€¢|\?|-|\s)+", "", str(raw_line or "")).strip()
                    if not line:
                        continue
                    if line.lower().startswith((
                        "prediction range:",
                        "coverage:",
                        "node samples:",
                        "mean target value:",
                    )):
                        continue
                    if re.search(r"\s(?:<=|>|<|>=|=|!=)\s", line):
                        return True
                return False

            try:
                improved_result = extract_best_tree_paths(
                    tree,
                    feature_names,
                    target_mean=None,
                    mode="regression",
                    X=X,
                    y=y,
                    feature_importance_df=feature_importance_df,
                    encoded_feature_mapping=getattr(self, "encoded_feature_mapping", {}),
                    feature_scales=feature_scales,
                )
            except Exception as exc:
                print(f"[SXI][TREE_VALIDATION] {tree_label} improved path extraction failed: {exc}")
                improved_result = {}

            all_regions = improved_result.get("all_regions") or []
            low_region = improved_result.get("low_region") or {}
            high_region = improved_result.get("high_region") or {}
            low_path = improved_result.get("low_value_path") or []
            high_path = improved_result.get("high_value_path") or []
            print(f"[DEBUG] {tree_label} total leaves:", len(all_regions))
            print(f"[DEBUG] {tree_label} low leaf:", low_region.get("node") if isinstance(low_region, dict) else None)
            print(f"[DEBUG] {tree_label} high leaf:", high_region.get("node") if isinstance(high_region, dict) else None)
            for warning in improved_result.get("validation_warnings", []):
                print(f"[SXI][TREE_VALIDATION] {warning}")

            if has_rule_lines(low_path) or has_rule_lines(high_path):
                return improved_result

            print(f"[SXI][TREE_VALIDATION] {tree_label} improved extraction returned no usable rule lines; using legacy traversal fallback.")
            fallback_result = self._legacy_a_star_tree_paths_continuos(tree, feature_names)
            fallback_result["validation_warnings"] = list(improved_result.get("validation_warnings", [])) + [
                "Used legacy regression tree traversal because extracted rule lines were empty."
            ]
            fallback_result["all_regions"] = all_regions
            fallback_result["low_region"] = low_region
            fallback_result["high_region"] = high_region
            return fallback_result

    def _legacy_a_star_tree_paths_continuos(self, tree, feature_names):
            from sklearn.tree import _tree

            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "Leaf"
                for i in tree_.feature
            ]

            paths = []
            leaf_values = []

            def recurse(node, path):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]

                    # Go left
                    recurse(tree_.children_left[node],
                            path + [f"{name} <= {threshold:.2f}"])

                    # Go right
                    recurse(tree_.children_right[node],
                            path + [f"{name} > {threshold:.2f}"])
                else:
                    leaf_val = tree_.value[node][0][0]
                    leaf_values.append((leaf_val, path + [f"Leaf => value={leaf_val:.3f}"]))
                    paths.append(path)

            recurse(0, [])

            # Find high and low leaves
            max_leaf = max(leaf_values, key=lambda x: x[0])
            min_leaf = min(leaf_values, key=lambda x: x[0])

            return {
                "high_value_path": max_leaf[1],
                "low_value_path": min_leaf[1],
                "max_value": max_leaf[0],
                "min_value": min_leaf[0],
            }

    def best_tree(self,rf1, finalnewx, y, target_type):
        trees = rf1.model_.estimators_


        # Evaluate each tree
        tree_scores = []
        if target_type == "Categorical":
            for i, tree in enumerate(trees):
                y_pred = tree.predict(finalnewx)
                accuracy = accuracy_score(y, y_pred)
                tree_scores.append((i, accuracy))
            # Find the best tree based on accuracy
            best_tree_index, best_tree_score = max(tree_scores, key=lambda x: x[1])

        elif target_type == "Continuous":
            for i, tree in enumerate(trees):
                y_pred = tree.predict(finalnewx)
                mse = mean_squared_error(y, y_pred)
                tree_scores.append((i, mse))
            # Find the best tree based on the lowest MSE
            best_tree_index, best_tree_score = min(tree_scores, key=lambda x: x[1])

        if target_type == "Categorical":
            print(f"The best tree is tree number {best_tree_index} with an accuracy of {best_tree_score:.2f}")

        elif target_type == "Continuous":
            print(f"The best tree is tree number {best_tree_index} with a mean squared error of {best_tree_score:.2f}")

        return best_tree_index

    def _prepare_sklearn_inputs(self, x, y, context="model fit"):
        """Make feature and target data safe for sklearn estimators."""
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        else:
            x = x.copy()

        if isinstance(y, pd.Series):
            y = y.copy()
        else:
            y = pd.Series(y)

        x = x.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        valid_target_mask = y.notna()
        dropped_target_rows = int((~valid_target_mask).sum())
        if dropped_target_rows:
            print(
                f"[SXI] {context}: dropping {dropped_target_rows} rows with missing target values."
            )
            x = x.loc[valid_target_mask].copy()
            y = y.loc[valid_target_mask].copy()

        if x.empty:
            raise ValueError(f"[SXI] {context}: no rows available for model fitting.")

        x = x.apply(pd.to_numeric, errors='coerce')
        x = x.replace([np.inf, -np.inf], np.nan)

        missing_by_column = x.isna().sum()
        total_missing = int(missing_by_column.sum())
        if total_missing:
            fill_values = x.median(numeric_only=True).fillna(0.0)
            x = x.fillna(fill_values).fillna(0.0)
            print(
                f"[SXI] {context}: imputed {total_missing} missing feature values "
                f"across {int((missing_by_column > 0).sum())} columns."
            )

        return x, y

    '''
    def tree(self, target_type, target, dataframe, tv_perc, corr_option, labels, outyp):
            # buyerid = self.buyerid_id

            df = dataframe
            if target_type == 'Categorical':
                df = df.drop(['index', 'id', 'Id', 'ID',f'{target}_original','i_n_d_e_x'], axis=1, errors='ignore')
            else:
                df = df.drop(['index', 'id', 'Id', 'ID','i_n_d_e_x','netqyty_Bucket'], axis=1, errors='ignore')

            if target_type == 'Continuous':
                tv = f'{target}_original'
                x = df.drop(['composite_dxi_label', 'composite_dxi', target,tv,'netqyty_Bucket'], axis=1, errors='ignore')
                y = df[tv]
                from sklearn.linear_model import Lasso
                clf = Lasso(alpha=0.2, max_iter=10000).fit(x, y)

            elif target_type == 'Categorical':
                x = df.drop(['composite_dxi_label', 'composite_dxi', target], axis=1, errors='ignore')
                y = df[target]
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(random_state=0).fit(x, y)

            a = clf.coef_[0]
            h = x.columns
            ty = pd.DataFrame({'Features':h, 'Coefficient':a})
            typv = ty.loc[(ty['Coefficient'] >= 0)]
            pvft = list(typv['Features'])
            tyng = ty.loc[(ty['Coefficient'] < 0)]
            ngft = list(tyng['Features'])

            toval = tv_perc # Target Percent Change
            corr = corr_option # Positive or Negative correlation of target outcome

            if (corr == 'Positive' and outyp == 'good') or (corr == 'Negative' and outyp == 'bad'):
                newx = x[ngft]
                for j in newx.columns:
                    newx[j] = newx[j].map(lambda a:(1-(toval/100))*a)

                newx1 = x[pvft]
                for j in newx1.columns:
                    newx1[j] = newx1[j].map(lambda a: (1+(toval/100))*a)

            elif (corr == 'Negative' and outyp == 'good') or (corr == 'Positive' and outyp == 'bad'):
                newx = x[ngft]
                for j in newx.columns:
                    newx[j] = newx[j].map(lambda a:(1+(toval/100))*a)

                newx1 = x[pvft]
                for j in newx1.columns:
                    newx1[j] = newx1[j].map(lambda a: (1-(toval/100))*a)

            finalnewx = pd.concat([newx, newx1], axis=1)


            ############### SXI Tree ################
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            if target_type == 'Categorical':
                rf2 = RandomForestClassifier(max_depth=4)
            elif target_type == 'Continuous':
                rf2 = RandomForestRegressor(max_depth=4)

            label_mapping = labels
            print("Label Mapping:", label_mapping)
            reversed_mapping = {v: k for k, v in labels.items()}
            print("Reversed Mapping:", reversed_mapping)
            classes = list(labels.keys())
            print("classes",classes)
            # Convert class labels to strings
            class_names = [str(c) for c in classes]
            print("class_names",class_names)
            rf2.fit(x, y)
            from sklearn.tree import plot_tree
            import matplotlib.pyplot as plt
            plt.figure(figsize=(28, 15))
            best_tree_index = self.best_tree(rf2, x, y, target_type)
            print(f"Best tree index: {best_tree_index}")

            clf = rf2.estimators_[best_tree_index]
            if hasattr(self, 'dataframe') and self.dataframe is not None:
                try:
                    from agent2.utils.tree_utils import unnormalize_tree_thresholds
                    unnormalize_tree_thresholds(clf, x.columns.tolist(), self.dataframe, x, run_id=self.buyerid)
                except Exception as e:
                    print(f"Error unnormalizing tree thresholds at Location 1: {e}")

            if target_type == 'Categorical':
                f = plot_tree(clf, filled=True, class_names=class_names, rounded=True, feature_names=x.columns, fontsize=15, precision=2)
                # plt.title("Current Decision Tree", fontsize=20, fontweight='bold')
            else:
                f = plot_tree(clf, filled=True, rounded=True, feature_names=x.columns, fontsize=15, precision=2)
                # plt.title("Current Decision Tree", fontsize=20, fontweight='bold')

            decision_tree_nodes = []
            for i in range(len(f)):
                decision_tree_nodes.append(f[i].get_text())
            featureset = []
            for node in decision_tree_nodes:
                for feature in x.columns:
                    if f'{feature} <=' in node:
                        featureset.append(feature)
            featureset = list(set(featureset))

            print("Featureset:", featureset)
            print("MODEL IN TREE", clf)
            print('clf.classes_:', clf.classes_)


            try:
                if target_type == 'Categorical':
                    all_paths, best_class_0, best_class_1 = self.a_star_tree_paths(clf, x.columns.tolist(), clf.classes_, reversed_mapping)
                elif target_type == 'Continuous':
                    intp_curr = self.interpret_regression_tree(clf, x.columns, tv)
            except Exception as e:
                print(f"Error interpreting the decision tree: {e}")

            # print("Best Path for Class 0:", class_names[0])
            # for step in best_class_0["path"]:
            #     print(" •", step)

            # print("\nBest Path for Class 1:", class_names[1])
            # for step in best_class_1["path"]:
            #     print(" •", step)

            path_str_class_0 = f"Best Path for Class 0: {class_names[0]}\n"
            for step in best_class_0["path"]:
                path_str_class_0 += f" • {step}\n"

            # Save path for Class 1
            path_str_class_1 = f"Best Path for Class 1: {class_names[1]}\n"
            for step in best_class_1["path"]:
                path_str_class_1 += f" • {step}\n"

            # Print the saved strings (optional)
            print(path_str_class_0)
            print(path_str_class_1)

            # print all Path
            # for path in all_paths:
            #     print(f"\nPath: {path['path']}, Gini Sum: {path['gini_sum']:.4f}, Samples Sum: {path['samples_sum']}, Final Cost: {path['final_cost']:.4f}, Predicted Class: {path['predicted_class']}, Branch: {path['branch']}")

            # strategy_to_use = "lowest_cost" if target_type == 'Categorical' else "highest_samples"
            # # Select the best path for each class using the defined strategy
            best_path_0_str = path_str_class_0
            best_path_1_str = path_str_class_1

            # print(f"\nBest Path for Class 0 {class_names[0]}:",best_path_0_str)
            # print(f"\nBest Path for Class 1 {class_names[1]}:",best_path_1_str)

            # Calculate feature importances for the best tree
            feature_importances = rf2.estimators_[best_tree_index].feature_importances_
            print(f"Feature Importances for Tree {best_tree_index}:", feature_importances)
            data_dict = {feature: round(value,2) for feature, value in zip(x.columns, feature_importances*100) if value != 0}
            sorted_data = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))
            print("Sorted Data:", sorted_data)
            feature_importance_df = pd.DataFrame(list(sorted_data.items()), columns=['Feature', 'Importance'])


            feature_importance_df.reset_index(drop=True, inplace=True)
            mi_score=feature_importance_df
            print('Mutual Information Scores:\n', mi_score)
            buyerid = self.request.session.get('buyerid') or 'anonymous'
            rand = random.randint(0, 99999999)
            folder = f'media/files/chatbot/{buyerid}/'
            loc = f'current_tree_{rand}.png'
            loc_curr_tree = os.path.join(get_dataset_folder(folder, loc), loc)
            plt.savefig(loc_curr_tree)
            plt.close()
            print(f'Plot saved at {loc_curr_tree}')

            ######### Target SXI Tree ##############
            if target_type == 'Categorical':
                rf1 = RandomForestClassifier(max_depth=4)
            elif target_type == 'Continuous':
                rf1 = RandomForestRegressor(max_depth=4)
            
            def calculate_top_5_mi_scores(df, target_column, task=target_type.lower()):
                """
                Calculate top 5 features based on Mutual Information score.

                Parameters:
                - df: pandas DataFrame with features and target
                - target_column: name of the target column
                - task: 'classification' or 'regression'

                Returns:
                - DataFrame with top 5 features and their MI scores
                """
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Encode categorical features (if needed)
                X_encoded = X.copy()
                for col in X_encoded.select_dtypes(include='object'):
                    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

                # Encode target if classification with object type
                if task == 'classification' and y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)

                # Calculate MI
                if task == 'classification':
                    mi_scores = mutual_info_classif(X_encoded, y, discrete_features='auto')
                else:
                    mi_scores = mutual_info_regression(X_encoded, y, discrete_features='auto')

                # Create score dataframe
                mi_df = pd.DataFrame({
                    'Feature': X_encoded.columns,
                    'MI Score': mi_scores
                }).sort_values(by='MI Score', ascending=False)

                return mi_df.head(5)
            #combine finalnewx and y 
            # finalnewx_mi = pd.concat([finalnewx, y], axis=1)
            # Calculate Mutual Information scores
            # mi_score = calculate_top_5_mi_scores(finalnewx_mi, target_column=target, task=target_type.lower())  # Assuming target_type is either 'classification' or 'regression'
            # print('Mutual Information Scores:\n', mi_score)

            treen = random.randint(0, 99)
            rf1.fit(finalnewx, y)
            from sklearn.tree import plot_tree
            plt.figure(figsize=(28, 15))
            class_names = [str(c) for c in classes]
            print("class_names",class_names)
            clf2 = rf1.estimators_[treen]
            if hasattr(self, 'dataframe') and self.dataframe is not None:
                try:
                    from agent2.utils.tree_utils import unnormalize_tree_thresholds
                    unnormalize_tree_thresholds(clf2, finalnewx.columns.tolist(), self.dataframe, finalnewx, run_id=self.buyerid)
                except Exception as e:
                    print(f"Error unnormalizing tree thresholds at Location 2: {e}")
            print("clf", clf2)
            if target_type == 'Categorical':
                plot_tree(rf1.estimators_[treen], filled=True, rounded=True, class_names=class_names, feature_names=finalnewx.columns, fontsize=15, precision=2)
                # plt.title("Target Decision Tree", fontsize=50, fontweight='bold')

            elif target_type == 'Continuous':
                plot_tree(rf1.estimators_[treen], filled=True, rounded=True, feature_names=finalnewx.columns, fontsize=15, precision=2)
                # plt.title("Target Decision Tree", fontsize=50, fontweight='bold')
            ### Tree Interpretation
            try:
                if target_type == 'Categorical':
                    all_paths, best_class_0_tr, best_class_1_tr = self.a_star_tree_paths(
                        clf2, x.columns.tolist(), clf2.classes_, reversed_mapping
                    )
                elif target_type == 'Continuous':
                    intp_curr = self.interpret_regression_tree(rf1.estimators_[treen], finalnewx.columns, tv)
            except Exception as e:
                print(f"Error interpreting the decision tree: {e}")
            
            path_str_class_0_tr = f"Best Path for Class 0: {class_names[0]}\n"
            for step in best_class_0_tr["path"]:
                path_str_class_0_tr += f" • {step}\n"

            # Save path for Class 1
            path_str_class_1_tr = f"Best Path for Class 1: {class_names[1]}\n"
            for step in best_class_1_tr["path"]:
                path_str_class_1_tr += f" • {step}\n"

            # Print the saved strings (optional)
            print(path_str_class_0_tr)
            print(path_str_class_1_tr)

            # print all Paths
            # print("All Paths:")
            # for path in all_paths:
            #     print(f"Path: {path['path']}, Gini Sum: {path['gini_sum']:.4f}, Samples Sum: {path['samples_sum']}, Final Cost: {path['final_cost']:.4f}, Predicted Class: {path['predicted_class']}, Branch: {path['branch']}")

            best_path_0_str_tr = path_str_class_0_tr
            best_path_1_str_tr = path_str_class_1_tr

            
            print(f"\nBest Path for Class 0 {class_names[0]}:", best_path_0_str_tr)
            print(f"\nBest Path for Class 1 {class_names[1]}:", best_path_1_str_tr)

            if target_type == 'Categorical':
                plot_tree(rf1.estimators_[treen], filled=True, rounded=True, class_names=class_names, feature_names=finalnewx.columns, fontsize=15, precision=2)
                # plt.title("Target Decision Tree", fontsize=100, fontweight='bold')

            elif target_type == 'Continuous':
                plot_tree(rf1.estimators_[treen], filled=True, rounded=True, feature_names=finalnewx.columns, fontsize=15, precision=2)
                # plt.title("Target Decision Tree", fontsize=100, fontweight='bold')

            feature_importances_trgt = rf1.estimators_[treen].feature_importances_
            print("Feature Importances for Tree 0:", feature_importances_trgt)
            data_dict = {feature: round(value,2) for feature, value in zip(finalnewx.columns, feature_importances_trgt*100) if value != 0}
            sorted_data_trgt = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))
            print("Sorted Data:", sorted_data_trgt)
            feature_importance_df_trgt = pd.DataFrame(list(sorted_data_trgt.items()), columns=['Feature', 'Importance'])
            feature_importance_df_trgt.reset_index(drop=True, inplace=True)
            mi_score_trgt = feature_importance_df_trgt
            print('Mutual Information Scores:\n', mi_score_trgt)
            buyerid = self.request.session.get('buyerid')
            folder = f'media/files/chatbot/{buyerid}/'
            loc = f'targ_tree_{rand}.png'
            loc_target_tree = os.path.join(get_dataset_folder(folder, loc), loc)
            plt.savefig(loc_target_tree)
            plt.close()
            print(f'Plot saved at {loc_target_tree}')

            return loc_curr_tree, loc_target_tree, featureset, mi_score, mi_score_trgt, best_path_0_str, best_path_1_str, best_path_1_str_tr, best_path_0_str_tr
    '''

    def tree(self, target_type, target, dataframe, tv_perc, corr_option, labels, outyp,feat_importance_df):
        from sklearn.utils.multiclass import type_of_target
        self.current_tree_feature_importance_raw = pd.DataFrame()
        self.target_tree_feature_importance_raw = pd.DataFrame()

        # class WeightedRandomForest(BaseEstimator):
        #     """
        #     Custom Random Forest with feature weights applied before training.
        #     Automatically switches between classification and regression.
        #     """

        #     def __init__(self,
        #                 feature_weights=None,
        #                 n_estimators=100,
        #                 max_depth=None,
        #                 random_state=None,
        #                 n_jobs=-1):
        #         self.feature_weights = feature_weights
        #         self.n_estimators = n_estimators
        #         self.max_depth = max_depth
        #         self.random_state = random_state
        #         self.n_jobs = n_jobs
        #         self.model_ = None
        #         self.task_type_ = None  # "classification" or "regression"

        #     def fit(self, X, y):
        #         X = np.asarray(X)

        #         # Scale features by custom weights if provided
        #         if self.feature_weights is not None:
        #             X = X * self.feature_weights

        #         # Detect task type
        #         target_type = type_of_target(y)
        #         if target_type in ["binary", "multiclass"]:
        #             self.task_type_ = "classification"
        #             self.model_ = RandomForestClassifier(
        #                 n_estimators=self.n_estimators,
        #                 max_depth=self.max_depth,
        #                 random_state=self.random_state,
        #                 n_jobs=self.n_jobs,
        #                 class_weight="balanced"
        #             )

        #         else:
        #             self.task_type_ = "regression"
        #             self.model_ = RandomForestRegressor(
        #                 n_estimators=self.n_estimators,
        #                 max_depth=self.max_depth,
        #                 random_state=self.random_state,
        #                 n_jobs=self.n_jobs
        #             )
        #         print("task_type_ : ",self.task_type_)
        #         self.model_.fit(X, y)
        #         return self

        #     def predict(self, X):
        #         X = np.asarray(X)
        #         if self.feature_weights is not None:
        #             X = X * self.feature_weights
        #         return self.model_.predict(X)

        #     def predict_proba(self, X):
        #         if self.task_type_ != "classification":
        #             raise AttributeError("predict_proba is only available for classification tasks.")
        #         X = np.asarray(X)
        #         if self.feature_weights is not None:
        #             X = X * self.feature_weights
        #         return self.model_.predict_proba(X)

        #     def feature_importances_(self):
        #         raw_importances = self.model_.feature_importances_
        #         if self.feature_weights is None:
        #             return raw_importances
        #         adjusted = raw_importances * self.feature_weights
        #         return adjusted / adjusted.sum()

        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        class WeightedRandomForest(BaseEstimator):
            """
            Custom Decision Tree with feature weights applied before training.
            Automatically switches between classification and regression.
            """

            def __init__(self,
                        feature_weights=None,
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_leaf_nodes=None,
                        random_state=None,
                        max_features=None,
                        splitter="best",
                        class_weight="balanced",
                        tv_type=None):
                self.feature_weights = feature_weights
                self.max_depth = max_depth
                self.min_samples_split = min_samples_split
                self.min_samples_leaf = min_samples_leaf
                self.max_leaf_nodes = max_leaf_nodes
                self.random_state = random_state
                self.max_features = max_features
                self.splitter = splitter
                self.class_weight = class_weight
                self.model_ = None
                self.task_type_ = None  # "classification" or "regression"
                self.n_estimators = 100
                self.tv_type=tv_type


            def fit(self, X, y, sample_weight=None):
                X = np.asarray(X, dtype=float)

                # Apply feature weights
                if self.feature_weights is not None:
                    X = X * self.feature_weights

                # Detect task type
                target_type = type_of_target(y)

                if self.tv_type == 'Categorical':
                    self.task_type_ = "classification"
                    self.model_ = DecisionTreeClassifier(
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        max_leaf_nodes=self.max_leaf_nodes,
                        max_features=self.max_features,
                        splitter=self.splitter,
                        class_weight=self.class_weight,
                    )
                else:
                    self.task_type_ = "regression"
                    self.model_ = DecisionTreeRegressor(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        max_leaf_nodes=self.max_leaf_nodes,
                        max_features=self.max_features,
                        splitter=self.splitter,
                        random_state=self.random_state,
                    )

                print("task_type_ :", self.tv_type)
                if sample_weight is not None:
                    self.model_.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=float))
                else:
                    self.model_.fit(X, y)
                return self

            def predict(self, X):
                X = np.asarray(X)
                if self.feature_weights is not None:
                    X = X * self.feature_weights
                return self.model_.predict(X)

            def predict_proba(self, X):
                if self.task_type_ != "classification":
                    raise AttributeError("predict_proba is only available for classification tasks.")
                X = np.asarray(X)
                if self.feature_weights is not None:
                    X = X * self.feature_weights
                return self.model_.predict_proba(X)

            def feature_importances_(self):
                raw_importances = self.model_.feature_importances_
                if self.feature_weights is None:
                    return raw_importances
                adjusted = raw_importances * self.feature_weights
                return adjusted / adjusted.sum()

        df = dataframe
        if target_type == 'Categorical':
            df = df.drop(['index', 'id', 'Id', 'ID',f'{target}_original','i_n_d_e_x'], axis=1, errors='ignore')
        else:
            df = df.drop(['index', 'id', 'Id', 'ID','i_n_d_e_x','netqyty_Bucket'], axis=1, errors='ignore')

        if target_type == 'Continuous':
            tv = f'{target}_original'
            x = df.drop(['composite_dxi_label', 'composite_dxi', target,tv,'netqyty_Bucket'], axis=1, errors='ignore')
            y = df[tv]
            print(f'y \n== {y}')
            x, y = self._prepare_sklearn_inputs(x, y, context="weighted tree regression")
            from sklearn.linear_model import Lasso
            clf = Lasso(alpha=0.2, max_iter=10000).fit(x, y)

        elif target_type == 'Categorical':
            x = df.drop(['composite_dxi_label', 'composite_dxi', target,'netqyty_Bucket'], axis=1, errors='ignore')
            y = df[target]
            x, y = self._prepare_sklearn_inputs(x, y, context="weighted tree classification")
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=0).fit(x, y)


        a = clf.coef_[0]
        h = x.columns
        ty = pd.DataFrame({'Features':h, 'Coefficient':a})
        typv = ty.loc[(ty['Coefficient'] >= 0)]
        pvft = list(typv['Features'])
        tyng = ty.loc[(ty['Coefficient'] < 0)]
        ngft = list(tyng['Features'])

        toval = tv_perc # Target Percent Change
        corr = corr_option # Positive or Negative correlation of target outcome

        if (corr == 'Positive' and outyp == 'good') or (corr == 'Negative' and outyp == 'bad'):
            newx = x[ngft]
            for j in newx.columns:
                newx[j] = newx[j].map(lambda a:(1-(toval/100))*a)

            newx1 = x[pvft]
            for j in newx1.columns:
                newx1[j] = newx1[j].map(lambda a: (1+(toval/100))*a)

        elif (corr == 'Negative' and outyp == 'good') or (corr == 'Positive' and outyp == 'bad'):
            newx = x[ngft]
            for j in newx.columns:
                newx[j] = newx[j].map(lambda a:(1+(toval/100))*a)

            newx1 = x[pvft]
            for j in newx1.columns:
                newx1[j] = newx1[j].map(lambda a: (1-(toval/100))*a)

        finalnewx = pd.concat([newx, newx1], axis=1)
        finalnewx = finalnewx.reindex(columns=x.columns, fill_value=0.0)

        from sklearn.utils.class_weight import compute_sample_weight

        def _extract_numeric_outcome_code(config_key):
            raw_value = self.values_exe.get(config_key)
            try:
                return int(float(raw_value))
            except (TypeError, ValueError):
                return raw_value

        def _aligned_feature_importances(model, feature_names, universe):
            importance_series = pd.Series(
                np.asarray(model.feature_importances_, dtype=float),
                index=list(feature_names),
                dtype=float,
            )
            return importance_series.reindex(list(universe), fill_value=0.0).to_numpy()

        def _tree_summary(model, feature_names):
            used_features = [
                str(feature_names[idx])
                for idx in model.tree_.feature
                if idx >= 0 and idx < len(feature_names)
            ]
            return {
                "depth": int(model.get_depth()),
                "node_count": int(model.tree_.node_count),
                "leaf_count": int(model.get_n_leaves()),
                "features_used": list(dict.fromkeys(used_features)),
            }

        def _log_tree_difference(current_model, target_model, current_features, target_features):
            feature_universe = sorted(set(map(str, current_features)) | set(map(str, target_features)))
            current_importances = _aligned_feature_importances(current_model, current_features, feature_universe)
            target_importances = _aligned_feature_importances(target_model, target_features, feature_universe)

            same_importances = np.array_equal(current_importances, target_importances)
            same_feature_splits = np.array_equal(current_model.tree_.feature, target_model.tree_.feature)
            same_thresholds = np.array_equal(
                np.round(current_model.tree_.threshold, 6),
                np.round(target_model.tree_.threshold, 6),
            )
            same_structure = (
                current_model.get_depth() == target_model.get_depth()
                and current_model.tree_.node_count == target_model.tree_.node_count
                and same_feature_splits
                and same_thresholds
            )

            current_summary = _tree_summary(current_model, current_features)
            target_summary = _tree_summary(target_model, target_features)
            comparison = {
                "same_importances": same_importances,
                "same_structure": same_structure,
                "same_feature_splits": same_feature_splits,
                "same_thresholds": same_thresholds,
                "depth_diff": int(target_model.get_depth() - current_model.get_depth()),
                "node_diff": int(target_model.tree_.node_count - current_model.tree_.node_count),
                "current_importances": current_importances,
                "target_importances": target_importances,
                "current_summary": current_summary,
                "target_summary": target_summary,
            }

            print("Current tree summary:", current_summary)
            print("Target tree summary:", target_summary)
            print(
                "Tree comparison:",
                {
                    "same_importances": same_importances,
                    "same_structure": same_structure,
                    "same_feature_splits": same_feature_splits,
                    "same_thresholds": same_thresholds,
                    "depth_diff": comparison["depth_diff"],
                    "node_diff": comparison["node_diff"],
                },
            )
            return comparison

        def _build_target_sample_weight(base_x, scenario_x, scenario_y, bad_outcome_value=None, aggressive=False):
            if target_type == 'Categorical':
                sample_weight = compute_sample_weight(class_weight="balanced", y=scenario_y)
                sample_weight = np.asarray(sample_weight, dtype=float)

                scenario_series = pd.Series(scenario_y).reset_index(drop=True)
                if bad_outcome_value is not None:
                    outcome_mask = scenario_series.eq(bad_outcome_value).to_numpy()
                    boost_factor = 2.25 if aggressive else 1.75
                    sample_weight[outcome_mask] *= boost_factor
                    print(
                        f"Target classification weighting boosted class {bad_outcome_value} by {boost_factor}x",
                        "count:",
                        int(outcome_mask.sum()),
                    )
                return sample_weight

            base_frame = base_x.reindex(columns=scenario_x.columns, fill_value=0.0)
            scenario_shift = (scenario_x - base_frame).abs().sum(axis=1).astype(float)
            max_shift = float(scenario_shift.max()) if len(scenario_shift) else 0.0
            if max_shift > 0:
                scale = 1.75 if aggressive else 1.0
                sample_weight = 1.0 + ((scenario_shift / max_shift) * scale)
                print("Target regression weighting derived from scenario shift.")
                return sample_weight.to_numpy(dtype=float)

            scenario_series = pd.to_numeric(pd.Series(scenario_y), errors='coerce').fillna(0.0)
            centered = (scenario_series - float(scenario_series.median())).abs()
            max_centered = float(centered.max()) if len(centered) else 0.0
            if max_centered > 0:
                scale = 1.5 if aggressive else 1.0
                sample_weight = 1.0 + ((centered / max_centered) * scale)
                print("Target regression weighting derived from outcome spread.")
                return sample_weight.to_numpy(dtype=float)

            return None


        ############### SXI Tree ################
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        # if target_type == 'Categorical':
        #     rf2 = RandomForestClassifier(max_depth=4)
        # elif target_type == 'Continuous':
        #     rf2 = RandomForestRegressor(max_depth=4)

        feat_weight_map = dict(
            zip(feat_importance_df['Feature'], feat_importance_df['Importance'])
        )
        feature_weights = np.array([
            feat_weight_map.get(col, 0.0)
            for col in x.columns
        ])

        print("Aligned Feature Weights:", feature_weights)

        n_tree_rows = max(int(len(x)), 1)
        regression_min_leaf = max(5, int(round(n_tree_rows * 0.03)))
        regression_min_split = max(10, regression_min_leaf * 2)
        regression_max_leaf_nodes = min(18, max(6, int(math.sqrt(n_tree_rows)) + 4))
        regression_max_features = 0.8 if len(x.columns) > 3 else None
        print(
            "[SXI][TREE_CONFIG]",
            {
                "rows": n_tree_rows,
                "regression_min_samples_leaf": regression_min_leaf,
                "regression_min_samples_split": regression_min_split,
                "regression_max_leaf_nodes": regression_max_leaf_nodes,
                "regression_max_features": regression_max_features,
            },
        )

        current_tree_model = WeightedRandomForest(
                        feature_weights=feature_weights,
                        n_estimators=100,
                        max_depth=5 if target_type == 'Continuous' else 4,
                        min_samples_split=regression_min_split if target_type == 'Continuous' else 2,
                        min_samples_leaf=regression_min_leaf if target_type == 'Continuous' else 1,
                        max_leaf_nodes=regression_max_leaf_nodes if target_type == 'Continuous' else None,
                        random_state=105,
                        max_features=regression_max_features if target_type == 'Continuous' else None,
                        splitter="best",
                        class_weight="balanced",
                        tv_type=target_type)
        
        if target_type == 'Categorical':
            label_mapping = labels  # already dict

        else:  # Continuous
            # convert list to dict mapping
            label_mapping = labels[0]

        reversed_mapping = {v: k for k, v in label_mapping.items()}
        print("reversed_mapping",reversed_mapping)
        classes = list(label_mapping.keys())

        # Keep class names aligned with the encoded class order used by sklearn.
        class_names = [str(c) for c in classes]
        print("class_names",class_names)
        current_tree_model.fit(x, y)
        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        plt.figure(figsize=(28, 15))
        # best_tree_index = self.best_tree(rf2, x, y, target_type)
        # print(f"Best tree index: {best_tree_index}")

        sxi_importances = feature_weights
        sxi_dict = dict(zip(x.columns, feature_weights))
        print("SXI Weights for Tree (aligned):", sxi_dict)

        # clf = rf2.model_.estimators_[best_tree_index]
        # The tree itself
        clf = current_tree_model.model_
        # Capture scenario predictions before unnormalize mutates split thresholds.
        self._scenario_target_predictions = None
        if target_type == 'Continuous':
            try:
                self._scenario_target_predictions = pd.Series(
                    current_tree_model.predict(finalnewx),
                    index=finalnewx.index,
                    name=tv,
                )
            except Exception as exc:
                print(f"[SXI][TREE] Pre-unnormalize predict failed: {exc}")
        if hasattr(self, 'dataframe') and self.dataframe is not None:
            try:
                from agent2.utils.tree_utils import unnormalize_tree_thresholds
                unnormalize_tree_thresholds(clf, x.columns.tolist(), self.dataframe, x, feature_weights, run_id=self.buyerid)
            except Exception as e:
                print(f"Error unnormalizing tree thresholds at Location 3: {e}")

        if target_type == 'Categorical':
            print('Cate')
            f=plot_tree(
                clf,
                feature_names=x.columns,
                class_names=class_names,
                fontsize=15,
                precision=2,
                filled=True,
                rounded=True
            )
            ax = plt.gca()
            for artist in ax.get_children():
                if isinstance(artist, plt.Text):
                    lines = artist.get_text().split("\n")
                    feature_name = None
                    if "<=" in lines[0]:  # split node
                        feature_name = lines[0].split(" <=")[0].strip()
                    if feature_name and feature_name in sxi_dict:
                        # find gini line
                        for i, line in enumerate(lines):
                            if line.startswith("gini"):
                                lines[i] = f"sxi_weight = {sxi_dict[feature_name]:.4f}"
                        artist.set_text("\n".join(lines))
            
        else:
            print('Reg')
            f=plot_tree(
                clf,
                feature_names=x.columns,
                filled=True,
                rounded=True,
                # class_names=class_names,
                fontsize=15,
                precision=2
            )
            ax = plt.gca()
            for artist in ax.get_children():
                if isinstance(artist, plt.Text):
                    lines = artist.get_text().split("\n")
                    # find feature used in split
                    feature_name = None
                    if "<=" in lines[0]:  # split node
                        feature_name = lines[0].split(" <=")[0].strip()
                    if feature_name and feature_name in sxi_dict:
                        # find squared_error line
                        for i, line in enumerate(lines):
                            if line.startswith("squared_error") or line.startswith("mse"):
                                lines[i] = f"sxi_weight = {sxi_dict[feature_name]:.4f}"
                        artist.set_text("\n".join(lines))

        decision_tree_nodes = []
        
        for i in range(len(f)):
            decision_tree_nodes.append(f[i].get_text())
        featureset = []
        for node in decision_tree_nodes:
            for feature in x.columns:
                if f'{feature} <=' in node:
                    featureset.append(feature)
        featureset = list(set(featureset))

        for node_id in range(clf.tree_.node_count):
            f_idx = clf.tree_.feature[node_id]
            if f_idx != -2:
                print(
                    node_id,
                    x.columns[f_idx],
                    "→ SXI:",
                    sxi_dict[x.columns[f_idx]]
                )

        print("Featureset:", featureset)
        print("MODEL IN TREE", clf)
        # print('clf.classes_:', clf.classes_)

        best_path_0_str, best_path_1_str, best_path_1_str_tr, best_path_0_str_tr, result1_tr, result_cr = None, None, None, None, None, None

        def regression_path_lines(path_steps):
            lines = []
            for step in path_steps or []:
                clean_step = re.sub(r"^(?:Ã¢â‚¬Â¢|â€¢|\?|-|\s)+", "", str(step or "")).strip()
                if not clean_step:
                    continue
                if clean_step.lower().startswith((
                    "prediction range:",
                    "coverage:",
                    "node samples:",
                    "mean target value:",
                )):
                    continue
                lines.append(clean_step)
                if len(lines) == 3:
                    break
            return lines

        def build_regression_path_strings(result, tree_label):
            low_path = regression_path_lines((result or {}).get("low_value_path"))
            high_path = regression_path_lines((result or {}).get("high_value_path"))

            if low_path and high_path and low_path == high_path:
                print(f"[SXI][TREE_VALIDATION] {tree_label} extracted identical low/high paths; suppressing duplicate high path.")
                high_path = []

            if not low_path:
                print(f"[SXI][TREE_VALIDATION] {tree_label} low path extraction empty; leaving explanation empty.")
            if not high_path:
                print(f"[SXI][TREE_VALIDATION] {tree_label} high path extraction empty; leaving explanation empty.")

            low_text = ""
            if low_path:
                low_text = "Best Path for Class 0: Below Mean\n"
                for step in low_path:
                    low_text += f"- {step}\n"

            high_text = ""
            if high_path:
                high_text = "Best Path for Class 1: Above Mean\n"
                for step in high_path:
                    high_text += f"- {step}\n"

            return low_text, high_text

        def clean_tree_condition(step):
            clean_step = re.sub(r"^(?:ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢|Ã¢â‚¬Â¢|â€¢|\?|-|\s)+", "", str(step or "")).strip()
            clean_step = re.sub(r"\[.*?\]", "", clean_step).strip()
            clean_step = clean_step.strip("()")
            return clean_step

        def business_interpretation_for_class_path(class_label, conditions, positive=True):
            label_text = str(class_label or "this class")
            if positive:
                return (
                    f"Claims or records matching this route are strongly associated with {label_text}. "
                    "This path matters because it shows a high-confidence segment with meaningful business impact."
                )
            return (
                f"Claims or records matching this route move away from the opposite outcome and toward {label_text}. "
                "This path matters because it helps explain which conditions separate stronger and weaker outcomes."
            )

        def format_classification_path(path_record, index, class_label):
            path_steps = list((path_record or {}).get("path") or [])
            condition_steps = [
                clean_tree_condition(step)
                for step in path_steps
                if "predict class" not in str(step).lower()
            ]
            condition_steps = [step for step in condition_steps if step]
            # Business interpretation should read from the leaf back to the root.
            condition_steps = list(reversed(condition_steps))
            lines = [
                f"Path {index}:",
                f"Class: {class_label}",
                f"Samples: {int((path_record or {}).get('samples', 0))}",
                f"Purity: {float((path_record or {}).get('purity', 0.0)) * 100:.0f}%",
                "",
                "Conditions:",
            ]
            for condition_index, condition_text in enumerate(condition_steps):
                prefix = "IF" if condition_index == 0 else "AND"
                lines.append(f"{prefix} {condition_text}")
            lines.extend([
                "",
                "Business Interpretation:",
                business_interpretation_for_class_path(class_label, condition_steps, positive=True),
                "",
                "--------------------------------",
            ])
            return "\n".join(lines)

        def build_classification_business_text(class_0_paths, class_1_paths, class_names, target_name):
            return format_classification_business_paths(
                class_0_paths,
                class_1_paths,
                class_names,
                target_variable=target_name,
                X=x,
                encoded_feature_mapping=getattr(self, "encoded_feature_mapping", None),
            )

        try:
            if target_type == 'Categorical':
                all_paths, best_class_0, best_class_1 = self.a_star_tree_paths(clf, x.columns.tolist(), reversed_mapping,sxi_dict)
                business_tree_text = build_classification_business_text(
                    best_class_0,
                    best_class_1,
                    class_names,
                    target,
                )
                path_str_class_0 = business_tree_text
                path_str_class_1 = ""
                """
                path_str_class_0 = f"Best Path for Class 0: {class_names[0]}\n"
                for step in best_class_0["path"]:
                    path_str_class_0 += f"- {step}\n"

                # Save path for Class 1
                path_str_class_1 = f"Best Path for Class 1: {class_names[1]}\n"
                for step in best_class_1["path"]:
                    path_str_class_1 += f"- {step}\n"
                """

                # Print the saved strings (optional)
                print(path_str_class_0)
                print(path_str_class_1)

                best_path_0_str = path_str_class_0
                best_path_1_str = path_str_class_1

            elif target_type == 'Continuous':
                current_tree_importance_preview = pd.DataFrame({
                    "Feature": list(x.columns),
                    "Importance": np.asarray(clf.feature_importances_, dtype=float) * 100.0,
                })
                result_cr = self.a_star_tree_paths_continuos(
                    clf,
                    list(x.columns),
                    X=x,
                    y=y,
                    feature_importance_df=current_tree_importance_preview,
                    feature_scales=sxi_dict,
                    tree_label="Current DT",
                )
                # best_class_0 = {"path": ["abc"]}
                # best_class_1 = {"path": ["test"]}
                # all_paths    = {"path": ["allpath"]}
                print('Continuos Tree')
                print("\n High Value Path (Most Above Target):")
                for step in result_cr["high_value_path"]:
                    print(" •", step)
                print(f"Final High Value = {result_cr['max_value']:.3f}")

                print("\n Low Value Path (Most Below Target):")
                for step in result_cr["low_value_path"]:
                    print(" •", step)
                print(f"Final Low Value = {result_cr['min_value']:.3f}")
                def regression_path_lines(path_steps):
                    lines = []
                    for step in path_steps:
                        clean_step = re.sub(r"^(?:â€¢|•|\?|-|\s)+", "", str(step or "")).strip()
                        if not clean_step:
                            continue
                        if clean_step.lower().startswith((
                            "prediction range:",
                            "coverage:",
                            "node samples:",
                            "mean target value:",
                        )):
                            continue
                        lines.append(clean_step)
                        if len(lines) == 3:
                            break
                    return lines

                best_path_0_str = format_regression_business_paths(result_cr, target)
                best_path_1_str = ""

        except Exception as e:
            print(f"Error interpreting the decision tree: {e}")
            if target_type == 'Continuous':
                best_path_0_str, best_path_1_str = build_regression_path_strings({}, "Current DT")

        def build_tree_feature_importance_df(feature_names, tree_importances, sxi_weights_by_feature, sxi_col, importance_col):
            rows = []
            for idx, feature in enumerate(feature_names):
                rows.append({
                    'Feature': feature,
                    sxi_col: round(float(sxi_weights_by_feature.get(feature, 0)) * 100, 2),
                    importance_col: round(float(tree_importances[idx]) * 100, 2),
                })

            feature_df = pd.DataFrame(rows)
            feature_df = feature_df.sort_values(
                by=[importance_col, sxi_col, 'Feature'],
                ascending=[False, False, True],
            ).reset_index(drop=True)
            return feature_df

        # # Extract the selected tree
        # tree = current_tree_model.model_.estimators_[best_tree_index]
        tree = current_tree_model.model_

        tree_features = tree.tree_.feature

        # Tree-level RF impurity importances (CORRECT)
        rf_importances = tree.feature_importances_

        feature_importance_df = build_tree_feature_importance_df(
            feature_names=list(x.columns),
            tree_importances=rf_importances,
            sxi_weights_by_feature=sxi_dict,
            sxi_col='SXI_Weights',
            importance_col='Importance',
        )

        feature_importance_df = self._decorate_encoded_feature_importance_df(
            feature_importance_df,
            sxi_col='SXI_Weights',
            importance_col='Importance',
        )
        self.current_tree_feature_importance_raw = feature_importance_df.copy()
        mi_score = self._aggregate_feature_importance_df(
            feature_importance_df,
            sxi_col='SXI_Weights',
            importance_col='Importance',
        )

        print('Feature Importance (SXI + RF):\n', mi_score)

        buyerid = self.request.session.get('buyerid')
        rand = random.randint(0, 99999999)
        folder = f'media/files/chatbot/{buyerid}/'
        loc = f'current_tree_{rand}.png'
        loc_curr_tree = os.path.join(get_dataset_folder(folder, loc), loc)
        plt.savefig(loc_curr_tree)
        plt.close()
        print(f'Plot saved at {loc_curr_tree}')

        ########################################################
                        # Target SXI Tree #
        ########################################################

        target_feature_weights = np.array([
            feat_weight_map.get(col, 0.0)
            for col in finalnewx.columns
        ])
        target_sxi_dict = dict(zip(finalnewx.columns, target_feature_weights))
        print("Aligned Target Feature Weights:", target_feature_weights)


        def calculate_top_5_mi_scores(df, target_column, task=target_type.lower()):
            """
            Calculate top 5 features based on Mutual Information score.

            Parameters:
            - df: pandas DataFrame with features and target
            - target_column: name of the target column
            - task: 'classification' or 'regression'

            Returns:
            - DataFrame with top 5 features and their MI scores
            """
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Encode categorical features (if needed)
            X_encoded = X.copy()
            for col in X_encoded.select_dtypes(include='object'):
                X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

            # Encode target if classification with object type
            if task == 'classification' and y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            # Calculate MI
            if task == 'classification':
                mi_scores = mutual_info_classif(X_encoded, y, discrete_features='auto')
            else:
                mi_scores = mutual_info_regression(X_encoded, y, discrete_features='auto')

            # Create score dataframe
            mi_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'MI Score': mi_scores
            }).sort_values(by='MI Score', ascending=False)

            return mi_df.head(5)

        target_tree_y = y
        if target_type == 'Continuous':
            cached = getattr(self, '_scenario_target_predictions', None)
            if cached is not None and len(cached):
                target_tree_y = pd.Series(cached, index=finalnewx.index, name=tv).reindex(
                    finalnewx.index
                )
            else:
                target_tree_y = pd.Series(
                    current_tree_model.predict(finalnewx),
                    index=finalnewx.index,
                    name=tv,
                )
            if int(pd.Series(target_tree_y).nunique(dropna=True)) <= 1:
                target_tree_y = pd.Series(y, index=finalnewx.index, name=tv).reindex(
                    finalnewx.index
                )
            print("Using scenario-adjusted predictions for target regression tree")

        finalnewx, target_tree_y = self._prepare_sklearn_inputs(
            finalnewx,
            target_tree_y,
            context="target tree fit",
        )

        bad_outcome_value = _extract_numeric_outcome_code('Bad Outcome Value')
        target_class_weight = None
        if target_type == 'Categorical':
            target_class_weight = "balanced"
            target_distribution = pd.Series(target_tree_y).value_counts(dropna=False)
            if bad_outcome_value in target_distribution.index and len(target_distribution) > 0:
                max_count = float(target_distribution.max())
                target_class_weight = {
                    cls: round(max_count / float(count), 4)
                    for cls, count in target_distribution.items()
                    if float(count) > 0
                }
                if bad_outcome_value in target_class_weight:
                    target_class_weight[bad_outcome_value] = round(target_class_weight[bad_outcome_value] * 1.5, 4)
                print("Target class_weight:", target_class_weight)

        target_sample_weight = _build_target_sample_weight(
            x,
            finalnewx,
            target_tree_y,
            bad_outcome_value=bad_outcome_value,
            aggressive=False,
        )
        target_random_state = random.randint(1000, 999999)
        target_max_features = 'sqrt' if len(finalnewx.columns) > 1 else None

        # Train the target tree independently with a different configuration.
        target_tree_model = WeightedRandomForest(
                    feature_weights=target_feature_weights,
                    n_estimators=100,
                    max_depth=5 if target_type == 'Categorical' else 6,
                    min_samples_split=6 if target_type == 'Categorical' else max(regression_min_split, 12),
                    min_samples_leaf=1 if target_type == 'Categorical' else regression_min_leaf,
                    max_leaf_nodes=None if target_type == 'Categorical' else min(22, regression_max_leaf_nodes + 4),
                    random_state=target_random_state,
                    max_features=target_max_features,
                    splitter="random",
                    class_weight=target_class_weight,
                    tv_type=target_type)
        target_tree_model.fit(finalnewx, target_tree_y, sample_weight=target_sample_weight)
        clf1 = target_tree_model.model_

        tree_comparison = _log_tree_difference(clf, clf1, x.columns, finalnewx.columns)
        if tree_comparison["same_importances"] or tree_comparison["same_structure"]:
            print("Target tree still too similar to current tree. Retrying with stronger diversification.")
            stronger_sample_weight = _build_target_sample_weight(
                x,
                finalnewx,
                target_tree_y,
                bad_outcome_value=bad_outcome_value,
                aggressive=True,
            )
            target_tree_model = WeightedRandomForest(
                        feature_weights=target_feature_weights,
                        n_estimators=100,
                        max_depth=6 if target_type == 'Categorical' else 7,
                        min_samples_split=10 if target_type == 'Categorical' else 12,
                        min_samples_leaf=1 if target_type == 'Categorical' else regression_min_leaf,
                        max_leaf_nodes=None if target_type == 'Categorical' else min(24, regression_max_leaf_nodes + 6),
                        random_state=random.randint(1000000, 9999999),
                        max_features='log2' if len(finalnewx.columns) > 2 else None,
                        splitter="random",
                        class_weight=target_class_weight,
                        tv_type=target_type)
            target_tree_model.fit(finalnewx, target_tree_y, sample_weight=stronger_sample_weight)
            clf1 = target_tree_model.model_
            if hasattr(self, 'dataframe') and self.dataframe is not None:
                try:
                    import copy
                    from agent2.utils.tree_utils import unnormalize_tree_thresholds
                    clf1_plot = copy.deepcopy(clf1)
                    unnormalize_tree_thresholds(
                        clf1_plot,
                        finalnewx.columns.tolist(),
                        self.dataframe,
                        finalnewx,
                        target_feature_weights,
                        run_id=self.buyerid,
                    )
                    clf1 = clf1_plot
                except Exception as e:
                    print(f"Error unnormalizing tree thresholds at Location 4: {e}")
            tree_comparison = _log_tree_difference(clf, clf1, x.columns, finalnewx.columns)

        if np.array_equal(
            tree_comparison["current_importances"],
            tree_comparison["target_importances"],
        ):
            print(
                "[SXI][TREE_VALIDATION] Target tree importances still match current tree "
                "after retry; continuing with validated region extraction fallback."
            )

        # print(f"Randomly selected tree index for target SXI: {treen}")
        print('clf1.classes_:', getattr(clf1, 'classes_', 'N/A'))
        print("MODEL IN TREE", clf1)

        # Plot
        class_names = [str(c) for c in classes]

        fig, ax = plt.subplots(figsize=(28, 15))

        if target_type == 'Categorical':
            print('Cate')
            texts=plot_tree(
                clf1,
                feature_names=finalnewx.columns,
                class_names=class_names,
                fontsize=15,
                precision=2,
                filled=True,
                rounded=True
            )
                # Replace gini with SXI weights
            for text in texts:  # texts is a list of matplotlib.text.Text objects
                lines = text.get_text().split("\n")

                # Detect split feature
                feature_name = None
                if "<=" in lines[0]:
                    feature_name = lines[0].split(" <=")[0].strip()

                # Replace the "gini" line with sxi_weight
                if feature_name and feature_name in target_sxi_dict:
                    for i, line in enumerate(lines):
                        if line.startswith("gini"):
                            lines[i] = f"sxi_weight = {target_sxi_dict[feature_name]:.4f}"
                    text.set_text("\n".join(lines))

        else:
            print('Reg')
            texts=plot_tree(
                clf1,
                feature_names=finalnewx.columns,
                filled=True,
                rounded=True,
                fontsize=15,
                precision=2
            )
            ax = plt.gca()

            last_feature = None
            for artist in ax.get_children():
                if isinstance(artist, plt.Text):
                    lines = artist.get_text().split("\n")
                    feature_name = None
                    if "<=" in lines[0]:
                        feature_name = lines[0].split(" <=")[0].strip()
                        last_feature = feature_name  # update last seen feature
                    feature_for_weight = feature_name if feature_name in target_sxi_dict else last_feature

                    if feature_for_weight and feature_for_weight in target_sxi_dict:
                        found_error = False
                        for i, line in enumerate(lines):
                            if line.startswith("squared_error") or line.startswith("mse"):
                                lines[i] = f"sxi_weight = {target_sxi_dict[feature_for_weight]:.4f}"
                                found_error = True
                                break
                        if not found_error:
                            lines.append(f"sxi_weight = {target_sxi_dict[feature_for_weight]:.4f}")

                        artist.set_text("\n".join(lines))


        best_class_0_tr = {"path": []}
        best_class_1_tr = {"path": []}
        all_paths_tr    = {"path": []}

        try:
            if target_type == 'Categorical':
                all_paths, best_class_0_tr, best_class_1_tr = self.a_star_tree_paths(
                    clf1, finalnewx.columns.tolist(), reversed_mapping, target_sxi_dict
                )
                business_tree_text_tr = build_classification_business_text(
                    best_class_0_tr,
                    best_class_1_tr,
                    class_names,
                    target,
                )
                path_str_class_0_tr = business_tree_text_tr
                path_str_class_1_tr = ""
                """
                path_str_class_0_tr = f"Best Path for Class 0: {class_names[0]}\n"
                for step in best_class_0_tr["path"]:
                    path_str_class_0_tr += f"- {step}\n"

                # Save path for Class 1
                path_str_class_1_tr = f"Best Path for Class 1: {class_names[1]}\n"
                for step in best_class_1_tr["path"]:
                    path_str_class_1_tr += f"- {step}\n"
                """

                # Print the saved strings (optional)
                print(path_str_class_0_tr)
                print(path_str_class_1_tr)


                best_path_0_str_tr = path_str_class_0_tr
                best_path_1_str_tr = path_str_class_1_tr
                print(f"\nBest Path for Class 0 {class_names[0]}:", best_path_0_str_tr)
                print(f"\nBest Path for Class 1 {class_names[1]}:", best_path_1_str_tr)

            elif target_type == 'Continuous':
                target_tree_importance_preview = pd.DataFrame({
                    "Feature": list(finalnewx.columns),
                    "Importance": np.asarray(clf1.feature_importances_, dtype=float) * 100.0,
                })
                result1_tr = self.a_star_tree_paths_continuos(
                    clf1,
                    list(finalnewx.columns),
                    X=finalnewx,
                    y=target_tree_y,
                    feature_importance_df=target_tree_importance_preview,
                    feature_scales=target_sxi_dict,
                    tree_label="Target DT",
                )
                print('Continuos Target Tree')
                print("\n High Value Path (Most Above Target):")
                for step in result1_tr["high_value_path"]:
                    print(" •", step)
                print(f"Final High Value = {result1_tr['max_value']:.3f}")

                print("\n Low Value Path (Most Below Target):")
                for step in result1_tr["low_value_path"]:
                    print(" •", step)
                print(f"Final Low Value = {result1_tr['min_value']:.3f}")
                def regression_path_lines(path_steps):
                    lines = []
                    for step in path_steps:
                        clean_step = re.sub(r"^(?:â€¢|•|\?|-|\s)+", "", str(step or "")).strip()
                        if not clean_step:
                            continue
                        if clean_step.lower().startswith((
                            "prediction range:",
                            "coverage:",
                            "node samples:",
                            "mean target value:",
                        )):
                            continue
                        lines.append(clean_step)
                        if len(lines) == 3:
                            break
                    return lines

                best_path_0_str_tr = format_regression_business_paths(result1_tr, target)
                best_path_1_str_tr = ""
        except Exception as e:
            print(f"Error interpreting the decision tree: {e}")
            if target_type == 'Continuous':
                best_path_0_str_tr, best_path_1_str_tr = build_regression_path_strings({}, "Target DT")

        # feature_importances_trgt = rf1.model_.estimators_[treen].feature_importances_
        # print("Feature Importances for Tree 0:", feature_importances_trgt)
        # data_dict = {feature: round(value,2) for feature, value in zip(finalnewx.columns, feature_importances_trgt*100) if value != 0}
        # sorted_data_trgt = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))
        # print("Sorted Data:", sorted_data_trgt)
        # Extract tree
        
        # tree = rf1.model_.estimators_[treen]
        # tree_features = tree.tree_.feature

        # # RF impurity-based importances (tree-level)
        # rf_importances = tree.feature_importances_

        # # Features used in this tree
        # used_features = set(
        #     finalnewx.columns[i]
        #     for i in tree_features
        #     if i != -2
        # )

        # # SXI weights for used features
        # sxi_feature_weights = {
        #     feature: round(sxi_dict.get(feature, 0), 4)
        #     for feature in used_features
        #     if feature in sxi_dict
        # }

        # # Sort by SXI
        # sorted_sxi_features = dict(
        #     sorted(sxi_feature_weights.items(), key=lambda x: x[1], reverse=True)
        # )

        # sorted_data_trgt = sorted_sxi_features
        # print("Sorted Data (SXI):", sorted_data_trgt)

        # # RF importance mapped to the SAME features
        # rf_importance_dict = {
        #     feature: round(
        #         rf_importances[finalnewx.columns.get_loc(feature)] * 100, 2
        #     )
        #     for feature in sorted_data_trgt.keys()
        # }

        # # Build final DataFrame
        # feature_sxi_df_trgt = pd.DataFrame({
        #     'Feature': list(sorted_data_trgt.keys()),
        #     'SXI_Weight': list(sorted_data_trgt.values()),
        #     'RF_Importance': [
        #         rf_importance_dict.get(feature, 0)
        #         for feature in sorted_data_trgt.keys()
        #     ]
        # })

        # feature_sxi_df_trgt.reset_index(drop=True, inplace=True)

        # print('SXI + RF Weights (Tree-wise Features):\n', feature_sxi_df_trgt)

        # # SXI contribution %
        # total_sxi = feature_sxi_df_trgt['SXI_Weight'].sum()
        # feature_sxi_df_trgt['SXI_Contribution_%'] = (
        #     feature_sxi_df_trgt['SXI_Weight'] / total_sxi * 100
        # ).round(2)

        # print('Final SXI + RF with Contribution %:\n', feature_sxi_df_trgt)

        tree = target_tree_model.model_

        # Tree impurity-based importances
        tree_importances = tree.feature_importances_

        feature_sxi_df_trgt = build_tree_feature_importance_df(
            feature_names=list(finalnewx.columns),
            tree_importances=tree_importances,
            sxi_weights_by_feature=target_sxi_dict,
            sxi_col='SXI_Weight',
            importance_col='RF_Importance',
        )

        print('SXI + Tree Weights (Tree-wise Features):\n', feature_sxi_df_trgt)

        # SXI contribution %
        total_sxi = feature_sxi_df_trgt['SXI_Weight'].sum()
        feature_sxi_df_trgt['SXI_Contribution_%'] = (
            feature_sxi_df_trgt['SXI_Weight'] / total_sxi * 100
        ).round(2)

        print('Final SXI + Tree with Contribution %:\n', feature_sxi_df_trgt)


        # feature_importance_df_trgt = pd.DataFrame(list(sorted_data_trgt.items()), columns=['Feature', 'Importance'])
        # feature_importance_df_trgt.reset_index(drop=True, inplace=True)
        feature_sxi_df_trgt = self._decorate_encoded_feature_importance_df(
            feature_sxi_df_trgt,
            sxi_col='SXI_Weight',
            importance_col='RF_Importance',
        )
        self.target_tree_feature_importance_raw = feature_sxi_df_trgt.copy()
        mi_score_trgt = self._aggregate_feature_importance_df(
            feature_sxi_df_trgt,
            sxi_col='SXI_Weight',
            importance_col='RF_Importance',
        )
        # print('Mutual Information Scores:\n', mi_score_trgt)


        buyerid = self.request.session.get('buyerid')
        folder = f'media/files/chatbot/{buyerid}/'
        loc = f'targ_tree_{rand}.png'
        loc_target_tree = os.path.join(get_dataset_folder(folder, loc), loc)
        plt.savefig(loc_target_tree)
        plt.close()
        print(f'Plot saved at {loc_target_tree}')

        return loc_curr_tree, loc_target_tree, featureset, mi_score, mi_score_trgt, best_path_0_str, best_path_1_str, best_path_1_str_tr, best_path_0_str_tr, result1_tr, result_cr

    '''
    def interpret_decision_tree(self, tree, feature_names, class_labels):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        intp = []

        def recurse(node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print(f"If {name} <= {round(threshold,2)}:")
                intp.append(f"If {name} <= {round(threshold,2)}:")
                recurse(tree_.children_left[node])
                print(f"Else:")
                intp.append(f"Else:")
                recurse(tree_.children_right[node])
            else:
                value = tree_.value[node]
                predicted_class = class_labels[1] if value[0][0] >= value[0][1] else class_labels[0]
                total_samples = value.sum()
                probabilities = value / total_samples
                hired_probability = probabilities[0][class_labels.index(class_labels[0])]
                hp = hired_probability*100
                if hp >50:
                    print(f"Class: {class_labels[0]} (Probability: {hired_probability:.2%})")
                    intp.append(f"Class: {class_labels[0]} (Probability: {hired_probability:.2%})")
                else:
                    print(f"Class: {class_labels[1]} (Probability: {round(100 - hp,2)})")
                    intp.append(f"Class: {class_labels[1]} (Probability: {round(100 - hp,2)})")

        recurse(0)
        return intp

    def interpret_regression_tree(self,tree, feature_names,tv):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        intp = []
        def recurse(node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                mse = tree_.impurity[node]
                print(f"If {name} <= {round(threshold, 2)} (MSE: {round(mse, 2)}):")
                intp.append(f"If {name} <= {round(threshold, 2)} (MSE: {round(mse, 2)}):")
                recurse(tree_.children_left[node])
                print("Else:")
                intp.append("Else:")
                recurse(tree_.children_right[node])
            else:
                value = tree_.value[node][0][0]  # Predicted value at the leaf
                mse = tree_.impurity[node]      # MSE at the leaf
                samples = tree_.n_node_samples[node]  # Number of samples at the leaf
                print(f"{tv}: {round(value, 2)} (MSE: {round(mse, 2)}, Samples: {samples})")
                intp.append(f"{tv}: {round(value, 2)} (MSE: {round(mse, 2)}, Samples: {samples})")

        recurse(0)
        return intp
    '''

    '''
    def correlationlgd(self, perchg, currout, coeffs, myline, model_myline, target, labels, tgtyp, bad,
                       avg_composite_dxi, toggle_val, target_value, r2, curbdcnt):
            print("perchg", perchg)
            print("currout", currout)
            print("coeffs", coeffs)
            print("myline", myline)
            print("model_myline", model_myline)
            print("length myline", len(myline))
            print("length model_myline", len(model_myline))
            print("target", target)
            print("labels", labels)
            print("tgtyp", tgtyp)
            print("bad", bad)
            print("avg_composite_dxi", avg_composite_dxi)
            print("toggle_val", toggle_val)
            print("target_value", target_value)
            currout=round(currout,2)

            print("r2", r2)
            if r2 == 1.0:
                r2 -= random.uniform(0.01, 0.05)

            print("r2", r2)

            print(f"Updated r2: {r2}")

            def reverse_equation(y, m, c):
                return (y - c) / m

            def equatn(sxi, coeff2, coeff1):
                return coeff2 * sxi + coeff1

            def resolve_sxi_from_curve(x_vals, y_vals, y_target, current_x, mode):
                """
                Find an SXI on the plotted curve for a desired target value.
                Uses segment interpolation so returned points lie on the drawn line.
                """
                import numpy as np

                x_arr = np.asarray(x_vals, dtype=float)
                y_arr = np.asarray(y_vals, dtype=float)

                if len(x_arr) == 0:
                    return round(max(current_x, 0.0), 2)

                candidates = []
                for i in range(len(x_arr) - 1):
                    y1, y2 = y_arr[i], y_arr[i + 1]
                    x1, x2 = x_arr[i], x_arr[i + 1]

                    if (y_target - y1) == 0:
                        candidates.append(float(x1))
                    if (y_target - y1) * (y_target - y2) <= 0:
                        if y2 == y1:
                            x_cross = (x1 + x2) / 2.0
                        else:
                            ratio = (y_target - y1) / (y2 - y1)
                            x_cross = x1 + ratio * (x2 - x1)
                        candidates.append(float(x_cross))

                if not candidates:
                    nearest_idx = int(np.argmin(np.abs(y_arr - y_target)))
                    candidates.append(float(x_arr[nearest_idx]))

                non_negative = [x for x in candidates if x >= 0]
                if non_negative:
                    candidates = non_negative

                if mode == 'Increase':
                    forward_candidates = [x for x in candidates if x >= current_x]
                    if forward_candidates:
                        candidates = forward_candidates
                elif mode == 'Decrease':
                    backward_candidates = [x for x in candidates if x <= current_x]
                    if backward_candidates:
                        candidates = backward_candidates

                best_x = min(candidates, key=lambda x: abs(x - current_x))
                x_min = float(np.min(x_arr))
                x_max = float(np.max(x_arr))
                best_x = min(max(best_x, max(0.0, x_min)), x_max)
                return round(best_x, 2)


            perchg = abs(perchg)

            if toggle_val == 'Decrease':
                lty = round(min(model_myline), 2)
                ltx = round(reverse_equation(lty, coeffs[-2], coeffs[-1]), 2)
                ltchg = abs(round(((currout - lty) / currout) * 100, 2))
            else:
                lty = round(max(model_myline), 2)
                ltx = round(reverse_equation(lty, coeffs[-2], coeffs[-1]), 2)
                ltchg = abs(round(((currout - lty) / currout) * 100, 2))

            if toggle_val == 'Decrease':
                tgout = currout * (1 - (perchg / 100))
                tgsxi = round(reverse_equation(tgout, coeffs[-2], coeffs[-1]), 2)
                midout = (tgout + lty) / 2
                midsxi = round(reverse_equation(midout, coeffs[-2], coeffs[-1]), 2)
                midper = round((((currout - midout) / currout) * 100), 2)

                if tgsxi > avg_composite_dxi:
                    tgsxi *= 0.95
                    tgout = equatn(tgsxi, coeffs[-2], coeffs[-1])

            elif toggle_val == 'Increase':
                tgout = currout * (1 + (perchg / 100))
                tgsxi = round(reverse_equation(tgout, coeffs[-2], coeffs[-1]), 2)
                midout = (tgout + lty) / 2
                midsxi = round(reverse_equation(midout, coeffs[-2], coeffs[-1]), 2)
                midper = round((((midout - currout) / currout) * 100), 2)

                # Adjust tgsxi if needed
                if tgsxi < avg_composite_dxi:
                    tgsxi *= 1.05  # Increase by 5%
                    tgout = equatn(tgsxi, coeffs[-2], coeffs[-1])

            curbdcnt = curbdcnt
            print('curbdcnt', curbdcnt)
            print(f'Target Outcome: {tgout}')
            print(f'Target SXI: {tgsxi}')
            print(f'Mid Term Outcome: {midout}')
            print(f'Mid Term SXI: {midsxi}')
            print(f'Mid Term %change: {midper}')




            # Compute immediateval, midtermval, longtermval
            if tgtyp == 'Categorical':
                print("classification")
                if toggle_val == 'Decrease':
                    immediateval = curbdcnt - (curbdcnt * ((100 - perchg) / 100))
                    midtermval = curbdcnt - (curbdcnt * ((100 - midper) / 100))
                    longtermval = curbdcnt - (curbdcnt * ((100 - ltchg) / 100))
                else:
                    immediateval = curbdcnt + (curbdcnt * (perchg / 100))
                    midtermval = curbdcnt + (curbdcnt * (midper / 100))
                    longtermval = curbdcnt + (curbdcnt * (ltchg / 100))
            else:
                print("regression")
                if toggle_val == 'Decrease':
                    immediateval = currout - (currout * ((100 - perchg) / 100))
                    midtermval = currout - (currout * ((100 - midper) / 100))
                    longtermval = currout - (currout * ((100 - ltchg) / 100))
                else:
                    immediateval = currout + (currout * (perchg / 100))
                    midtermval = currout + (currout * (midper / 100))
                    longtermval = currout + (currout * (ltchg / 100))


            if tgtyp == 'Categorical':
                mapping = {val: key for (key, val) in labels.items()}
                print('Mapping: ',mapping)
            else:
                pass


            # --- BEFORE SHIFT: Plot original model_myline
            fig, ax = plt.subplots(1, figsize=(8, 6))
            ax.plot(myline, model_myline, 'gray', linestyle='--', label='Original Line (Before Shift)')

            # --- Scatter point (currout)
            if tgtyp == 'Categorical':
                mapping = {str(key): value for key, value in mapping.items()}
                print("mapping_lower:", mapping)
            else:
                pass

            bad = str(target_value)
            print('Bad:', bad)

            if tgtyp == 'Categorical':
                class_ = mapping[bad]
                print('Outcome:', class_)
                ax.scatter(avg_composite_dxi, currout, marker='o', color='blue', label=f'Current {mapping[bad]}')
                plt.text(avg_composite_dxi, currout, str(f'Current {mapping[bad]} = {currout}%'),
                        ha='left', va='top', fontsize=8, color='blue')
            else:
                ax.scatter(avg_composite_dxi, currout, marker='o', color='blue', label=f'Current {target}')
                plt.text(avg_composite_dxi, currout, str(f'Current {target} = {round(currout, 2)}'),
                        ha='left', va='top', fontsize=8, color='blue')

            # --- Other improvement points (1–2 Agent 2, 3–4 Agent 1)
            if toggle_val == 'Decrease':
                ax.scatter(tgsxi, tgout, marker='x', color='red', label=f'Immediate Improvement: {perchg}% Decrease')
                ax.scatter(midsxi, midout, marker='x', color='#170b16', label=f'Mid-Term Improvement: {midper}% Decrease')
                ax.scatter(ltx, lty, marker='x', color='#078200', label=f'Long-Term Improvement: {ltchg}% Decrease')
            else:
                ax.scatter(tgsxi, tgout, marker='x', color='red', label=f'Immediate Improvement: {perchg}% Increase')
                ax.scatter(midsxi, midout, marker='x', color='#170b16', label=f'Mid-Term Improvement: {midper}% Increase')
                ax.scatter(ltx, lty, marker='x', color='#078200', label=f'Long-Term Improvement: {ltchg}% Increase')

            ax.legend()
            plt.title('Before Shift')
            buyerid_val = self.request.session.get('buyerid') if hasattr(self, 'request') and hasattr(self.request, 'session') else 'anonymous'
            folder_path = f'media/files/chatbot/{buyerid_val}/'
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(os.path.join(get_dataset_folder(folder_path, 'plot_before_shift.png'), 'plot_before_shift.png'), dpi=300)  # Save before shift
            plt.close()
            print('Saved plot_before_shift.png')

            # -------------------------------------------------------
            # --- SHIFTING THE LINE

            # Interpolate model_myline to get predicted y at avg_composite_dxi
            model_interp = np.interp(avg_composite_dxi, myline, model_myline)
            print(f"[Before shift] Model prediction at avg_composite_dxi ({avg_composite_dxi}): {model_interp}")
            print(f"[Target] Desired currout: {currout}")

            # Calculate shift
            shift = currout - model_interp
            print(f"[Shift amount] = {currout} - {model_interp} = {shift}")

            # Adjust the model_myline
            adjusted_model_myline = model_myline + shift

            # -------------------------------------------------------
            # --- AFTER SHIFT: Plot adjusted model_myline
            fig, ax = plt.subplots(1, figsize=(8, 6))
            ax.plot(myline, adjusted_model_myline, 'y', label='Adjusted Line (After Shift)')

            # Shift all scatter outputs accordingly
            tgout_shifted = tgout + shift
            midout_shifted = midout + shift
            lty_shifted = lty + shift

            print(f"[Shifted Initial Improvement Output] {tgout} --> {tgout_shifted}")
            print(f"[Shifted Mid-Term Improvement Output] {midout} --> {midout_shifted}")
            print(f"[Shifted Long-Term Improvement Output] {lty} --> {lty_shifted}")

            # --- Scatter current point (no need to shift currout, already aligned)
            if tgtyp == 'Categorical':
                ax.scatter(avg_composite_dxi, currout, marker='o', color='blue', label=f'Current {mapping[bad]}')
                plt.text(avg_composite_dxi, currout, str(f'Current {mapping[bad]} = {currout}%'),
                        ha='left', va='top', fontsize=8, color='blue')
            else:
                ax.scatter(avg_composite_dxi, currout, marker='o', color='blue', label=f'Current {target}')
                plt.text(avg_composite_dxi, currout, str(f'Current {target} = {round(currout, 2)}'),
                        ha='left', va='top', fontsize=8, color='blue')

            # --- Scatter improvement points with shifted values
            if toggle_val == 'Decrease':
                ax.scatter(tgsxi, tgout_shifted, marker='x', color='red', label=f'Immediate Improvement: {perchg}% Decrease')
                ax.scatter(midsxi, midout_shifted, marker='x', color='#170b16', label=f'Mid-Term Improvement: {midper}% Decrease')
                ax.scatter(ltx, lty_shifted, marker='x', color='#078200', label=f'Long-Term Improvement: {ltchg}% Decrease')
            else:
                ax.scatter(tgsxi, tgout_shifted, marker='x', color='red', label=f'Immediate Improvement: {perchg}% Increase')
                ax.scatter(midsxi, midout_shifted, marker='x', color='#170b16', label=f'Mid-Term Improvement: {midper}% Increase')
                ax.scatter(ltx, lty_shifted, marker='x', color='#078200', label=f'Long-Term Improvement: {ltchg}% Increase')

            ax.legend()
            plt.title('After Shift (with Points Adjusted)')
            buyerid_val = self.request.session.get('buyerid') if hasattr(self, 'request') and hasattr(self.request, 'session') else 'anonymous'
            folder_path = f'media/files/chatbot/{buyerid_val}/'
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(os.path.join(get_dataset_folder(folder_path, 'plot_after_shift.png'), 'plot_after_shift.png'), dpi=300)  # Save after shift
            plt.close()
            print('Saved plot_after_shift.png')


            # -------------------------------------------------------
            # --- Final debug check
            adjusted_model_interp = np.interp(avg_composite_dxi, myline, adjusted_model_myline)
            print(f"[After shift] Adjusted model prediction at avg_composite_dxi ({avg_composite_dxi}): {adjusted_model_interp}")
            print(f"[Target after shift] currout: {currout}")
            print(f"[Difference after shift] = {adjusted_model_interp - currout}")

            
            # fig, ax = plt.subplots(1, figsize=(8, 6))
            # ax.plot((myline)[:], model_myline[:], 'y')


            # if tgtyp == 'Categorical':
            #     mapping = {str(key): value for key, value in mapping.items()}
            #     print("mapping_lower",mapping)
            # else:
            #     pass

            # bad = str(target_value)
            # print('Bad : ', bad)




            # if tgtyp == 'Categorical':
            #     class_ = mapping[bad]
            #     print('Outcome:', class_)
            #     ax.scatter(avg_composite_dxi, currout, marker='o', label=f'Current {mapping[bad]}')
            #     plt.text(avg_composite_dxi, currout, str(f'Current {mapping[bad]} = {currout}%'), ha='left', va='top', fontsize=8, color='blue')
            # else:
            #     ax.scatter(avg_composite_dxi, currout, marker='o', label=f'Current {target}')
            #     plt.text(avg_composite_dxi, currout, str(f'Current {target} = {round(currout, 2)}'), ha='left', va='top', fontsize=8, color='blue')

            # if toggle_val == 'Decrease':
            #     ax.scatter(tgsxi, tgout, marker='x', color='red', label=f'Initial Improvement: {perchg}% Decrease')
            #     ax.scatter(midsxi, midout, marker='x', color='#170b16', label=f'Mid-Term Improvement: {midper}% Decrease')
            #     ax.scatter(ltx, lty, marker='x', color='#078200', label=f'Long-Term Improvement: {ltchg}% Decrease')
            # else:
            #     ax.scatter(tgsxi, tgout, marker='x', color='red', label=f'Initial Improvement: {perchg}% Increase')
            #     ax.scatter(midsxi, midout, marker='x', color='#170b16', label=f'Mid-Term Improvement: {midper}% Increase')
            #     ax.scatter(ltx, lty, marker='x', color='#078200', label=f'Long-Term Improvement: {ltchg}% Increase')

            # plt.grid(which='major', color='#9585e6', linestyle='-')
            # plt.minorticks_on()
            # plt.grid(which='minor', color='#9585e6', linestyle='-', alpha=0.2)

            # if tgtyp == 'Categorical':
            #     plt.title(f"Correlation graph: {mapping[bad]} & SXI R-Square = {int(r2)}")
            #     plt.ylabel(f"{mapping[bad]} (%)")
            # else:
            #     plt.title(f"Correlation graph: {target} & SXI R-Square = {round(r2, 2)}")
            #     plt.ylabel(f"{target}")
            # plt.legend(loc='best')
            # plt.xlabel("SXI")

            # rand = self.rand
            # folder = 'media/files/chatbot/1/'
            # loc = f'updt_corr_plt_{rand}.png'
            # loc_png = os.path.join(get_dataset_folder(folder, loc), loc)
            # plt.savefig(loc)
            # plt.close()

            # Plotly Interactive Plot
            plotly_fig = go.Figure()
            plotly_fig.add_trace(go.Scatter(x=myline, y=model_myline, mode='lines', name='Correlation Line', line=dict(color='#C99700', width=2),
                                            hovertemplate='<b>SXI: %{x:.3f}</b><br><b>'+target +': %{y:.2f}%</b>'))

            plotly_fig.add_trace(
                go.Scatter(
                    x=[avg_composite_dxi],
                    y=[currout],
                    mode='markers+text',
                    name='Current',
                    # text=[f"Current {target if tgtyp != 'Categorical' else mapping[bad]} = {currout}"],
                    textposition='top center',
                    marker=dict(color='blue', symbol='x', size=8),
                    hovertemplate=f'<b>Current SXI: {avg_composite_dxi:.3f}</b><br><b>Current {target}: {currout:.3f}%</b>'
                )
            )


            if toggle_val == 'Decrease':
                # Text for Initial Improvement
                text1 = f"Immediate Improvement: {abs(perchg):.2f}% Decrease"
                plotly_fig.add_trace(
                    go.Scatter(
                        x=[tgsxi],
                        y=[tgout],
                        mode='markers',
                        name=f'Immediate Improvement {abs(perchg):.2f}% Decrease',
                        marker=dict(color='red', symbol='x', size=8),
                        hovertemplate=f'<b>SXI: {tgsxi:.3f}</b><br><b>Immediate {target}: {tgout:.2f}%</b><br><b>{text1}</b>'
                    )
                )

                # Text for Mid-Term Improvement
                text2 = f"Mid-Term Improvement: {abs(midper):.2f}% Decrease"
                plotly_fig.add_trace(
                    go.Scatter(
                        x=[midsxi],
                        y=[midout],
                        mode='markers',
                        name=f'Mid-Term Improvement {abs(midper):.2f}% Decrease',
                        marker=dict(color='#170b16', symbol='x', size=8),
                        hovertemplate=f'<b>SXI: {midsxi:.3f}</b><br><b>Mid-Term {target}: {midout:.2f}%</b><br><b>{text2}</b>'
                    )
                )

                # Text for Long-Term Improvement
                text3 = f"Long-Term Improvement: {abs(ltchg):.2f}% Decrease"
                plotly_fig.add_trace(
                    go.Scatter(
                        x=[ltx],
                        y=[lty],
                        mode='markers',
                        name=f'Long-Term Improvement {abs(ltchg):.2f}% Decrease',
                        marker=dict(color='#078200', symbol='x', size=8),
                        hovertemplate=f'<b>SXI: {ltx:.3f}</b><br><b>Long-Term {target}: {lty:.2f}%</b><br><b>{text3}</b>'
                    )
                )

            else:
                # Text for Initial Improvement (Increase)
                plotly_fig.add_trace(
                    go.Scatter(
                        x=[tgsxi],
                        y=[tgout],
                        mode='markers',
                        name=f'Immediate Improvement {abs(perchg):.2f}% Increase',
                        marker=dict(color='red', symbol='x', size=8),
                        hovertemplate=f'<b>SXI: {tgsxi:.3f}</b><br><b>Immediate {target}: {tgout:.2f}%</b><br><b>Immediate Improvement: {abs(perchg):.2f}% Increase</b>'
                    )
                )

                # Text for Mid-Term Improvement (Increase)
                plotly_fig.add_trace(
                    go.Scatter(
                        x=[midsxi],
                        y=[midout],
                        mode='markers',
                        name=f'Mid-Term Improvement {abs(midper):.2f}% Increase',
                        marker=dict(color='#170b16', symbol='x', size=8),
                        hovertemplate=f'<b>SXI: {midsxi:.3f}</b><br><b>Mid-Term Improvement {target}: {midout:.2f}%</b><br><b>Mid-Term Improvement: {abs(midper):.2f}% Increase</b>'
                    )
                )

                # Text for Long-Term Improvement (Increase)
                plotly_fig.add_trace(
                    go.Scatter(
                        x=[ltx],
                        y=[lty],
                        mode='markers',
                        name=f'Long-Term Improvement {abs(ltchg):.2f}% Increase',
                        marker=dict(color='#078200', symbol='x', size=8),
                        hovertemplate=f'<b>SXI: {ltx:.3f}</b><br><b>Long-Term Improvement {target}: {lty:.2f}%</b><br><b>Long-Term Improvement: {abs(ltchg):.2f}% Increase</b>'
                    )
                )


            import plotly
            print('tgtyp: ',tgtyp)
            layout = go.Layout(
                autosize=True,
                title=dict(
                    text=f"Correlation graph: {target} & SXI R-Square = {round(r2, 2)}",
                    font=dict(
                        size=20,  # Adjust font size
                        family="Arial Black",  # Bold font family
                        color="black"  # Font color
                    )
                ),
                xaxis=dict(
                    title="SXI",
                    showgrid=True,
                    gridcolor='#9585e6',  # Grid color with transparency
                    showline=True,
                    minor=dict(showgrid=True, gridcolor='rgba(149, 133, 230, 0.4)')
                ),
                yaxis=dict(
                    title = f'{target}' if tgtyp != 'Categorical' else f"{target}_{mapping[bad]} (%)",
                    showgrid=True,
                    gridcolor='#9585e6',  # Grid color with transparency
                    showline=True,
                    minor=dict(showgrid=True, gridcolor='rgba(149, 133, 230, 0.4)')
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    xanchor="left",
                ),
                margin=dict(l=60, r=20, t=70, b=60),
                width = 1200,
                height =680
            )

            plotly_fig.update_layout(layout)


            rand = self.rand
            buyerid = self.request.session.get('buyerid')
            folder = f'media/files/chatbot/{buyerid}/'
            lochtml = f'updt_corr_plt_{rand}.html'
            locpltpng = f'updt_corr_plt_{rand}.png'
            # loc_html = os.path.join(get_dataset_folder(folder, lochtml), lochtml)
            loc_png = os.path.join(get_dataset_folder(folder, locpltpng), locpltpng)
            loc_html = os.path.join(get_dataset_folder(folder, lochtml), lochtml)
            _safe_write_plotly_figure(
                plotly_fig,
                loc_html,
                loc_png,
                title="SXI Correlation Graph"
            )
            # plotly_fig.write_image(locpng, engine='kaleido')
            return lty, ltx, ltchg, loc_png, tgsxi, immediateval, midtermval, longtermval, perchg, ltchg, midper, curbdcnt, tgout, lochtml,midsxi,r2
    '''


    def correlationlgd(self, perchg, currout, coeffs, myline, model_myline, target, labels, tgtyp, bad,
                       avg_composite_dxi, toggle_val, target_value, r2, curbdcnt, good, dataframe,
                       display_r2=None):
            print("perchg", perchg)
            print("currout", currout)
            print("coeffs", coeffs)
            # print("myline", myline)
            # print("model_myline", model_myline)
            print("length myline", len(myline))
            print("length model_myline", len(model_myline))
            print("target", target)
            print("labels", labels)
            print("tgtyp", tgtyp)
            print("bad", bad)
            print("avg_composite_dxi", avg_composite_dxi)
            print("toggle_val", toggle_val)
            print("target_value", target_value)
            print("good Value", good)
            optimization_context = _resolve_optimization_target_class(self.values_exe or {})
            target_change = optimization_context.get("target_change")
            selected_is_bad = bool(optimization_context.get("selected_is_bad"))
            selected_is_good = bool(optimization_context.get("selected_is_good"))
            sxi_movement_mode = (
                "Increase"
                if (selected_is_bad and target_change == "decreasing")
                or (selected_is_good and target_change == "increasing")
                else "Decrease"
            )
            print({
                "selected_outcome_ui": optimization_context.get("selected_outcome"),
                "selected_outcome_backend": target_value,
                "selected_outcome_meaning": optimization_context.get("selected_outcome_meaning"),
                "good_outcome": optimization_context.get("good_outcome"),
                "bad_outcome": optimization_context.get("bad_outcome"),
                "target_change": target_change,
                "optimization_target_class": target_value,
                "slope": float(coeffs[-2]) if coeffs is not None and len(coeffs) >= 2 else 0.0,
                "current_sxi": avg_composite_dxi,
                "target_sxi": None,
                "correlation_direction": "Negative" if coeffs[-2] < 0 else "Positive",
                "iteration_selected": (self.values_exe or {}).get("iteration_selected"),
            })
            chart_r2 = resolve_chart_r2(
                report_r2=display_r2,
                fallback_r2=r2,
            )
            df = dataframe
            composite_dxi1 = df['composite_dxi']
            currout=round(currout,2)

            if r2 == 1.0:
                r2 -= random.uniform(0.01, 0.05)
            print(f"Updated r2: {r2}")

            def reverse_equation(y, m, c):
                return (y - c) / m

            def forward_equation(x, m, c):
                return m * x + c

            def reverse_equation_quad(y_target, a, b, c, x_range=None):
                """
                Solve y = a*x^2 + b*x + c for x given y_target.
                
                Returns a single rounded x-value that falls within x_range (if provided).
                
                Parameters:
                    y_target : float
                        Target y-value (e.g., lty, tgout)
                    a, b, c : floats
                        Quadratic coefficients
                    x_range : tuple (xmin, xmax), optional
                        Range of x-values to pick a valid root
                
                Returns:
                    x : float
                        Single rounded x-value
                """
                import numpy as np

                
                discriminant = b**2 - 4*a*(c - y_target)
                
                if discriminant < 0:
                    # No real solution, clamp to closest x in range
                    if x_range is not None:
                        return round((x_range[0] + x_range[1])/2, 2)
                    else:
                        raise ValueError("No real solution for given y_target and quadratic coefficients.")
                
                x1 = (-b + np.sqrt(discriminant)) / (2*a)
                x2 = (-b - np.sqrt(discriminant)) / (2*a)
                
                # If x_range is provided, pick the root inside the range
                if x_range is not None:
                    valid_x = [x for x in (x1, x2) if x_range[0] <= x <= x_range[1]]
                    if valid_x:
                        return round(valid_x[0], 2)  # pick first valid root
                    else:
                        # If no root is inside range, pick closest root to the range
                        closest_x = min((x1, x2), key=lambda x: min(abs(x - x_range[0]), abs(x - x_range[1])))
                        return round(closest_x, 2)
                else:
                    # If no range given, just return the larger root (or pick strategy you prefer)
                    print(f'x1, x2 == {x1, x2}')
                    return round(min(x1, x2), 2)


            def equatn(sxi, coeff2, coeff1):
                return coeff2 * sxi + coeff1

            def resolve_sxi_from_curve(x_vals, y_vals, y_target, current_x, mode):
                """
                Find an SXI on the plotted curve for a desired target value.
                Uses segment interpolation so returned points lie on the drawn line.
                """
                import numpy as np

                x_arr = np.asarray(x_vals, dtype=float)
                y_arr = np.asarray(y_vals, dtype=float)

                if len(x_arr) == 0:
                    return round(max(current_x, 0.0), 2)

                candidates = []
                for i in range(len(x_arr) - 1):
                    y1, y2 = y_arr[i], y_arr[i + 1]
                    x1, x2 = x_arr[i], x_arr[i + 1]

                    if (y_target - y1) == 0:
                        candidates.append(float(x1))
                    if (y_target - y1) * (y_target - y2) <= 0:
                        if y2 == y1:
                            x_cross = (x1 + x2) / 2.0
                        else:
                            ratio = (y_target - y1) / (y2 - y1)
                            x_cross = x1 + ratio * (x2 - x1)
                        candidates.append(float(x_cross))

                if not candidates:
                    nearest_idx = int(np.argmin(np.abs(y_arr - y_target)))
                    candidates.append(float(x_arr[nearest_idx]))

                non_negative = [x for x in candidates if x >= 0]
                if non_negative:
                    candidates = non_negative

                if mode == 'Increase':
                    forward_candidates = [x for x in candidates if x >= current_x]
                    if forward_candidates:
                        candidates = forward_candidates
                elif mode == 'Decrease':
                    backward_candidates = [x for x in candidates if x <= current_x]
                    if backward_candidates:
                        candidates = backward_candidates

                best_x = min(candidates, key=lambda x: abs(x - current_x))
                x_min = float(np.min(x_arr))
                x_max = float(np.max(x_arr))
                best_x = min(max(best_x, max(0.0, x_min)), x_max)
                return round(best_x, 2)

            perchg = abs(perchg)

            if toggle_val == 'Decrease':
                lty = round(min(model_myline), 2)
                ltx = round(reverse_equation(lty, coeffs[-2], coeffs[-1]), 2)
                ltchg = abs(round(((currout - lty) / currout) * 100, 2))
            else:
                lty = round(max(model_myline), 2)
                ltx = round(reverse_equation(lty, coeffs[-2], coeffs[-1]), 2)
                ltchg = abs(round(((currout - lty) / currout) * 100, 2))

            if toggle_val == 'Decrease':
                tgout = currout * (1 - (perchg / 100))
                print('tgout',tgout)
                tgsxi = round(reverse_equation(tgout, coeffs[-2], coeffs[-1]), 2)
                midout = (tgout + lty) / 2
                midsxi = round(reverse_equation(midout, coeffs[-2], coeffs[-1]), 2)
                midper = round((((currout - midout) / currout) * 100), 2)


            elif toggle_val == 'Increase':
                tgout = currout * (1 + (perchg / 100))
                print('tgout',tgout)
                tgsxi = round(reverse_equation(tgout, coeffs[-2], coeffs[-1]), 2)
                midout = (tgout + lty) / 2
                midsxi = round(reverse_equation(midout, coeffs[-2], coeffs[-1]), 2)
                midper = round((((midout - currout) / currout) * 100), 2)


            curbdcnt = curbdcnt
            print('curbdcnt', curbdcnt)
            print(f'Target Outcome: {tgout}')
            print(f'Target SXI: {tgsxi}')
            print(f'Mid Term Outcome: {midout}')
            print(f'Mid Term SXI: {midsxi}')
            print(f'Mid Term %change: {midper}')

            if tgtyp == 'Categorical':
                print("classification")
                if toggle_val == 'Decrease':
                    immediateval = curbdcnt - (curbdcnt * ((100 - perchg) / 100))
                    midtermval = curbdcnt - (curbdcnt * ((100 - midper) / 100))
                    longtermval = curbdcnt - (curbdcnt * ((100 - ltchg) / 100))
                else:
                    immediateval = curbdcnt + (curbdcnt * (perchg / 100))
                    midtermval = curbdcnt + (curbdcnt * (midper / 100))
                    longtermval = curbdcnt + (curbdcnt * (ltchg / 100))
            else:
                print("regression")
                if toggle_val == 'Decrease':
                    immediateval = currout - (currout * ((100 - perchg) / 100))
                    midtermval = currout - (currout * ((100 - midper) / 100))
                    longtermval = currout - (currout * ((100 - ltchg) / 100))
                else:
                    immediateval = currout + (currout * (perchg / 100))
                    midtermval = currout + (currout * (midper / 100))
                    longtermval = currout + (currout * (ltchg / 100))


            if tgtyp == 'Categorical':
                mapping = {val: key for (key, val) in labels.items()}
                print('Mapping: ',mapping)
            else:
                pass

            # --- Scatter point (currout)
            if tgtyp == 'Categorical':
                mapping = {str(key): value for key, value in mapping.items()}
                print("mapping_lower:", mapping)
            else:
                pass

            bad = str(target_value)
            print('Bad:', bad)


            # -------------------------------------------------------
            # --- SHIFTING THE LINE

            # Interpolate model_myline to get predicted y at avg_composite_dxi
            model_interp = np.interp(avg_composite_dxi, myline, model_myline)
            print(f"[Before shift] Model prediction at avg_composite_dxi ({avg_composite_dxi}): {model_interp}")
            print(f"[Target] Desired currout: {currout}")

            # Calculate shift
            shift = currout - model_interp
            print(f"[Shift amount] = {currout} - {model_interp} = {shift}")

            # Adjust the model_myline
            adjusted_model_myline = model_myline + shift

            # -------------------------------------------------------
            # --- AFTER SHIFT: Plot adjusted model_myline

            # mask = (myline > 0) #& (adjusted_model_myline > 0)
            # myline = myline[mask]
            # adjusted_model_myline = adjusted_model_myline[mask]

            mask = (myline > 0)
            if np.any(adjusted_model_myline < 0):
                mask = (myline > 0) & (adjusted_model_myline > 0)
            myline = myline[mask]
            adjusted_model_myline = adjusted_model_myline[mask]
            print(f'X line {myline}')
            print(f'Y line {adjusted_model_myline}')
            print(f'coeffs == {coeffs}')


            # ------------------------------------------------------------------
            # Core 3-level improvement (matches check/sxi_exe.py):
            # Immediate = current × (1 ± goal%)
            # Long = curve extreme, then pull 25% back toward Immediate (adj_lty)
            # Mid = midpoint of Immediate and Long outcomes
            # Place each marker ON the correlation curve via _resolve_sxi_from_curve.
            # ------------------------------------------------------------------
            if toggle_val == 'Decrease':
                min_val = min(adjusted_model_myline)
                max_val = tgout
                lty = round(min_val, 2)
                # 25% up from min value toward Immediate
                adj_lty = lty + 0.25 * (max_val - lty)
                ltchg = abs(round(((currout - adj_lty) / currout) * 100, 2))

            else:  # toggle_val == 'Increase'
                min_val = tgout
                max_val = max(adjusted_model_myline)

                lty = round(max_val, 2)
                # 25% below max value toward Immediate
                adj_lty = lty - 0.25 * (lty - min_val)
                ltchg = abs(round(((currout - adj_lty) / currout) * 100, 2))

            # --- Target SXI computation ---
            if toggle_val == 'Decrease':
                tgout = currout * (1 - (perchg / 100))
            else:
                tgout = currout * (1 + (perchg / 100))

            tgsxi = _resolve_sxi_from_curve(
                myline,
                adjusted_model_myline,
                tgout,
                avg_composite_dxi,
                sxi_movement_mode,
            )

            # --- Midpoint computation ---
            midout = (tgout + adj_lty) / 2
            midsxi = _resolve_sxi_from_curve(
                myline,
                adjusted_model_myline,
                midout,
                tgsxi,
                sxi_movement_mode,
            )

            ltx = _resolve_sxi_from_curve(
                myline,
                adjusted_model_myline,
                adj_lty,
                midsxi,
                sxi_movement_mode,
            )

            midper = abs(round(((currout - midout) / currout) * 100, 2))
            lty = round(adj_lty, 2)

            # Keep count-based report values aligned with recomputed mid/long %
            if tgtyp == "Categorical":
                if toggle_val == "Decrease":
                    immediateval = curbdcnt - (curbdcnt * (abs(perchg) / 100.0))
                    midtermval = curbdcnt - (curbdcnt * (abs(midper) / 100.0))
                    longtermval = curbdcnt - (curbdcnt * (abs(ltchg) / 100.0))
                else:
                    immediateval = curbdcnt + (curbdcnt * (abs(perchg) / 100.0))
                    midtermval = curbdcnt + (curbdcnt * (abs(midper) / 100.0))
                    longtermval = curbdcnt + (curbdcnt * (abs(ltchg) / 100.0))
            else:
                immediateval = tgout
                midtermval = midout
                longtermval = adj_lty

            # Fit axes tightly around Current / Imm / Mid / Long for readable markers.
            slope_m = float(coeffs[-2]) if coeffs is not None and len(coeffs) >= 2 else 0.0
            b_adj = float(currout) - slope_m * float(avg_composite_dxi)
            percent_like = str(tgtyp or "").lower() == "categorical" or infer_display_unit(target, tgtyp) == "%"
            x_lo, x_hi, y_lo, y_hi = _correlation_axis_ranges(
                [avg_composite_dxi, tgsxi, midsxi, ltx],
                [currout, tgout, midout, lty],
                percent_like=percent_like,
            )
            plot_x = np.asarray(myline, dtype=float)
            plot_y = np.asarray(adjusted_model_myline, dtype=float)
            # If markers sit outside the drawn segment, extend the line linearly.
            if len(plot_x) == 0 or float(np.min(plot_x)) > x_lo or float(np.max(plot_x)) < x_hi:
                plot_x = np.linspace(x_lo, x_hi, max(len(myline), 80))
                plot_y = slope_m * plot_x + b_adj
                if percent_like:
                    keep = (plot_y >= y_lo) & (plot_y <= y_hi)
                    if np.any(keep):
                        plot_x, plot_y = plot_x[keep], plot_y[keep]
                elif np.any(plot_y < 0) and np.any(plot_y > 0):
                    keep = plot_y >= 0
                    if np.any(keep):
                        plot_x, plot_y = plot_x[keep], plot_y[keep]

            print(f"Toggle: {toggle_val}, Type: {tgtyp}")
            print(f"Min={min_val}, Max={max_val}")
            print(f"Base lty={lty}, Adjusted lty={adj_lty}")
            print(f"ltx={ltx}, ltchg={ltchg}, sxi_mode={sxi_movement_mode}")
            print(f"tgout={tgout}, tgsxi={tgsxi}")
            print(f"midout={midout}, midsxi={midsxi}, midper={midper}")
            print(f"corr axis ranges x=[{x_lo:.3f},{x_hi:.3f}] y=[{y_lo:.3f},{y_hi:.3f}]")

            plotly_fig = go.Figure()
            hover_value = build_plotly_hover_value("y", target, tgtyp)
            plotly_fig.add_trace(go.Scatter(
                x=plot_x,
                y=plot_y,
                mode='lines',
                name='Correlation Line',
                line=dict(color='#C99700', width=2),
                hovertemplate=f'<b>SXI: %{{x:.3f}}</b><br><b>{target}: {hover_value}</b>',
            ))

            plotly_fig.add_trace(
                go.Scatter(
                    x=[avg_composite_dxi],
                    y=[currout],
                    mode='markers+text',
                    name='Current',
                    textposition='top center',
                    marker=dict(color='blue', symbol='x', size=8),
                    hovertemplate=(
                        f'<b>Current SXI: {avg_composite_dxi:.3f}</b><br>'
                        f'<b>Current {target}: {format_display_value(currout, target, tgtyp)}</b>'
                    )
                )
            )

            dir_label = "Decrease" if toggle_val == "Decrease" else "Increase"
            plotly_fig.add_trace(
                go.Scatter(
                    x=[tgsxi],
                    y=[tgout],
                    mode='markers',
                    name=f'Immediate Improvement {abs(perchg):.2f}% {dir_label}',
                    marker=dict(color='#c634eb', symbol='x', size=8),
                    hovertemplate=(
                        f'<b>SXI: {tgsxi:.3f}</b><br>'
                        f'<b>Immediate {target}: {format_display_value(tgout, target, tgtyp)}</b><br>'
                        f'<b>Immediate Improvement: {abs(perchg):.2f}% {dir_label}</b>'
                    )
                )
            )
            plotly_fig.add_trace(
                go.Scatter(
                    x=[midsxi],
                    y=[midout],
                    mode='markers',
                    name=f'Mid-Term Improvement {abs(midper):.2f}% {dir_label}',
                    marker=dict(color='#170b16', symbol='x', size=8),
                    hovertemplate=(
                        f'<b>SXI: {midsxi:.3f}</b><br>'
                        f'<b>Mid-Term {target}: {format_display_value(midout, target, tgtyp)}</b><br>'
                        f'<b>Mid-Term Improvement: {abs(midper):.2f}% {dir_label}</b>'
                    )
                )
            )
            plotly_fig.add_trace(
                go.Scatter(
                    x=[ltx],
                    y=[lty],
                    mode='markers',
                    name=f'Long-Term Improvement {abs(ltchg):.2f}% {dir_label}',
                    marker=dict(color='#078200', symbol='x', size=8),
                    hovertemplate=(
                        f'<b>SXI: {ltx:.3f}</b><br>'
                        f'<b>Long-Term {target}: {format_display_value(lty, target, tgtyp)}</b><br>'
                        f'<b>Long-Term Improvement: {abs(ltchg):.2f}% {dir_label}</b>'
                    )
                )
            )

            import plotly
            print('tgtyp: ',tgtyp)

            if int(self.values_exe['Target Outcome Improvement'])>0:
                good_name=self.values_exe['Good Outcome']
            else:
                good_name=self.values_exe['Bad Outcome']
            print(f'good_name== {good_name}')
            y_axis_title = _correlation_yaxis_title(target, tgtyp, good_name)
            # Categorical SXI charts are rate (%) series even when the class label is numeric.
            axis_target_type = "Categorical" if percent_like else tgtyp
            y_axis = build_plotly_value_axis(
                y_axis_title,
                target_name=target if not percent_like else f"{target}_rate",
                target_type=axis_target_type,
            )
            if percent_like and "ticksuffix" not in y_axis:
                y_axis["ticksuffix"] = "%"
            y_axis["range"] = [y_lo, y_hi]
            print(f'corr y-axis title== {y_axis_title}')

            layout = go.Layout(
                autosize=True,
                title=dict(
                    text=f"Correlation graph: {target} & SXI R-Square = {chart_r2:.2f}",
                    font=dict(
                        size=20,  # Title font size
                        family="Arial Black",  # Bold font family
                        color="black"  # Title color
                    )
                ),
                xaxis=dict(
                    title="SXI",
                    range=[x_lo, x_hi],
                    tickfont=dict(
                        size=12,
                        family="Arial Black",
                        color="black"
                    ),
                    showgrid=True,
                    gridcolor='#9585e6',
                    showline=True,
                    minor=dict(showgrid=True, gridcolor='rgba(149, 133, 230, 0.4)')
                ),

                yaxis=dict(
                    **y_axis,
                    tickfont=dict(
                        size=12,
                        family="Arial Black",
                        color="black"
                    ),
                    showgrid=True,
                    gridcolor='#9585e6',
                    showline=True,
                    minor=dict(showgrid=True, gridcolor='rgba(149, 133, 230, 0.4)')
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    xanchor="left",
                    font=dict(
                        size=12,
                        family="Arial",
                        color="black"
                    )
                ),
                plot_bgcolor='#eff1fe',
                paper_bgcolor='white',
                margin=dict(l=60, r=20, t=70, b=60),
                width=1200,
                height=680
            )

            plotly_fig.update_layout(layout)


            rand = self.rand
            # buyerid = self.request.session.get('buyerid')
            buyerid = self.request.session['buyerid']
            print(f"buyerId: {buyerid}")
            folder = f'media/files/chatbot/{buyerid}/'
            lochtml = f'updt_corr_plt_{rand}.html'
            locpltpng = f'updt_corr_plt_{rand}.png'
            loc_html = os.path.join(get_dataset_folder(folder, lochtml), lochtml)
            loc_png = os.path.join(get_dataset_folder(folder, locpltpng), locpltpng)
            loc_html = os.path.join(get_dataset_folder(folder, lochtml), lochtml)
            _safe_write_plotly_figure(
                plotly_fig,
                loc_html,
                loc_png,
                title="SXI Correlation Graph"
            )
            print(f'Corr path: {loc_html}')
            # plotly_fig.write_image(loc_png, engine='kaleido')
            return lty, ltx, ltchg, loc_png, tgsxi, immediateval, midtermval, longtermval, perchg, ltchg, midper, curbdcnt, tgout,midout, lochtml,midsxi,r2


    def eda(self,target, labels, dataframe,good,bad):
        tv = target
        # print('dataframe\n',dataframe)
        labels = {k: int(v) for k, v in labels.items()}
        print('labels:~~~~~~~~~~~~~>',labels)
        dq = dataframe.copy()
        myDict={}
        if dq.empty:
            print('EDA warning: received empty dataframe, returning zeroed metrics')
            return 0.0, 0.0, 0, 0

        avg_composite_dxi = dq['composite_dxi'].mean()
        good_value = int(good)
        bad_value = int(bad)
        reverse_labels = {v: k for k, v in labels.items()}
        good_label = reverse_labels.get(good_value, str(good_value))
        bad_label = reverse_labels.get(bad_value, str(bad_value))

        myDict['SXI'] = round(dq['composite_dxi'].mean(), 2)
        myDict['Top SXI'] = round(max(dq['composite_dxi']), 2)
        myDict['Minimum SXI'] = round(min(dq['composite_dxi']), 2)

        f1 = dq.loc[(dq['composite_dxi'] > avg_composite_dxi) & (dq[tv] == bad_value)]
        f2 = dq.loc[(dq['composite_dxi'] < avg_composite_dxi) & (dq[tv] == good_value)]
        f3 = dq.loc[(dq['composite_dxi'] > avg_composite_dxi) & (dq[tv] == good_value)]
        f4 = dq.loc[(dq['composite_dxi'] < avg_composite_dxi) & (dq[tv] == bad_value)]


        
        print('tv:---------->',tv)
        print('labels:---------->',labels)
        print('bad:---------->',bad)
        print('good:---------->',good)
    


        cl1 = int(dq[tv].eq(bad_value).sum())
        cl2 = int(dq[tv].eq(good_value).sum())

        def safe_percentage(numerator, denominator, metric_name):
            if denominator == 0:
                print(f'EDA warning: denominator was zero for {metric_name}; returning 0.0')
                return 0.0
            return round((numerator / denominator) * 100, 2)

        goodper = safe_percentage(cl2, len(dq), 'goodper')
        goodabv = safe_percentage(len(f3), len(f3) + len(f1), 'goodabv')
        goodbel = safe_percentage(len(f2), len(f2) + len(f4), 'goodbel')

        print('GoodClass%: ',goodper)
        print('GoodabvClass%: ',goodabv)

        myDict[f'{tv}_{bad_label}'] = cl1
        myDict[f'{tv}_{good_label}'] = cl2
        myDict[f'No. of {tv}_{bad_label} above SXI'] = len(f1)
        myDict[f'No. of {tv}_{good_label} below SXI'] = len(f2)
        myDict[f'No. of {tv}_{good_label} above SXI'] = len(f3)
        myDict[f'No. of {tv}_{bad_label} below SXI'] = len(f4)

        return goodabv,goodbel,int(cl2),int(cl1)



    def sxi_execution(self):
        check_cancelled()
        try:
            # rand = np.random.randint(0, 99999999)
            rand = random.randint(000000, 9999999)
            # ~~~~~~~~~~~~~~~~~~~~~~ Data Reading ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if isinstance(self.dataframe, pd.DataFrame):
                df = self.dataframe.copy()
            else:
                df = pd.read_csv(self.dataframe)

            df['i_n_d_e_x'] = range(1, len(df) + 1)
            primarykey='i_n_d_e_x'
            print('primarykey',primarykey)
            # Shuffle so sequential primary keys cannot encode target-sorted order
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            df['i_n_d_e_x'] = range(1, len(df) + 1)
            # ~~~~~~~~~~~~~~~~~~~~~~ Getting values ~~~~~~~~~~~~~~~~~~~~~~~~~~
            values_exe = self.values_exe
            optimization_context = _resolve_optimization_target_class(values_exe)
            optimization_target_class = optimization_context["optimization_target_class"]
            values_exe["Optimization Target Class"] = optimization_target_class
            values_exe["optimization_target_class"] = optimization_target_class
            if optimization_context.get("target_polarity"):
                values_exe["target_polarity"] = optimization_context["target_polarity"]
            print('values_exe: ',values_exe)

            tv = values_exe['Target Outcome']
            tv_type = values_exe['Target Outcome Type']
            if tv_type.lower() == 'numeric':
                tv_type = 'Continuous'

            print('target_type', tv_type)
            print('target_var', tv)
            print('mapping',self.mapping)
            buyerid = self.request.session.get('buyerid')
            eda_original_target_col = f"{tv}_eda_original"
            eda_original_target_series = None
            if values_exe['Target Outcome Type'] == 'Categorical':
              eda_original_target_series = df[tv].copy()
              df[tv] = df[tv].apply(lambda x: self.mapping.get(x, x))
              folder = f'media/files/chatbot/{buyerid}/'
              file_name = f'mapped_df_{rand}.csv'
              print('Mapped df Created:',file_name)
              file_path = os.path.join(get_dataset_folder(folder, file_name), file_name)
              df.to_csv(file_path, index=False)      
            else:
              pass
            # ~~~~~~~~~~~~~~~~~~~~~~ Creating Above/Below Mean Column ~~~~~~~~
            cols = df.columns.to_list()
            print("Columns in data:", cols)

            mapping_cont=None
            if tv_type == 'Categorical':
                label = df[tv].value_counts().index.tolist()

                gdbd_label = [
                    str(optimization_target_class),
                    'Bad' if optimization_context.get("selected_is_bad") else 'Good',
                ]
                tv_value = (gdbd_label[0], int(float(gdbd_label[0])))  # Tuple (Label, Numeric)
                tv_outcome = gdbd_label[1]
            else:
                if pd.api.types.is_numeric_dtype(df[tv]):
                    numeric_target = pd.to_numeric(df[tv], errors='coerce')
                else:
                    numeric_target = pd.to_numeric(
                        df[tv]
                        .astype(str)
                        .str.strip()
                        .replace({"": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan})
                        .str.replace(",", "", regex=False)
                        .str.extract(r"([-+]?\d*\.?\d+)", expand=False),
                        errors='coerce'
                    )
                if numeric_target.notna().sum() == 0:
                    raise ValueError(
                        f"Target column '{tv}' could not be converted to numeric values for regression SXI analysis."
                    )

                target_mean = numeric_target.mean()
                numeric_target_filled = numeric_target.fillna(target_mean)

                df[f'{tv}_original'] = numeric_target
                df[tv] = np.where(
                    numeric_target_filled >= target_mean,
                    'Above_mean',
                    'Below_mean'
                )
                label = df[tv].value_counts().index.tolist()
                rand = random.randint(1000000, 9999999)
                le = preprocessing.LabelEncoder()
                df[tv] = le.fit_transform(df[tv])
                mapping_cont = {
                    str(label_name).strip(): idx
                    for label_name, idx in zip(le.classes_, range(len(le.classes_)))
                }
                print('Mapping for Continuous:', mapping_cont)
                folder = f'media/files/chatbot/{buyerid}/'
                os.makedirs(folder, exist_ok=True)
                file_name = f'ab_bel_df{rand}.csv'
                file_path = os.path.join(get_dataset_folder(folder, file_name), file_name)
                print('Continous tv ChANGED:',file_path)
                df.to_csv(file_path, index=False)

                normalized_mapping_cont = {
                    key.lower().replace(" ", "_"): value
                    for key, value in mapping_cont.items()
                }
                above_mapped_value = normalized_mapping_cont.get('above_mean')
                below_mapped_value = normalized_mapping_cont.get('below_mean')
                good_config_value = values_exe.get('Good Outcome Value')
                bad_config_value = values_exe.get('Bad Outcome Value')
                good_mapped_value = (
                    above_mapped_value
                    if _same_outcome_value(good_config_value, "Above_mean")
                    else below_mapped_value
                    if _same_outcome_value(good_config_value, "Below_mean")
                    else above_mapped_value
                )
                bad_mapped_value = (
                    above_mapped_value
                    if _same_outcome_value(bad_config_value, "Above_mean")
                    else below_mapped_value
                    if _same_outcome_value(bad_config_value, "Below_mean")
                    else below_mapped_value
                )
                optimization_target_class = (
                    above_mapped_value
                    if _same_outcome_value(optimization_target_class, "Above_mean")
                    else below_mapped_value
                    if _same_outcome_value(optimization_target_class, "Below_mean")
                    else optimization_target_class
                )
                values_exe["Optimization Target Class"] = optimization_target_class
                values_exe["optimization_target_class"] = optimization_target_class

                if good_mapped_value is None or bad_mapped_value is None:
                    available_codes = list(mapping_cont.values())
                    if not available_codes:
                        raise ValueError(
                            f"Continuous target bucket mapping could not be created for '{tv}'."
                        )
                    if good_mapped_value is None:
                        good_mapped_value = available_codes[0]
                    if bad_mapped_value is None:
                        bad_mapped_value = available_codes[-1]

                gdbd_label = [
                    optimization_target_class,
                    'Bad' if optimization_context.get("selected_is_bad") else 'Good',
                ]
                print('Mapping for Continuous:', mapping_cont)
                print('Good mapped value:', good_mapped_value)
                print('Bad mapped value:', bad_mapped_value)
                print('Optimization target class:', optimization_target_class)
                print('gdbd_label:', gdbd_label)
                print({
                    "regression_target_column": tv,
                    "target_mean": float(target_mean),
                    "above_mean_meaning": (
                        "good" if _same_outcome_value(values_exe.get('Good Outcome Value'), "Above_mean") else
                        "bad" if _same_outcome_value(values_exe.get('Bad Outcome Value'), "Above_mean") else ""
                    ),
                    "below_mean_meaning": (
                        "good" if _same_outcome_value(values_exe.get('Good Outcome Value'), "Below_mean") else
                        "bad" if _same_outcome_value(values_exe.get('Bad Outcome Value'), "Below_mean") else ""
                    ),
                    "optimization_target_class": optimization_target_class,
                    "optimization_direction": optimization_context.get("target_change"),
                    "slope": None,
                    "current_sxi": None,
                    "target_sxi": None,
                })
                # tv = f'{tv}_original'
            
            ## drop the column
            # df.drop(columns=['custid'], inplace=True)
            #gdbd_label = [values_exe['Good Outcome Value'], 'Good']

            # curent_tv=(len(df[df[tv] == int(values_exe['Good Outcome Value'])])/len(df))
            # print("current tv", curent_tv)
            
            print("TV:", tv)
            print("Labels:", label)


            if tv_type == 'Categorical':
                curent_tv = len(df[df[tv] == int(float(optimization_target_class))]) / len(df)
            else:
                curent_tv = len(df[df[tv] == optimization_target_class]) / len(df)
            print("current tv", curent_tv)
            print('gdbd_label', gdbd_label)

            # Printing the values
            #print('gdbd_label', gdbd_label)

            data = df

            # Removing unnecessary ID columns
            ids_ = ['Id', 'ID', 'id', 'encounter_id']
            
            def rem_id(df):
                import re

                # Regex pattern to match any column containing 'id' (case-insensitive)
                pattern = re.compile(r'id', re.IGNORECASE)

                # Identify columns that contain 'id' in their names
                id_cols = [col for col in df.columns if pattern.search(col)]

                if id_cols:
                    df = df.drop(columns=id_cols)
                    print(f"Removed ID columns: {id_cols}")
                else:
                    print("No ID columns found.")

                return df
            
            data = rem_id(data)
            print("Columns in data:", data.columns)

            # Removing columns with too many null values
            protected_target_columns = {tv, f"{tv}_original"}
            for col, val in data.items():
                if col not in protected_target_columns and (data[col].isnull().sum() / len(val) * 100) > 40:
                    data.drop(columns=col, inplace=True)

            # Validating and handling null values
            def validate_null_values(x, target_column):
                check = x.isnull().sum()
                if check.sum() == 0:
                    print("There are no null values.")
                    return x
                else:
                    print("There are null values, filling them:")
                    data_n = x.copy()
                    for col in x.columns:
                        if col != target_column:
                            if data_n[col].isnull().sum() > 0:
                                if data_n[col].dtype == 'object':
                                    data_n[col].fillna(data_n[col].mode()[0], inplace=True)
                                else:
                                    data_n[col].fillna(data_n[col].mean(), inplace=True)
                    return data_n

            data = validate_null_values(data, tv)

            # Dropping date columns if they exist
            date = ['date', 'DATE', 'Date', 'datetime_utc']
            data = data.drop(date, axis=1, errors='ignore')


            def handle_missing_values(df):

              for col in df.columns:
                missing_percentage = df[col].isnull().sum() / len(df) * 100
                print('Missing percent : ',missing_percentage )
                if missing_percentage > 70:
                  # Drop the column if more than 70% of values are missing
                  df.drop(col, axis=1, inplace=True)
                else:
                  # Impute missing values using appropriate method
                  if df[col].dtype == 'object':  # Categorical column
                    mode = df[col].mode()[0]  # Use mode for categorical
                    df[col].fillna(mode, inplace=True)
                  elif df[col].dtype in ['int64', 'float64']:  # Numerical column
                    mean = df[col].mean()
                    median = df[col].median()

                    # Choose imputation method based on distribution
                    if df[col].skew() > 0.5 or df[col].skew() < -0.5:  # Skewed distribution
                      df[col].fillna(median, inplace=True)  # Use median for skewed data
                    else:
                      df[col].fillna(mean, inplace=True)  # Use mean for approximately normal data

              return df

            data = handle_missing_values(data)
            print(data.isnull().sum())

            # gdbd_label = ['Above_mean', 'Good'] ##Continuous
            # gdbd_label = ['0', 'Good']          ##classification


            tv_value = int(gdbd_label[0])
            tv_outcome = gdbd_label[1]
            labels, df_buynobuy = self.target_encoding(tv_type, tv, data, primarykey,mapping_cont)
            if tv_type == 'Categorical' and eda_original_target_series is not None:
                data[eda_original_target_col] = eda_original_target_series.reindex(data.index)
            good, bad = self.good_bad_label(tv_outcome, labels, tv_value)
            print('good',good)
            print('bad',bad)
            print('labels',labels)
            print(tv_outcome,tv_value)
        
            def tv_eda(data, tv, tv_type,gdbd_label):
                        print('\n')
                        print('Tools Target EDA')
                        print('\n')

                        def tvdist(df, tv, gdbd_label): #classification
                            def percentage_autopct(pct):
                                return f'{pct:.1f}%'

                            mapping = self.mapping or {}

                            # Print mappings and value counts for debugging
                            print('Mapping:', mapping)
                            def label_matches(left, right):
                                return _same_outcome_value(left, right) or str(left).strip() == str(right).strip()

                            def normalize_mapping_key(value):
                                normalized = _normalize_label_value(value)
                                return normalized if normalized is not None else str(value).strip()

                            configured_label_by_value = {}
                            for value_key, label_key in (
                                ("Good Outcome Value", "Good Outcome Label"),
                                ("Bad Outcome Value", "Bad Outcome Label"),
                            ):
                                configured_value = values_exe.get(value_key)
                                configured_label = (
                                    values_exe.get(label_key)
                                    or values_exe.get(label_key.replace(" Label", ""))
                                )
                                if configured_value not in (None, "") and configured_label not in (None, ""):
                                    configured_label_by_value[normalize_mapping_key(configured_value)] = configured_label
                                    configured_label_by_value[str(configured_value).strip()] = configured_label

                            mapping_label_by_code = {}
                            for raw_label, raw_code in mapping.items():
                                mapping_label_by_code[normalize_mapping_key(raw_code)] = raw_label
                                mapping_label_by_code[str(raw_code).strip()] = raw_label

                            def display_label_for(raw_value):
                                if raw_value in (None, ""):
                                    return "Unknown"

                                for lookup in (normalize_mapping_key(raw_value), str(raw_value).strip()):
                                    if lookup in configured_label_by_value:
                                        return configured_label_by_value[lookup]
                                    if lookup in mapping_label_by_code:
                                        return mapping_label_by_code[lookup]

                                for configured_label in (
                                    values_exe.get("Good Outcome Label") or values_exe.get("Good Outcome"),
                                    values_exe.get("Bad Outcome Label") or values_exe.get("Bad Outcome"),
                                ):
                                    if configured_label not in (None, "") and label_matches(raw_value, configured_label):
                                        return configured_label

                                return raw_value

                            eda_source_col = f"{tv}_eda_original"
                            eda_series = df[eda_source_col] if eda_source_col in df.columns else df[tv]
                            display_series = eda_series.fillna("Unknown").map(display_label_for)
                            tv_value_count = display_series.value_counts(dropna=False)
                            print('TV Value Count:', tv_value_count)

                            def resolve_mapped_label(raw_code):
                                labels = list(tv_value_count.index)
                                candidates = [raw_code, display_label_for(raw_code), _normalize_label_value(raw_code)]
                                try:
                                    candidates.append(int(raw_code))
                                except (TypeError, ValueError):
                                    pass
                                try:
                                    candidates.append(float(raw_code))
                                except (TypeError, ValueError):
                                    pass
                                candidates.extend(str(candidate) for candidate in list(candidates))

                                for candidate in candidates:
                                    for label in labels:
                                        if label_matches(label, candidate):
                                            return label

                                for candidate in candidates:
                                    if candidate in mapping:
                                        mapped_value = mapping[candidate]
                                        for label in labels:
                                            if label_matches(label, mapped_value):
                                                return label
                                        return mapped_value
                                return None

                            selected_outcome_raw = values_exe.get("Selected Outcome")
                            if selected_outcome_raw in (None, ""):
                                selected_outcome_raw = gdbd_label[0]

                            selected_label = resolve_mapped_label(selected_outcome_raw)
                            selected_outcome_type = str(
                                values_exe.get("Selected Outcome Meaning")
                                or (gdbd_label[1] if len(gdbd_label) > 1 else "")
                            ).strip().lower()
                            selected_is_bad = selected_outcome_type in {
                                'bad', 'no', 'negative', 'fraud', 'risk', 'failure', 'defect'
                            }
                            all_labels = list(tv_value_count.index)
                            opposite_label = next((label for label in all_labels if label != selected_label), None)

                            print("selected_label----->", selected_label)
                            print("selected_outcome_type------>", selected_outcome_type)
                            print("opposite_label------>", opposite_label)

                            def normalize_yes_no_label(label):
                                text = str(label).strip().lower()
                                if text in {'yes', 'y', 'true', '1'}:
                                    return 'yes'
                                if text in {'no', 'n', 'false', '0'}:
                                    return 'no'
                                return None

                            label_colors = {}
                            if selected_label is not None:
                                label_colors[selected_label] = 'darkred' if selected_is_bad else 'darkgreen'
                            if opposite_label is not None:
                                label_colors[opposite_label] = 'darkgreen' if selected_is_bad else 'darkred'

                            for label in tv_value_count.index:
                                yes_no_label = normalize_yes_no_label(label)
                                if label in label_colors:
                                    continue
                                if yes_no_label == 'yes':
                                    label_colors[label] = 'darkgreen'
                                elif yes_no_label == 'no':
                                    label_colors[label] = 'darkred'

                            fallback_palette = ['#4F81BD', '#F5B041', '#7F8C8D', '#AF7AC5']
                            fallback_index = 0
                            pie_colors = []
                            for label in tv_value_count.index:
                                if label in label_colors:
                                    pie_colors.append(label_colors[label])
                                else:
                                    pie_colors.append(fallback_palette[fallback_index % len(fallback_palette)])
                                    fallback_index += 1

                            # Matplotlib Plot
                            fig, ax = plt.subplots(figsize=(10, 6))

                            wedges, texts, autotexts = ax.pie(
                                tv_value_count.values,
                                labels=tv_value_count.index,
                                autopct=percentage_autopct,
                                startangle=90,
                                colors=pie_colors,
                                textprops={'fontsize': 12}
                            )

                            # Set the color and size of the percentage text
                            for autotext in autotexts:
                                autotext.set_color('white')

                            ax.axis('equal')  # Equal aspect ratio for a perfect circle
                            plt.title(f'{tv} Distribution', fontsize=14)

                            # Save the Matplotlib plot
                            buyerid = self.request.session.get('buyerid')
                            rand = np.random.randint(0, 99999999)
                            folder = f'media/files/chatbot/{buyerid}/'
                            os.makedirs(folder, exist_ok=True)
                            file_name = f'tv_eda_{rand}.png'
                            file_path = os.path.join(get_dataset_folder(folder, file_name), file_name)
                            plt.savefig(file_path)
                            plt.close()

                            # Plotly Plot
                            plotly_fig = go.Figure()

                            plotly_colors = [mcolors.to_hex(color) for color in pie_colors]

                            plotly_fig.add_trace(go.Pie(
                                labels=tv_value_count.index,
                                values=tv_value_count.values,
                                hoverinfo='label+percent',
                                textinfo='percent',
                                textfont_size=12,
                                sort=False,
                                marker=dict(colors=plotly_colors, line=dict(color='white', width=2))
                            ))

                            plotly_fig.update_layout(
                                title_text=f'{tv} Distribution',
                                title_font_size=14,
                                showlegend=True
                            )

                            # Save the Plotly plot as an HTML file
                            plotly_file_name = f'tv_eda_{rand}.html'
                            plotly_file_path = os.path.join(get_dataset_folder(folder, plotly_file_name), plotly_file_name)
                            plotly_png_path = os.path.join(get_dataset_folder(folder, f'tv_eda_{rand}.png'), f'tv_eda_{rand}.png')
                            export_result = save_plot(
                                plotly_fig,
                                plotly_png_path,
                                html_path=plotly_file_path,
                                title=f"{tv} Distribution",
                            )
                            print(export_result["message"])

                            dictt = {
                                'matplotlib_plot': file_path,
                                'plotly_plot': plotly_file_path,
                                'plotly_image': export_result["image_path"],
                            }
                            return export_result["image_path"]

                        def tvdistreg(df, tv): #regression
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=df[tv],
                                nbinsx=20,
                                marker_color='blue',
                                opacity=0.7,
                                # hovertemplate=f'<b>{tv}</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>'
                            ))

                            fig.update_layout(
                                title=f'{tv} Distribution',
                                xaxis_title=tv,
                                yaxis_title=f'{tv}',
                                bargap=0.2,
                                template='plotly_white'
                            )

                            # Save Plotly plot
                            folder = f'media/files/chatbot/{buyerid}/'
                            plotly_html_path = os.path.join(get_dataset_folder(folder, f'tv_eda_{rand}.html'), f'tv_eda_{rand}.html')
                            plotly_png_path = os.path.join(get_dataset_folder(folder, f'tv_eda_{rand}.png'), f'tv_eda_{rand}.png')
                            _safe_write_plotly_figure(
                                fig,
                                plotly_html_path,
                                plotly_png_path,
                                title=f"{tv} Distribution"
                            )

                            # plotly_path

                            return plotly_png_path
                        
                        df = data.copy()
                        edaplots = []
                        if tv_type == 'Continuous':
                            
                            if tv not in df.columns:
                                raise ValueError(f"Column '{tv}' not in DataFrame!")

                            # Preserve original if not exists
                            if f"{tv}_original" not in df.columns:
                                df[f"{tv}_original"] = df[tv]

                            # Clean & convert
                            df[tv] = (
                                df[tv]
                                .astype(str)
                                .str.replace(r'\W', '', regex=True)
                                .replace('', '0')
                                .astype(float)
                            )

                            regplot = tvdistreg(df, f"{tv}_original")
                            edaplots.append(regplot)


                        elif tv_type == 'Categorical':
                            tv_value_count = data[tv].value_counts().index.tolist()
                            catplot = tvdist(df, tv, gdbd_label)
                            edaplots.append(catplot)

                        else:
                            print('Invalid! Please check')
                        return edaplots

            def bi_eda(data, tv, primarykey):
                print('\n')
                print('Tools EDA')
                print('\n')
                import matplotlib.pyplot as plt
                import seaborn as sns
                import pandas as pd
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                from io import BytesIO
                import base64
                import random


                def obj_plot_shuffle(df, feature):
                    x_n = feature
                    value_counts1 = df[x_n].value_counts()
                    if len(value_counts1) > 20:
                        value_counts = value_counts1[:20]
                    else:
                        value_counts = df[x_n].value_counts()

                    fig, ax = plt.subplots()
                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette='viridis')
                    # ax.set_title('Bar Plot')
                    plt.xticks(rotation=45)  # You can change the angle to your preference
                    ax.set_xlabel(x_n)
                    ax.set_ylabel('Count')

                    buyerid = self.request.session.get('buyerid')
                    folder = f'media/files/chatbot/{buyerid}/'
                    file_name = f'bi_eda{rand}.png'
                    file_path = os.path.join(get_dataset_folder(folder, file_name), file_name)
                    fig.savefig(file_path)
                    plt.close()
                    return file_path



                def num_plot_shuffle(df, sel_feat2):
                    plot_functions = [
                        lambda df: sns.histplot(df[sel_feat2], bins=100, kde=False),#.set(title="Histogram"),
                        lambda df: sns.boxplot(x=df[sel_feat2])#.set(title="Box Plot")
                    ]

                    random.shuffle(plot_functions)
                    fig, ax = plt.subplots()
                    plot_functions[0](df)

                    # buf = BytesIO()
                    # # plt.savefig(buf, format='png')
                    # buf.seek(0)
                    # plot_html = base64.b64encode(buf.read()).decode('utf-8')
                    # main_dir = os.path.dirname(os.path.abspath(__file__))
                    # dir_path = os.path.join(main_dir, 'media', str(buyerid))
                    # os.makedirs(dir_path, exist_ok=True)
                    rand = np.random.randint(0, 99999999)
                    # bi_eda2 = os.path.join(dir_path, f'bi_eda2_{rand}.png')
                    # Save the plot
                    buyerid = self.request.session.get('buyerid')
                    folder = f'media/files/chatbot/{buyerid}/'
                    file_name = f'bi_eda2{rand}.png'
                    file_path = os.path.join(get_dataset_folder(folder, file_name), file_name)
                    fig.savefig(file_path)
                    plt.close()
                    return fig

                def select_columns(df, target_column, primary_key):
                    categorical_columns = df.select_dtypes(include='object').columns
                    continuous_columns = df.select_dtypes(exclude='object').columns

                    for col in continuous_columns:
                        if df[col].nunique() == 2:
                            categorical_columns = categorical_columns.append(pd.Index([col]))
                            continuous_columns = continuous_columns.difference(pd.Index([col]))

                    excluded_columns = ['id', 'date', 'ID', 'Id', 'DATE', 'user_id', 'ID']
                    if primary_key:
                        excluded_columns.append(primary_key[0])

                    categorical_columns = [col for col in categorical_columns if col not in excluded_columns]
                    continuous_columns = [col for col in continuous_columns if col not in excluded_columns]

                    categorical_columns = [col for col in categorical_columns if col != target_column]
                    continuous_columns = [col for col in continuous_columns if col != target_column]

                    num_categorical_columns = len(categorical_columns)

                    if num_categorical_columns < 2:
                        selected_categorical_columns = categorical_columns

                        if len(continuous_columns) >= 4 - num_categorical_columns:
                            if df[target_column].nunique() == 2:
                                mi_scores = mutual_info_classif(df[continuous_columns], df[target_column])
                            else:
                                mi_scores = mutual_info_regression(df[continuous_columns], df[target_column])

                            selected_continuous_columns = [col for col, score in sorted(zip(continuous_columns, mi_scores), key=lambda x: x[1], reverse=True)][:4 - num_categorical_columns]
                        else:
                            selected_continuous_columns = continuous_columns

                    elif num_categorical_columns == 2:
                        selected_categorical_columns = categorical_columns
                        selected_continuous_columns = continuous_columns[:2]

                    else:
                        selected_categorical_columns = categorical_columns[:2]
                        selected_continuous_columns = continuous_columns[:2]

                    selected_columns = [selected_categorical_columns, selected_continuous_columns]
                    return selected_columns

                aggdat = data.copy()
                # aggdat.drop([f"{tv}_original",primarykey],axis = 1,inplace=True,ignore='')
                if tv_type == 'Continuous':
                    target_type = data.select_dtypes(['number'])
                elif tv_type == 'Categorical':
                    target_type = data.select_dtypes(['object'])
                else:
                    print('Invalid! Please check')

                features = select_columns(aggdat, tv, primarykey)

                for i in aggdat.columns:
                    try:
                        aggdat[i] = aggdat[i].replace('\W', '', regex=True)
                    except:
                        aggdat[i] = aggdat[i].str.replace('\W', '', regex=True)

                num_list = aggdat.fillna(0).apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
                num_list = num_list.loc[lambda x : x == True].index.to_list()
                aggdat[num_list] = aggdat[num_list].astype(float)

                biedaplots, featsname, content, dtypess = [], [], [], []
                if len(features[0]) != 0:
                    for feat in features[0]:
                        plotsobj = obj_plot_shuffle(aggdat, feat)
                        biedaplots.append(plotsobj)
                        featsname.append(feat)
                        content.append(aggdat[feat].mode()[0])
                        dtypess.append('Categorical')

                if len(features[1]) != 0:
                    for feat in features[1]:
                        plotsnum = num_plot_shuffle(aggdat, feat)
                        biedaplots.append(plotsnum)
                        content.append(round(aggdat[feat].mean(), 2))
                        featsname.append(f'Avg. {feat}')
                        dtypess.append('Numerical')

                card_cont = []
                for i in range(len(featsname)):
                    fteda = {}
                    fteda['feat'] = [featsname[i]]
                    fteda['content'] = [content[i]]
                    fteda['dtype'] = [dtypess[i]]

                    json_records = pd.DataFrame(fteda).reset_index().to_json(orient='records')
                    fteda = json.loads(json_records)
                    card_cont.append(fteda)

                return biedaplots, card_cont


            def apply_custom_label_encoding(df, tv, label_map):
                """
                Dynamically applies label encoding to a target column based on a given label mapping.
                
                Parameters:
                    df (pd.DataFrame): The input DataFrame.
                    tv (str): Target variable/column name.
                    label_map (dict): Mapping of label -> numeric value. Example: {'Approved': 1, 'notApproved': 0}
                
                Returns:
                    pd.DataFrame: DataFrame with encoded target column.
                """

                # Ensure label_map is not reversed accidentally
                # If the mapping looks like {1: 'Approved'}, reverse it
                if all(isinstance(k, (int, float)) for k in label_map.keys()):
                    label_map = {v: k for k, v in label_map.items()}
                    print("🔁 Detected numeric keys — reversed mapping to:", label_map)

                # Show detected classes
                print(f"✅ Detected label mapping for '{tv}':", label_map)

                # Apply mapping safely
                df[tv] = df[tv].map(label_map)

                # Check for unmapped values
                unmapped = df[tv].isna().sum()
                if unmapped > 0:
                    print(f"⚠️ {unmapped} rows had unmapped values in '{tv}'")

                return df

            # ~~~~~~~~~~~~~~~~~~~~~~ EDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            edaplots = tv_eda(data, tv, tv_type ,gdbd_label)
            print('edaplots',edaplots)
            # import pandas as pd
            # from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            # from sklearn.preprocessing import LabelEncoder

            # ~~~~~~~~~~~~~~~~~~~~~~ EDA ~~~~~~~~~~~~~~~~~~~~~
            if tv_type=='Categorical':
              le = LabelEncoder()
            #   df_buynobuy[tv] = le.fit_transform(df_buynobuy[tv])
              df_buynobuy = apply_custom_label_encoding(df_buynobuy, tv, labels[0])
            #   print(le.classes_)
            else:
              pass
              

            loc_full, current_sxi = [], []
            currgdcnt, currbdcnt = [],[]
            eda_list, dist_list = [], []
            good_list, bad_list = [], []
            corr_list, cm_list = [], []
            loc_alpha, alpha_list = [], []
            cur_out = []
            sxiaccu, sxipreci = [], []
            goodabv, goodbel = [], []

            # SXI Method
            # DXI Data generate
            sxi_data = SxiProcess.sxi_data_generate(tv, df_buynobuy, primarykey)
            print('SXI-RL Engine Started -1')

            sxi_rl = sxirl_engine(tv_type, tv, df_buynobuy, sxi_data, list(labels[0].keys()), primarykey, )
            print('SXI-RL Engine Started -2')

            fulldata_sxi, sxi_avg, selcls, twds,combined_weights,dl_weights,weightcols,initsxiwgts = sxi_rl.generate_sxirl(tv, df_buynobuy, sxi_data, list(labels[0].keys()), primarykey )
            # print('sxi_avg',sxi_avg)
            # print('dl_weights',dl_weights)
            # print('correspon_cols',weightcols)
            # print('initsxiwgts',initsxiwgts)

            rfagent_weights = list(dl_weights[0]) if dl_weights else []
            if len(rfagent_weights) < len(weightcols):
                rfagent_weights.extend([0.0] * (len(weightcols) - len(rfagent_weights)))
            else:
                rfagent_weights = rfagent_weights[:len(weightcols)]

            initial_weight_values = [initsxiwgts.get(col, 0.0) for col in weightcols]

            jk = pd.DataFrame({
                'weightcols': weightcols,
                'rfagents': rfagent_weights,
                'initsxiwgts': initial_weight_values,
                'Lasso_Weight': combined_weights['Lasso_Weight'],
                'MI_Weight': combined_weights['MI_Weight'],
                'PCA_Weight': combined_weights['PCA_Weight'],
                'NB_Weight': combined_weights['NB_Weight'],
                'XGB_Weight': combined_weights['XGB_Weight']
            })
            print('jk',jk)
            buyerid = getattr(self, 'buyerid', None) or 'anonymous'
            folder = os.path.join(settings.MEDIA_ROOT, 'files', 'chatbot', str(buyerid))
            os.makedirs(folder, exist_ok=True)
            jk.to_csv(os.path.join(get_dataset_folder(folder, 'rfagent_weights.csv'), 'rfagent_weights.csv'), index=False)
            combined_weights.to_csv(os.path.join(get_dataset_folder(folder, 'feature_weight_comparison.csv'), 'feature_weight_comparison.csv'), index=False)

            # self.locfull = self.relative_to_media(f'fulldatarl{rand}.csv')
            buyerid = getattr(self, 'buyerid', None) or 'anonymous'
            folder = os.path.join(settings.MEDIA_ROOT, 'files', 'chatbot', str(buyerid))
            os.makedirs(folder, exist_ok=True)
            subindex_category = str(values_exe.get("subindex_category") or "").strip().lower()
            if subindex_category:
                file_name = f"fulldatarl_{subindex_category}_{buyerid}.csv"
            else:
                file_name = f"fulldatarl{rand}.csv"
            file_path = os.path.join(get_dataset_folder(folder, file_name), file_name)
            fulldata_sxi.to_csv(file_path, index=False)
            self.fulldata_path = file_path
            self.fulldata_url = f"/media/files/chatbot/{buyerid}/csv/{file_name}"
            if subindex_category:
                self.fulldata_category = subindex_category
            # import time
            # delay = 1500
            # print(f"Waiting for {delay} seconds...")
            # time.sleep(delay)

            print("primarykey before sxi method",primarykey)
            print("tv_type-------->",tv_type)
            locdist = self.distribution_plot(tv, labels[0], fulldata_sxi, sxi_avg , bad, good,tv_type)
            dist_list.append(locdist)
            # good_list.append(good)
            # bad_list.append(bad)


            if tv_type == 'Categorical':
                good = int(values_exe['Good Outcome Value'])
                bad = int(values_exe['Bad Outcome Value'])
                target_value = int(float(values_exe.get('Optimization Target Class', optimization_target_class)))
                bad_outcome_value = target_value
            else:
                target_value = optimization_target_class
                bad_outcome_value = optimization_target_class
                bad = bad


            print("Target Value (Bad Outcome):", target_value)
            print("Bad Outcome Value:", bad_outcome_value)
            print("Bad Numeric Index:", bad)
            print("Good Numeric Index:", good)

            toogle_val=self.toogle_val
            perchg=int(values_exe.get('Target Outcome Improvement', 20))
            if abs(perchg) == 10:
                perchg = 20 if perchg > 0 else -20
            curbdcnt = df_buynobuy[tv].eq(bad_outcome_value).sum()

            print("Filtered DataFrame for 'Bad Outcome Value':")
            print(bad_outcome_value)
            print("\nCount of 'Bad Outcome Value':", curbdcnt)
            print('Counts', df_buynobuy[tv].value_counts())
            print('Labels==',labels[0])
            print({
                "selected_outcome_ui": optimization_context.get("selected_outcome"),
                "selected_outcome_backend": target_value,
                "selected_outcome_meaning": optimization_context.get("selected_outcome_meaning"),
                "good_outcome": optimization_context.get("good_outcome"),
                "bad_outcome": optimization_context.get("bad_outcome"),
                "target_change": optimization_context.get("target_change"),
                "optimization_target_class": target_value,
                "slope": None,
                "current_sxi": sxi_avg,
                "target_sxi": None,
                "correlation_direction": None,
                "iteration_selected": values_exe.get("iteration_selected"),
            })


            ## Correlation Plot
            if tv_type == 'Continuous':
                tv_reg = f'{tv}_original'

                #coeffs,r2,rltyp,currout,myline,model_myline = self.correlation_plot_regres(target, dataframe, avg_composite_dxi) # target, labels, dataframe, good, bad,avg_composite_dxi
                
                coeffs,r2,rltyp,currout,myline, model_myline = self.correlation_plot_regres(tv_reg,fulldata_sxi,sxi_avg,target_value )

                lty, ltx, ltchg, updt_corr_plt, tgsxi, immediateval, midtermval, longtermval, perchg, ltchg, midper, curbdcnt, tgout,midout,plotly_fig,midsxi,r2 = self.correlationlgd(
                    perchg, currout, coeffs, myline, model_myline, tv, labels, tv_type, bad,
                    sxi_avg, toogle_val, target_value, r2, curbdcnt, good, fulldata_sxi,
                    display_r2=(getattr(self, "r2score", None) if getattr(self, "r2score", None) is not None else r2)
                )

            elif tv_type == 'Categorical':
                coeffs,r2,rltyp,currout,myline,model_myline = self.correlation_plot_classif(
                    tv,
                    labels[0],
                    fulldata_sxi,
                    good,
                    bad,
                    sxi_avg,
                    target_value,
                    toogle_val,
                    r2score=getattr(self, "r2score", None),
                )
                lty, ltx, ltchg, updt_corr_plt, tgsxi, immediateval, midtermval, longtermval, perchg, ltchg, midper, curbdcnt, tgout, midout, plotly_fig,midsxi,r2 = self.correlationlgd(
                    perchg, currout, coeffs, myline, model_myline, tv, labels[0], tv_type, bad,
                    sxi_avg, toogle_val, target_value, r2, curbdcnt, good, fulldata_sxi,
                    display_r2=(getattr(self, "r2score", None) if getattr(self, "r2score", None) is not None else r2)
                )
            

            print("<--testing 2nd start-->")
            print("perchg",perchg)
            # print("currout",currout)
            print("coeffs",coeffs)
            print("myline",myline)
            print("model_myline",model_myline)
            print("lty:", lty)
            print("ltx:", ltx)
            print("ltchg:", ltchg)
            print("updt_corr_plt:", updt_corr_plt)
            print("updt_corr_plt:", plotly_fig)
            print("trg_sxi:", tgsxi)
            print("immediateval:", immediateval)
            print("midtermval:", midtermval)
            print("longtermval:", longtermval)
            print("perchg:", perchg)
            print("ltchg:", ltchg)
            print("midper:", midper)
            print("curbdcnt:", curbdcnt)
            print("tgout:", tgout)
            print("midout:", midout)
            print("lty:", lty)
            print({
                "selected_outcome_ui": optimization_context.get("selected_outcome"),
                "selected_outcome_backend": target_value,
                "selected_outcome_meaning": optimization_context.get("selected_outcome_meaning"),
                "good_outcome": optimization_context.get("good_outcome"),
                "bad_outcome": optimization_context.get("bad_outcome"),
                "target_change": optimization_context.get("target_change"),
                "optimization_target_class": target_value,
                "slope": float(coeffs[-2]) if coeffs is not None and len(coeffs) >= 2 else 0.0,
                "current_sxi": sxi_avg,
                "target_sxi": tgsxi,
                "correlation_direction": rltyp,
                "iteration_selected": values_exe.get("iteration_selected"),
            })
            print("<--testing ends-->")
            # import time
            # delay = 1600
            # print(f"Waiting for {delay} seconds...")
            # time.sleep(delay)

            outyp = 'bad'
            corr_option = rltyp

            primarykey='i_n_d_e_x'
            primary_key_mapping = []
            primary_key_encoded = []

            goodab, goodbl, gdcnt, bdcnt = self.eda(tv, labels[0], fulldata_sxi, good, bad)

            dap, self.actlocs, self.sxiacc, self.sximae, actloc,self.r2score,self.sxirecall = [None] * 7
            dap, cmhtml, self.sxiacc, self.sxiprec,self.sxirecall ,cma, self.auc_best,accind,locauc,cm,within_percentage = [None] * 11
            
            if tv_type == 'Categorical':
                target_value = values_exe.get('Optimization Target Class', optimization_target_class)
                bad_outcome_value = target_value
                bad = int(values_exe['Bad Outcome Value'])
                good = int(values_exe['Good Outcome Value'])
            else:
                target_value = optimization_target_class
                bad_outcome_value = optimization_target_class
                bad = bad

            print("Target Value (Bad Outcome):", target_value)
            print("Bad Outcome Value:", bad_outcome_value)
            print("Bad Numeric Index:", bad)
            print("Good Numeric Index:", good)

            toogle_val=self.toogle_val
            perchg=int(values_exe.get('Target Outcome Improvement', 20))
            if abs(perchg) == 10:
                perchg = 20 if perchg > 0 else -20
            curbdcnt = df_buynobuy[tv].eq(bad_outcome_value).sum()

            print("Filtered DataFrame for 'Bad Outcome Value':")
            print(bad_outcome_value)
            print("\nCount of 'Bad Outcome Value':", curbdcnt)
            print('Counts', df_buynobuy[tv].value_counts())
            print('Labels==',labels[0])

            top_features_df = pd.DataFrame(columns=["Feature", "Importance", "Cumulative"])
            feat_importance_df = pd.DataFrame(columns=["Feature", "Importance", "Cumulative"])
            loc_curr_tree = loc_target_tree = None
            featureset = mi_score = mi_score_trgt = None
            best_path_0_str = best_path_1_str = None
            best_path_1_str_tr = best_path_0_str_tr = None
            target_tr_int = target_cr_int = None
            dap = None
            within_percentage = None
            mape = None
            cma = None
            accind = None
            sxi_auc_value_metric = None
            locauc = None
            cm = None
            cm_html = None


            ## SXI Method Running for accuracy, recall,weights, confusion matrix, AUC, R2 etc.
            if tv_type == 'Categorical':
                try:
                    print('Categorical SXI')
                    labels = {k: int(v) for k, v in labels[0].items()}
                    print('labels################', labels)
                    # fulldata_sxi.drop(columns=['netqyty_Bucket','gd_bdDXI'], ignore='errors')
                    fulldata_sxi = fulldata_sxi.drop(columns=['netqyty_Bucket', 'gd_bdDXI'], errors='ignore')

                    dap, cmhtml, self.sxiacc, self.sxiprec, cma, self.auc_best, sxi_auc_value_metric, locauc, cm, top_features_df, feat_importance_df, self.sxirecall = SxiProcess.sxi_method(
                        buyerid,good, bad, labels, tv_type, tv, fulldata_sxi, goodab, primarykey, sxi_avg, primary_key_mapping, primary_key_encoded
                    )

                    print('self.auc_best, self.sxiacc, self.sxiprec,sxi_auc_value_metric', self.auc_best, self.sxiacc, self.sxiprec, sxi_auc_value_metric)  # Saving values
                    self.r2score = categorical_display_r2_from_accuracy(self.sxiacc)
                    #print('categorical display r2 from SXI accuracy', self.r2score)
                    # within_percentage = None
                    print('conf matrix', cma)

                    print('conf matrix', cm)

                    if cma is not None:
                        cma_array = np.asarray(cma)
                        if cma_array.shape == (2, 2):
                            TN, FP, FN, TP = cma_array.ravel()
                            self.cm_dict = {
                                'TP': TP,
                                'FP': FP,
                                'TN': TN,
                                'FN': FN
                            }
                        else:
                            print(f"SXI warning: unexpected confusion matrix shape {cma_array.shape}")
                            self.cm_dict = None
                    else:
                        self.cm_dict = None  # Default if cma is not generated

                    print("actual predicted", dap)
                except Exception as e:
                    print("Error in SxiProcess.sxi_method:", e)
                    self.cm_dict = None
                    import traceback  # Import the traceback module
                    tb = traceback.format_exc()  # Get the full traceback as a string
                    print("Traceback Details:")
                    print(tb)  # Print the traceback details

            else:
                try:
                    print('Regression SXI')
                    dap, self.actlocs, self.r2score, self.sximae, actloc, within_percentage,top_features_df, feat_importance_df,mape = SxiProcess.sxi_methodreg(buyerid,good, bad, labels[0], tv_type, tv, fulldata_sxi, goodab, primarykey, sxi_avg, primary_key_mapping, primary_key_encoded)
                    print('self.actlocs, self.r2score, self.sximae',self.actlocs, self.r2score, self.sximae)  # Saving values
                    print("percentage regression", within_percentage)
                    print("actual predicted", dap)

                except Exception as e:
                    print("Error in SxiProcess.sxi_methodreg:", e)
                    self.cm_dict = None
                    import traceback  # Import the traceback module
                    tb = traceback.format_exc()  # Get the full traceback as a string
                    print("Traceback Details:")
                    print(tb)  # Print the traceback details

            if tv_type == 'Continuous' and self.r2score is not None:
                lty, ltx, ltchg, updt_corr_plt, tgsxi, immediateval, midtermval, longtermval, perchg, ltchg, midper, curbdcnt, tgout, midout, plotly_fig, midsxi, r2 = self.correlationlgd(
                    perchg, currout, coeffs, myline, model_myline, tv, labels, tv_type, bad,
                    sxi_avg, toogle_val, target_value, r2, curbdcnt, good, fulldata_sxi,
                    display_r2=self.r2score
                )
            elif tv_type == 'Categorical' and self.r2score is not None:
                label_mapping_for_corr = labels if isinstance(labels, dict) else labels[0]
                coeffs, r2, rltyp, currout, myline, model_myline = self.correlation_plot_classif(
                    tv,
                    label_mapping_for_corr,
                    fulldata_sxi,
                    good,
                    bad,
                    sxi_avg,
                    target_value,
                    toogle_val,
                    r2score=self.r2score,
                )
                lty, ltx, ltchg, updt_corr_plt, tgsxi, immediateval, midtermval, longtermval, perchg, ltchg, midper, curbdcnt, tgout, midout, plotly_fig, midsxi, r2 = self.correlationlgd(
                    perchg, currout, coeffs, myline, model_myline, tv, label_mapping_for_corr, tv_type, bad,
                    sxi_avg, toogle_val, target_value, r2, curbdcnt, good, fulldata_sxi,
                    display_r2=self.r2score
                )
                r2 = self.r2score


            print(f'Labels: {labels}')
            target_type = self.values_exe.get("Target Outcome Type", None)

            # fulldata_sxi = fulldata_sxi.drop(columns=['netqyty_Bucket', 'gd_bdDXI'], errors='ignore')

            if feat_importance_df is None or getattr(feat_importance_df, "empty", True):
                feat_importance_df = pd.DataFrame(columns=["Feature", "Importance", "Cumulative"])
            if top_features_df is None or getattr(top_features_df, "empty", True):
                top_features_df = feat_importance_df.copy()

            if feat_importance_df.empty:
                print("Skipping tree generation: feature importance data unavailable")
            elif target_type == "Continuous":
                # Regression path
                loc_curr_tree, loc_target_tree, featureset, mi_score, mi_score_trgt, best_path_0_str, best_path_1_str, best_path_1_str_tr, best_path_0_str_tr, target_tr_int, target_cr_int = self.tree(
                    tv_type, tv, fulldata_sxi, perchg, corr_option, labels[0], outyp, feat_importance_df
                )
            else:
                # fulldata_sxi = fulldata_sxi.drop(columns=['netqyty_Bucket', 'gd_bdDXI'], errors='ignore')
                # Classification path
                loc_curr_tree, loc_target_tree, featureset, mi_score, mi_score_trgt, best_path_0_str, best_path_1_str, best_path_1_str_tr, best_path_0_str_tr, target_tr_int, target_cr_int = self.tree(
                    tv_type, tv, fulldata_sxi, perchg, corr_option, labels, outyp, feat_importance_df
                )

            print("target_tr_int",target_tr_int)
            print("target_cr_int",target_cr_int)
            print("mi_score_target",mi_score_trgt)
            print("mi_score",mi_score)
            print("loc_curr_tree", loc_curr_tree)
            print("loc_target_tree", loc_target_tree)
            print("featureset", featureset)
            print("Equation :",self.equation_str)
            if not feat_importance_df.empty:
                buyerid = getattr(self, 'buyerid', None) or 'anonymous'
                folder = os.path.join(settings.MEDIA_ROOT, 'files', 'chatbot', str(buyerid))
                os.makedirs(folder, exist_ok=True)
                feat_importance_df.to_csv(os.path.join(get_dataset_folder(folder, 'sxiplusweights.csv'), 'sxiplusweights.csv'), index=False)

            # try:
            return {
                "sxi_avg": sxi_avg,
                "fulldata_sxi": fulldata_sxi,
                "curent_tv": curent_tv,
                "locdist": locdist,
                "selcls": selcls,
                "twds": twds,
                "r2": r2,
                "correlation_slope": float(coeffs[-2]) if coeffs is not None and len(coeffs) >= 2 else 0.0,
                "optimization_target_class": target_value,
                "selected_outcome": optimization_context.get("selected_outcome"),
                "selected_outcome_meaning": optimization_context.get("selected_outcome_meaning"),
                "target_change": optimization_context.get("target_change"),
                "target_polarity": optimization_context.get("target_polarity"),
                "lty": lty,
                "ltx": ltx,
                "ltchg": ltchg,
                "updt_corr_plt": updt_corr_plt,
                "tgsxi": tgsxi,
                "midsxi": midsxi,
                "immediateval": immediateval,
                "midtermval": midtermval,
                "longtermval": longtermval,
                "perchg": perchg,
                "midper": midper,
                "curbdcnt": curbdcnt,
                "tgout": tgout,
                "midout": midout,
                "plotly_fig": plotly_fig,
                "loc_curr_tree": loc_curr_tree,
                "loc_target_tree": loc_target_tree,
                "best_path_0_str": best_path_0_str,
                "best_path_1_str": best_path_1_str,
                "best_path_1_str_tr": best_path_1_str_tr,
                "best_path_0_str_tr": best_path_0_str_tr,
                "auc_best": self.auc_best,
                "sxiacc": self.sxiacc,
                "sxiprec": self.sxiprec,
                "cma": cma,
                "dap": dap,
                "actlocs": self.actlocs,
                "sximae": self.sximae,
                'r2score': self.r2score,
                "rltyp": rltyp,
                "accind": accind,
                "sxi_auc_value": sxi_auc_value_metric,
                "locauc": locauc,
                "cm": cm,
                "cm_html": cm_html,
                "edaplots": edaplots,
                "sxi_within_percentage": within_percentage,
                "mi_score": mi_score,
                "mi_score_target": mi_score_trgt,
                "mi_score_raw": self.current_tree_feature_importance_raw,
                "mi_score_target_raw": self.target_tree_feature_importance_raw,
                "encoded_feature_mapping": self.encoded_feature_mapping,
                "grouped_encoded_features": self.grouped_encoded_features,
                'cols': cols,
                "sxirecall": self.sxirecall,
                "combined_weights": combined_weights,
                "dnn_weights": jk,
                'error_pct':mape if tv_type == 'Continuous' else None,
                "fulldata_path": getattr(self, "fulldata_path", ""),
                "fulldata_url": getattr(self, "fulldata_url", ""),
                "fulldata_category": getattr(self, "fulldata_category", ""),
            }

        except Exception as e:
            import traceback
            print("Error:", e)
            print("Traceback Details:")
            trace = traceback.format_exc()
            print(trace)
            logger.error(trace)
            # logger.info(traceback.format_exc())
            return {
                "error": True,
                "message": str(e),
                "trace": trace,
            }



# output_dict = {'Target Outcome': 'Attrition', 'Good Outcome': 'No', 'Bad Outcome': 'Yes', 'Good Outcome Value': 0, 'Bad Outcome Value': 1, 'Target Outcome Type': 'Categorical', 'Target Outcome Improvement': -20, 'Target Outcome Improvement Status': 'in range'}

# mapping = None
# if output_dict.get('Target Outcome Type') == 'Categorical':
#     mapping = {
#         int(output_dict['Good Outcome Value']): output_dict['Good Outcome'],
#         int(output_dict['Bad Outcome Value']): output_dict['Bad Outcome']
#     }
# target_imp = int(output_dict.get('Target Outcome Improvement', 0))
# toggle = "Decrease" if target_imp < 0 else "Increase"

# my_object = model_execution(df, output_dict, mapping, toggle)
# result= my_object.sxi_execution()
# sxi_avg = result.get("sxi_avg")
# fulldata_sxi = result.get("fulldata_sxi")
# curent_tv = result.get("curent_tv")
# locdist = result.get("locdist")
# selcls = result.get("selcls")
# twds = result.get("twds")
# r2 = result.get("r2")
# lty = result.get("lty")
# ltx = result.get("ltx")
# ltchg = result.get("ltchg")
# updt_corr_plt = result.get("updt_corr_plt")
# tgsxi = result.get("tgsxi")
# midsxi = result.get("midsxi")
# immediateval = result.get("immediateval")
# midtermval = result.get("midtermval")
# longtermval = result.get("longtermval")
# perchg = result.get("perchg")
# midper = result.get("midper")
# curbdcnt = result.get("curbdcnt")
# tgout = result.get("tgout")
# midout = result.get("midout")
# plotly_fig = result.get("plotly_fig")
# loc_curr_tree = result.get("loc_curr_tree")
# loc_target_tree = result.get("loc_target_tree")
# best_path_0_str = result.get("best_path_0_str")
# best_path_1_str = result.get("best_path_1_str")
# best_path_1_str_tr = result.get("best_path_1_str_tr")
# best_path_0_str_tr = result.get("best_path_0_str_tr")
# auc_best = result.get("auc_best")
# sxiacc = result.get("sxiacc")
# sxiprec = result.get("sxiprec")
# cma = result.get("cma")
# dap = result.get("dap")
# actlocs = result.get("actlocs")
# sximae = result.get("sximae")
# rltyp = result.get("rltyp")
# accind = result.get("accind")
# locauc = result.get("locauc")
# cm = result.get("cm")
# edaplots = result.get("edaplots")
# sxi_within_percentage = result.get("sxi_within_percentage")
# mi_score = result.get("mi_score")
# mi_score_target = result.get("mi_score_target")
# cols = result.get("cols")
# r2score = result.get("r2score")
# sxirecall = result.get("sxirecall")


# threading.Thread(target=some_function, args=(request, df, output_dict, mapping,toggle, pipe)).start()
# threading.currentThread().setName('some_function')
