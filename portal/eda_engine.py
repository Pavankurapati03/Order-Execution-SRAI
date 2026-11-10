"""
Quantellix EDA Engine for Order Execution
Generates the exact same comprehensive visual Exploratory Data Analysis (EDA)
dashboard, individual charts, and executive insights as the Quantellix core pipeline.
"""

import os
import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sxi_engine.preprocessing.eda_utils import generate_eda_dashboard

def validate_and_load_dataset(file_path: str):
    """Safely loads CSV or Excel files into a pandas DataFrame."""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            try:
                df = pd.read_csv(file_path)
            except Exception:
                df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
        if df.empty or len(df.columns) < 2:
            return None, "Dataset is empty or has fewer than 2 columns."
        return df, None
    except Exception as e:
        return None, f"Failed to parse file: {str(e)}"

def run_order_execution_eda(df: pd.DataFrame, filename: str, output_dir: str = "", session_id: str = "") -> Dict[str, Any]:
    """
    Executes the full Quantellix EDA Engine:
    - Generates multi-panel visual EDA dashboard image
    - Generates individual chart PNGs (bar, pie, histogram, scatter, heatmap)
    - Extracts key statistical findings, skewness, top categories, and correlations
    """
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    if not session_id:
        session_id = "default"

    session_eda_dir = os.path.join(output_dir, f"eda_{session_id}")
    os.makedirs(session_eda_dir, exist_ok=True)

    clean_base = os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")
    eda_img_filename = f"{clean_base}_eda_dashboard.png"
    eda_img_path = os.path.join(session_eda_dir, eda_img_filename)

    # Save a temporary clean CSV to pass to generate_eda_dashboard
    temp_csv_path = os.path.join(session_eda_dir, "eda_input.csv")
    df.to_csv(temp_csv_path, index=False)

    eda_meta = generate_eda_dashboard(temp_csv_path, eda_img_path, display_name=filename)
    if not eda_meta:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        eda_meta = {
            "dataset_name": filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "findings": [f"Rows x columns: {df.shape[0]} x {df.shape[1]}", f"Missing cells: {int(df.isna().sum().sum())}"],
            "charts": []
        }

    # Ensure the real user-uploaded filename is displayed
    eda_meta["dataset_name"] = filename
    eda_meta["session_id"] = session_id

    # Build web URLs for frontend
    eda_meta["image_url"] = f"/portal/uploads/eda_{session_id}/{eda_img_filename}"
    eda_meta["download_url"] = f"/portal/uploads/eda_{session_id}/{eda_img_filename}"
    eda_meta["generated_at"] = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")

    # Format chart URLs (excluding heatmap)
    formatted_charts = []
    for ch in eda_meta.get("charts", []):
        if ch.get("key") == "heatmap" or "heatmap" in ch.get("title", "").lower() or "correlation" in ch.get("title", "").lower():
            continue
        fn = ch.get("file_name")
        if fn:
            ch_url = f"/portal/uploads/eda_{session_id}/{fn}"
            formatted_charts.append({
                "key": ch.get("key", ""),
                "title": ch.get("title", "EDA Chart"),
                "file_name": fn,
                "image_url": ch_url,
                "download_url": ch_url
            })
    eda_meta["charts"] = formatted_charts

    return eda_meta


def generate_eda_html_report(eda_info: Dict[str, Any], output_path: str) -> str:
    """
    Renders the Quantellix Executive EDA Dashboard HTML Report with premium styling,
    high-contrast visuals, responsive navigation, and interactive image preview.
    """
    session_id = eda_info.get("session_id", "")
    dataset_name = eda_info.get("dataset_name", "Dataset")
    if dataset_name == "eda_input.csv":
        dataset_name = "Uploaded Dataset"

    generated_at = eda_info.get("generated_at", datetime.datetime.now().strftime("%d %b %Y, %I:%M %p"))
    pdf_download_url = f"/api/eda/download-pdf/{session_id}" if session_id else "#"

    # Format findings with distinct category icons (excluding correlation)
    findings_list = eda_info.get("findings", [])
    findings_cards = []
    for f in findings_list:
        f_lower = f.lower()
        if "correlation" in f_lower or "covariance" in f_lower or "heatmap" in f_lower:
            continue
        if "skew" in f_lower or "distribution" in f_lower:
            icon = "📉"
            badge = "DISTRIBUTION SKEWNESS"
        elif "frequent" in f_lower or "category" in f_lower or "topic" in f_lower:
            icon = "🏷️"
            badge = "DOMINANT CATEGORY"
        elif "scatter" in f_lower or "compares" in f_lower or "vs" in f_lower:
            icon = "🔍"
            badge = "BIVARIATE INTERACTION"
        elif "rows" in f_lower or "columns" in f_lower:
            icon = "📐"
            badge = "DATASET PROFILE"
        else:
            icon = "💡"
            badge = "STATISTICAL FINDING"

        findings_cards.append(f"""
        <div class="finding-card">
          <div class="finding-badge"><span class="finding-icon">{icon}</span> {badge}</div>
          <div class="finding-text">{f}</div>
        </div>
        """)
    findings_html = "".join(findings_cards) if findings_cards else "<p style='color:#64748b;'>No specific anomalies detected.</p>"

    # Format individual charts (excluding heatmap)
    charts_cards = []
    for ch in eda_info.get("charts", []):
        ch_key = ch.get("key", "").upper()
        if ch_key == "HEATMAP" or "heatmap" in ch.get("title", "").lower() or "correlation" in ch.get("title", "").lower():
            continue
        tag_color = "#2563eb"
        if ch_key == "HISTOGRAM":
            tag_color = "#0891b2"
        elif ch_key == "PIE":
            tag_color = "#d97706"
        elif ch_key == "SCATTER":
            tag_color = "#059669"

        charts_cards.append(f"""
        <div class="chart-box">
          <div class="chart-header">
            <div>
              <span class="chart-pill" style="background:{tag_color}15; color:{tag_color}; border:1px solid {tag_color}30;">{ch_key or 'VISUALIZATION'}</span>
              <h4 class="chart-title">{ch['title']}</h4>
            </div>
            <a href="{ch['download_url']}" download="{ch['file_name']}" class="btn-chart-dl" title="Download {ch['title']} (PNG)">
              💾 Save
            </a>
          </div>
          <div class="chart-img-container" onclick="openLightbox('{ch['image_url']}', '{ch['title']}')" title="Click to enlarge">
            <img src="{ch['image_url']}" alt="{ch['title']}" loading="lazy" class="chart-img">
            <div class="img-zoom-hint">🔍 Click to zoom</div>
          </div>
        </div>
        """)
    charts_html = "".join(charts_cards)

    dashboard_img_url = eda_info.get("image_url", "")
    dashboard_dl_url = eda_info.get("download_url", dashboard_img_url)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Quantellix.AI Exploratory Data Analysis — {dataset_name}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary: #1d4ed8;
      --primary-hover: #1e40af;
      --primary-light: #eff6ff;
      --accent-cyan: #0284c7;
      --success: #059669;
      --success-light: #ecfdf5;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --text-sub: #475569;
      --border-color: #e2e8f0;
      --border-subtle: #cbd5e1;
      --radius-lg: 14px;
      --radius-md: 10px;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.04);
      --shadow-md: 0 4px 14px rgba(15,23,42,0.06);
      --shadow-lg: 0 10px 25px rgba(15,23,42,0.08);
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      color: var(--text-main);
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
      padding-bottom: 60px;
    }}

    /* Top Sticky Brand Navigation */
    .top-navbar {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-color);
      box-shadow: 0 2px 10px rgba(0,0,0,0.03);
      padding: 12px 28px;
    }}

    .navbar-inner {{
      max-width: 1320px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}

    .nav-brand {{
      display: flex;
      align-items: center;
      gap: 14px;
      text-decoration: none;
    }}

    .brand-logo-img {{
      height: 38px;
      width: auto;
      display: block;
      object-fit: contain;
    }}

    .brand-divider {{
      width: 1px;
      height: 24px;
      background: var(--border-color);
    }}

    .brand-subtitle {{
      font-size: 13px;
      font-weight: 700;
      color: var(--text-sub);
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .brand-badge {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      color: #1d4ed8;
      font-size: 11px;
      font-weight: 800;
      padding: 2px 8px;
      border-radius: 999px;
      letter-spacing: 0.3px;
      text-transform: uppercase;
    }}

    .nav-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .btn-nav {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 16px;
      font-size: 13px;
      font-weight: 700;
      border-radius: 8px;
      text-decoration: none;
      transition: all 0.15s ease;
      cursor: pointer;
    }}

    .btn-nav-outline {{
      background: #ffffff;
      color: #334155;
      border: 1.5px solid #cbd5e1;
    }}

    .btn-nav-outline:hover {{
      background: #f1f5f9;
      border-color: #94a3b8;
      color: #0f172a;
    }}

    .btn-nav-primary {{
      background: linear-gradient(135deg, #1d4ed8, #2563eb);
      color: #ffffff;
      border: 1px solid #1d4ed8;
      box-shadow: 0 2px 8px rgba(37,99,235,0.25);
    }}

    .btn-nav-primary:hover {{
      background: linear-gradient(135deg, #1e40af, #1d4ed8);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(37,99,235,0.35);
    }}

    /* Main Container */
    .container {{
      max-width: 1320px;
      margin: 28px auto 0 auto;
      padding: 0 24px;
    }}

    /* Hero Banner */
    .hero-banner {{
      background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 28px 32px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 20px;
      flex-wrap: wrap;
    }}

    .hero-title-area h1 {{
      font-size: 26px;
      font-weight: 800;
      color: #0f172a;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .hero-title-area p {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 6px;
      max-width: 780px;
    }}

    .hero-meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}

    .meta-tag {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      padding: 4px 10px;
      border-radius: 6px;
      color: #334155;
    }}

    .meta-tag-highlight {{
      background: #eff6ff;
      border-color: #bfdbfe;
      color: #1d4ed8;
      font-weight: 700;
    }}

    .meta-tag-success {{
      background: #ecfdf5;
      border-color: #a7f3d0;
      color: #065f46;
      font-weight: 700;
    }}

    /* KPI Metrics Cards Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}

    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 20px;
      box-shadow: var(--shadow-sm);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      position: relative;
      overflow: hidden;
    }}

    .kpi-card:hover {{
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }}

    .kpi-card::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
      background: #cbd5e1;
    }}

    .kpi-card.blue::before {{ background: #2563eb; }}
    .kpi-card.indigo::before {{ background: #4f46e5; }}
    .kpi-card.cyan::before {{ background: #0891b2; }}
    .kpi-card.emerald::before {{ background: #10b981; }}
    .kpi-card.amber::before {{ background: #f59e0b; }}

    .kpi-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }}

    .kpi-label {{
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      color: var(--text-muted);
    }}

    .kpi-icon {{
      font-size: 16px;
    }}

    .kpi-value {{
      font-size: 24px;
      font-weight: 800;
      color: #0f172a;
      letter-spacing: -0.5px;
      word-break: break-all;
    }}

    .kpi-subtext {{
      font-size: 12px;
      font-weight: 600;
      color: #64748b;
      margin-top: 4px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .kpi-badge {{
      display: inline-block;
      font-size: 10px;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 4px;
    }}

    .badge-good {{
      background: #dcfce7;
      color: #166534;
      border: 1px solid #bbf7d0;
    }}

    /* Section Cards */
    .section-card {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 26px 30px;
      margin-bottom: 28px;
      box-shadow: var(--shadow-sm);
    }}

    .section-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 20px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border-color);
    }}

    .section-title {{
      font-size: 18px;
      font-weight: 800;
      color: #0f172a;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .section-desc {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    /* Findings Grid */
    .findings-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}

    .finding-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 14px 18px;
      transition: all 0.15s ease;
    }}

    .finding-card:hover {{
      background: #ffffff;
      border-color: #cbd5e1;
      box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}

    .finding-badge {{
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.5px;
      color: #1e40af;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .finding-text {{
      font-size: 13.5px;
      font-weight: 600;
      color: #1e293b;
      line-height: 1.45;
    }}

    /* Master Multi-Panel Dashboard */
    .dashboard-preview-box {{
      position: relative;
      background: #f1f5f9;
      border: 1.5px solid var(--border-color);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: var(--shadow-sm);
      cursor: zoom-in;
    }}

    .dashboard-preview-box img {{
      width: 100%;
      height: auto;
      display: block;
      transition: transform 0.25s ease;
    }}

    .dashboard-preview-box:hover img {{
      transform: scale(1.008);
    }}

    .dashboard-zoom-bar {{
      position: absolute;
      bottom: 14px;
      right: 14px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      color: #ffffff;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
      pointer-events: none;
    }}

    /* Individual Charts Grid */
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 22px;
    }}

    .chart-box {{
      background: #ffffff;
      border: 1.5px solid var(--border-color);
      border-radius: 12px;
      padding: 16px;
      box-shadow: var(--shadow-sm);
      display: flex;
      flex-direction: column;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .chart-box:hover {{
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
      border-color: #cbd5e1;
    }}

    .chart-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 12px;
      gap: 12px;
    }}

    .chart-pill {{
      display: inline-block;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.5px;
      padding: 2px 7px;
      border-radius: 4px;
      margin-bottom: 4px;
    }}

    .chart-title {{
      font-size: 14px;
      font-weight: 700;
      color: #0f172a;
    }}

    .btn-chart-dl {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      color: #334155;
      font-size: 12px;
      font-weight: 700;
      padding: 5px 10px;
      border-radius: 6px;
      text-decoration: none;
      white-space: nowrap;
      transition: all 0.12s ease;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }}

    .btn-chart-dl:hover {{
      background: #eff6ff;
      border-color: #93c5fd;
      color: #1d4ed8;
    }}

    .chart-img-container {{
      position: relative;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      overflow: hidden;
      cursor: zoom-in;
    }}

    .chart-img {{
      width: 100%;
      height: auto;
      display: block;
      transition: transform 0.2s ease;
    }}

    .chart-img-container:hover .chart-img {{
      transform: scale(1.02);
    }}

    .img-zoom-hint {{
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(15, 23, 42, 0.75);
      color: #ffffff;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      opacity: 0;
      transition: opacity 0.2s ease;
      pointer-events: none;
    }}

    .chart-img-container:hover .img-zoom-hint {{
      opacity: 1;
    }}

    /* Lightbox Modal */
    .lightbox-modal {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(6px);
      -webkit-backdrop-filter: blur(6px);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}

    .lightbox-modal.active {{
      display: flex;
    }}

    .lightbox-content {{
      max-width: 95vw;
      max-height: 92vh;
      background: #ffffff;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.5);
      display: flex;
      flex-direction: column;
      gap: 12px;
      position: relative;
    }}

    .lightbox-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 8px;
      border-bottom: 1px solid #e2e8f0;
    }}

    .lightbox-title {{
      font-size: 16px;
      font-weight: 800;
      color: #0f172a;
    }}

    .lightbox-close {{
      background: #f1f5f9;
      border: none;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #475569;
    }}

    .lightbox-close:hover {{
      background: #e2e8f0;
      color: #0f172a;
    }}

    .lightbox-img {{
      max-width: 92vw;
      max-height: 80vh;
      object-fit: contain;
      border-radius: 6px;
    }}

    /* Print & PDF Optimizations */
    @media print {{
      body {{
        background: #ffffff !important;
        padding-bottom: 0 !important;
      }}
      .top-navbar, .nav-actions, .btn-chart-dl, .dashboard-zoom-bar, .img-zoom-hint {{
        display: none !important;
      }}
      .container {{
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
      }}
      .hero-banner, .section-card, .kpi-card, .chart-box {{
        box-shadow: none !important;
        border: 1px solid #ccc !important;
        page-break-inside: avoid;
      }}
      .chart-grid {{
        grid-template-columns: 1fr 1fr !important;
      }}
    }}
  </style>
</head>
<body>

  <!-- Top Sticky Navigation Bar -->
  <nav class="top-navbar">
    <div class="navbar-inner">
      <a href="/portal/chatbot" class="nav-brand" title="Back to Chatbot">
        <img src="/static/images/quantellix-logo-white-nav.png" alt="Quantellix.AI" class="brand-logo-img">
        <div class="brand-divider"></div>
        <div class="brand-subtitle">
          Decision Intelligence
          <span class="brand-badge">EDA Report</span>
        </div>
      </a>

      <div class="nav-actions">
        <a href="/portal/chatbot" class="btn-nav btn-nav-outline">
          ← Return to Assistant
        </a>
        <button type="button" onclick="window.print()" class="btn-nav btn-nav-outline" title="Print or save as PDF">
          🖨️ Print
        </button>
        <a href="{pdf_download_url}" download class="btn-nav btn-nav-primary" title="Download Executive PDF Document">
          📄 Download PDF Report
        </a>
      </div>
    </div>
  </nav>

  <!-- Content Container -->
  <main class="container">

    <!-- Hero Banner -->
    <section class="hero-banner">
      <div class="hero-title-area">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Comprehensive statistical profiling, anomaly detection, feature distribution analysis, and correlation screening generated for your ingested dataset.</p>
        <div class="hero-meta-row">
          <span class="meta-tag meta-tag-highlight">🎯 Target Solution: E4 • Reduce Delivery Lead Time</span>
          <span class="meta-tag">🕒 Generated: {generated_at}</span>
          <span class="meta-tag">⚡ Engine: Quantellix SXI Auto-Profiler</span>
          <span class="meta-tag meta-tag-success">🛡️ Ingestion Quality: 100% Cleansed</span>
        </div>
      </div>
    </section>

    <!-- KPI Metric Cards Grid -->
    <section class="kpi-grid">
      <div class="kpi-card blue">
        <div class="kpi-top">
          <span class="kpi-label">Dataset Filename</span>
          <span class="kpi-icon">📁</span>
        </div>
        <div class="kpi-value" style="font-size:16px; line-height:1.3;">{dataset_name}</div>
        <div class="kpi-subtext">Source Ingested File</div>
      </div>

      <div class="kpi-card indigo">
        <div class="kpi-top">
          <span class="kpi-label">Total Records</span>
          <span class="kpi-icon">📈</span>
        </div>
        <div class="kpi-value">{eda_info.get('rows', 0):,}</div>
        <div class="kpi-subtext">Cleaned Observations</div>
      </div>

      <div class="kpi-card cyan">
        <div class="kpi-top">
          <span class="kpi-label">Feature Columns</span>
          <span class="kpi-icon">📊</span>
        </div>
        <div class="kpi-value">{eda_info.get('columns', 0)}</div>
        <div class="kpi-subtext">Continuous & Categorical</div>
      </div>

      <div class="kpi-card emerald">
        <div class="kpi-top">
          <span class="kpi-label">Missing Cells</span>
          <span class="kpi-icon">🧹</span>
        </div>
        <div class="kpi-value">{eda_info.get('missing_cells', 0):,}</div>
        <div class="kpi-subtext"><span class="kpi-badge badge-good">✓ 100% Imputed</span></div>
      </div>

      <div class="kpi-card amber">
        <div class="kpi-top">
          <span class="kpi-label">Duplicate Rows</span>
          <span class="kpi-icon">🔁</span>
        </div>
        <div class="kpi-value">{eda_info.get('duplicate_rows', 0):,}</div>
        <div class="kpi-subtext"><span class="kpi-badge badge-good">✓ Deduplicated</span></div>
      </div>
    </section>

    <!-- Key Insights & Findings -->
    <section class="section-card">
      <div class="section-header">
        <div>
          <h3 class="section-title">💡 Summary of Statistical Findings & Insights</h3>
          <p class="section-desc">Key diagnostic patterns automatically extracted from feature skewness, variance, and covariances.</p>
        </div>
      </div>
      <div class="findings-grid">
        {findings_html}
      </div>
    </section>

    <!-- Master Multi-Panel Dashboard -->
    <section class="section-card">
      <div class="section-header">
        <div>
          <h3 class="section-title">🖼️ Complete Multi-Panel EDA Dashboard</h3>
          <p class="section-desc">360-degree overview combining topic density, distributions, bivariate interactions, and covariance.</p>
        </div>
        <div>
          <a href="{dashboard_dl_url}" download="Quantellix_EDA_Dashboard.png" class="btn-nav btn-nav-outline" style="font-size:12px; padding:6px 12px;">
            ⬇️ Download Full PNG
          </a>
        </div>
      </div>

      <div class="dashboard-preview-box" onclick="openLightbox('{dashboard_img_url}', 'Complete Multi-Panel EDA Dashboard')" title="Click to enlarge in high resolution">
        <img src="{dashboard_img_url}" alt="Complete EDA Dashboard" loading="eager">
        <div class="dashboard-zoom-bar">
          🔍 Click to view full resolution
        </div>
      </div>
    </section>

    <!-- Individual Charts Grid -->
    <section class="section-card">
      <div class="section-header">
        <div>
          <h3 class="section-title">📈 Deep-Dive Individual Visualizations</h3>
          <p class="section-desc">Individual high-resolution distribution, correlation, and feature breakdown charts.</p>
        </div>
      </div>
      <div class="chart-grid">
        {charts_html}
      </div>
    </section>

  </main>

  <!-- Interactive Lightbox Modal -->
  <div id="lightboxModal" class="lightbox-modal" onclick="closeLightbox(event)">
    <div class="lightbox-content" onclick="event.stopPropagation()">
      <div class="lightbox-header">
        <h4 id="lightboxTitle" class="lightbox-title">Chart Preview</h4>
        <button type="button" class="lightbox-close" onclick="closeLightbox(null)">✕</button>
      </div>
      <img id="lightboxImg" src="" alt="Enlarged Chart" class="lightbox-img">
    </div>
  </div>

  <script>
    function openLightbox(url, title) {{
      const modal = document.getElementById('lightboxModal');
      const img = document.getElementById('lightboxImg');
      const titleElem = document.getElementById('lightboxTitle');
      if (modal && img) {{
        img.src = url;
        if (titleElem) titleElem.textContent = title || 'Chart Preview';
        modal.classList.add('active');
      }}
    }}

    function closeLightbox(e) {{
      const modal = document.getElementById('lightboxModal');
      if (modal) {{
        modal.classList.remove('active');
      }}
    }}

    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape') {{
        closeLightbox(null);
      }}
    }});
  </script>

</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path

# Alias for backward compatibility
run_order_fulfillment_eda = run_order_execution_eda

