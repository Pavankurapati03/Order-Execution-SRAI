import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

def safe_read_file(path: str) -> Optional[pd.DataFrame]:
    try:
        if str(path).endswith(".xlsx") or str(path).endswith(".xls"):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception:
        return None

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.titlesize": 18,
})


def generate_eda_dashboard(dataset_path: str, output_path: str, display_name: str = "") -> Optional[Dict[str, Any]]:
    df = safe_read_file(dataset_path)
    if df is None or df.empty:
        return None

    file_name = display_name if display_name else os.path.basename(dataset_path)
    if file_name == "eda_input.csv":
        file_name = "Uploaded Dataset"

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    findings = _build_findings(df, num_cols, cat_cols)

    # Downsample exclusively for visual charts to prevent OOM on cloud free tiers
    df_plot = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df

    _render_dashboard(
        df=df_plot,
        file_name=file_name,
        num_cols=num_cols,
        cat_cols=cat_cols,
        findings=findings,
        output_path=output_path,
    )

    chart_files = _render_individual_charts(
        df=df_plot,
        num_cols=num_cols,
        cat_cols=cat_cols,
        output_path=output_path,
    )

    plt.close("all")

    return {
        "dataset_name": file_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "findings": findings,
        "charts": chart_files,
    }


def _build_findings(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    findings: List[str] = []

    if num_cols:
        skew_direction = "right" if float(df[num_cols[0]].dropna().skew() or 0) > 0 else "left"
        findings.append(f"{num_cols[0]} shows a {skew_direction}-skewed distribution")

    if cat_cols:
        top_category = df[cat_cols[0]].dropna()
        if not top_category.empty:
            findings.append(f"'{top_category.value_counts().idxmax()}' is the most frequent category")

    if len(num_cols) >= 2:
        findings.append(f"Scatter plot compares {num_cols[0]} and {num_cols[1]}")

    findings.append(f"Rows x columns: {df.shape[0]} x {df.shape[1]}")
    findings.append(f"Missing cells: {int(df.isna().sum().sum())}")

    return findings[:6]


def _top_correlation_pair(corr: pd.DataFrame) -> Optional[tuple[str, str, float]]:
    if corr.empty or corr.shape[1] < 2:
        return None

    mat = corr.abs().to_numpy(copy=True)
    np.fill_diagonal(mat, np.nan)
    corr_abs = pd.DataFrame(mat, index=corr.index, columns=corr.columns)
    top_corr = corr_abs.stack().sort_values(ascending=False)
    if top_corr.empty:
        return None

    left, right = top_corr.index[0]
    return str(left), str(right), float(top_corr.iloc[0])


def _render_dashboard(
    df: pd.DataFrame,
    file_name: str,
    num_cols: List[str],
    cat_cols: List[str],
    findings: List[str],
    output_path: str,
) -> None:
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.4, 2.6, 2.6])

    ax_header = fig.add_subplot(grid[0, :])
    ax_header.axis("off")
    ax_header.text(0.01, 0.5, "EDA Dashboard:", fontsize=22, fontweight="bold", va="center")
    ax_header.text(0.24, 0.5, file_name, fontsize=16, va="center")

    ax1 = fig.add_subplot(grid[1, 0])
    _plot_bar(df, cat_cols, ax1)

    ax2 = fig.add_subplot(grid[1, 1])
    _plot_pie(df, cat_cols, ax2)

    ax3 = fig.add_subplot(grid[2, 0])
    _plot_histogram(df, num_cols, ax3)

    ax4 = fig.add_subplot(grid[2, 1])
    _plot_scatter(df, num_cols, ax4)

    summary_text = "Summary of Key Findings\n" + "\n".join(f"- {finding}" for finding in findings)
    fig.text(
        0.01,
        0.01,
        summary_text,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="#F8FAFC", ec="#CBD5E1"),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_individual_charts(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    output_path: str,
) -> List[Dict[str, str]]:
    output_dir = os.path.dirname(output_path)
    output_stem = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    chart_specs = [
        ("bar", "Key Topic Distribution", _plot_bar, (7, 5)),
        ("pie", "Data Source Types", _plot_pie, (7, 5)),
        ("histogram", f"Distribution of {num_cols[0]}" if num_cols else "Distribution", _plot_histogram, (7, 5)),
        (
            "scatter",
            f"{num_cols[0]} vs {num_cols[1]}" if len(num_cols) >= 2 else "Scatter Plot",
            _plot_scatter,
            (7, 5),
        ),
    ]

    chart_files: List[Dict[str, str]] = []

    for slug, title, plotter, figsize in chart_specs:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        plotter(df, num_cols if plotter in {_plot_histogram, _plot_scatter, _plot_heatmap} else cat_cols, ax)
        plt.tight_layout()

        file_name = f"{output_stem}_{slug}.png"
        file_path = os.path.join(output_dir, file_name)
        fig.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        chart_files.append({
            "key": slug,
            "title": title,
            "file_name": file_name,
        })

    return chart_files


def _plot_bar(df: pd.DataFrame, cat_cols: List[str], ax: Any) -> None:
    if cat_cols:
        df[cat_cols[0]].fillna("Missing").value_counts().head(6).plot(
            kind="bar",
            ax=ax,
            color="#4C72B0",
        )
        ax.set_title("Key Topic Distribution")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)
        return

    ax.axis("off")
    ax.text(0.5, 0.5, "No categorical column available", ha="center", va="center")


def _plot_pie(df: pd.DataFrame, cat_cols: List[str], ax: Any) -> None:
    if cat_cols:
        df[cat_cols[0]].fillna("Missing").value_counts().head(4).plot(
            kind="pie",
            autopct="%1.0f%%",
            ax=ax,
            startangle=90,
        )
        ax.set_title("Data Source Types")
        ax.set_ylabel("")
        return

    counts = pd.Series({
        "Numeric": len(df.select_dtypes(include=np.number).columns),
        "Other": max(len(df.columns) - len(df.select_dtypes(include=np.number).columns), 0),
    })
    counts = counts[counts > 0]
    counts.plot(kind="pie", autopct="%1.0f%%", ax=ax, startangle=90)
    ax.set_title("Data Source Types")
    ax.set_ylabel("")


def _plot_histogram(df: pd.DataFrame, num_cols: List[str], ax: Any) -> None:
    if num_cols:
        series = df[num_cols[0]].dropna()
        if sns is not None:
            sns.histplot(series, kde=True, ax=ax)
        else:
            ax.hist(series, bins=30, color="#2563eb", edgecolor="white", alpha=0.8)
        ax.set_title(f"Distribution of {num_cols[0]}")
        return

    ax.axis("off")
    ax.text(0.5, 0.5, "No numeric column available", ha="center", va="center")


def _plot_scatter(df: pd.DataFrame, num_cols: List[str], ax: Any) -> None:
    if len(num_cols) >= 2:
        if sns is not None:
            sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]], ax=ax)
        else:
            ax.scatter(df[num_cols[0]], df[num_cols[1]], color="#2563eb", alpha=0.6, edgecolors="none")
        ax.set_title(f"{num_cols[0]} vs {num_cols[1]}")
        return

    ax.axis("off")
    ax.text(0.5, 0.5, "Need at least two numeric columns", ha="center", va="center")


def _plot_heatmap(df: pd.DataFrame, num_cols: List[str], ax: Any) -> None:
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        if sns is not None:
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        else:
            im = ax.imshow(corr, cmap="coolwarm", aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(corr.columns, fontsize=8)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
        ax.set_title("Correlation Heatmap")
        return

    ax.axis("off")
    ax.text(0.5, 0.5, "Correlation heatmap needs at least two numeric columns", ha="center", va="center")
