"""
Standalone Demo Script for SXI Engine.
Shows how to run the preprocessing, feature engineering, and model training
without requiring Django or a database.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure sxi_engine is on Python path
CURRENT_DIR = Path(__file__).resolve().parent
PKG_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = PKG_ROOT.parent
for p in [str(PROJECT_ROOT), str(PKG_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sxi_engine.preprocessing.eda_utils import generate_eda_dashboard
from sxi_engine.preprocessing.problem_contracts import get_problem, list_problems


def create_sample_ecommerce_data(num_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic customer dataset for demonstration."""
    np.random.seed(42)
    sessions = np.random.randint(1, 20, size=num_rows)
    engagement_time = np.random.uniform(10, 600, size=num_rows)
    add_to_carts = np.random.poisson(lam=1.5, size=num_rows)
    scroll_depth = np.random.uniform(0.1, 1.0, size=num_rows)
    cart_abandonment = (add_to_carts > 0) & (np.random.rand(num_rows) > 0.4)
    
    # Probability of conversion increases with engagement
    score = (
        0.3 * (sessions / 20) +
        0.4 * (engagement_time / 600) +
        0.5 * (add_to_carts / 5) -
        0.2 * cart_abandonment.astype(float)
    )
    is_converted = (score + np.random.normal(0, 0.1, size=num_rows) > 0.45).astype(int)
    revenue = np.where(is_converted == 1, np.random.exponential(scale=75, size=num_rows) + 20, 0.0)

    df = pd.DataFrame({
        "user_id": [f"USR_{i:04d}" for i in range(num_rows)],
        "total_sessions": sessions,
        "engagement_time": engagement_time,
        "add_to_carts": add_to_carts,
        "scroll_depth": scroll_depth,
        "cart_abandonment": cart_abandonment.astype(int),
        "device": np.random.choice(["mobile", "desktop", "tablet"], size=num_rows),
        "is_converted": is_converted,
        "revenue": np.round(revenue, 2),
    })
    return df


def main():
    print("==================================================")
    print("        SXI Engine Standalone Demonstration       ")
    print("==================================================")

    # 1. Show available business solutions
    problems = list_problems()
    print(f"\n[1] Registered Solutions in Catalog: {len(problems)} problem types")
    for p in problems[:4]:
        print(f"    • {p.get('badge')}: {p.get('display_name')} (Task: {p.get('mode')})")

    # 2. Create sample dataset
    output_dir = PKG_ROOT / "outputs" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv_path = output_dir / "sample_ecommerce.csv"

    df = create_sample_ecommerce_data(num_rows=300)
    df.to_csv(sample_csv_path, index=False)
    print(f"\n[2] Created synthetic sample dataset: {sample_csv_path}")
    print(f"    Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # 3. Automated EDA
    print("\n[3] Running Automated Exploratory Data Analysis (EDA)...")
    eda_result = generate_eda_dashboard(str(sample_csv_path), str(output_dir / "eda_charts"))
    if eda_result:
        print(f"    Missing cells: {eda_result['missing_cells']}")
        print(f"    Findings:")
        for f in eda_result["findings"]:
            print(f"      - {f}")

    print("\n==================================================")
    print("Demo completed successfully! Engine is ready for use.")
    print("==================================================")


if __name__ == "__main__":
    main()
