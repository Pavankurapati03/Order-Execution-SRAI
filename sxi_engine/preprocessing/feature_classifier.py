"""
Feature Classifier Module for Order Execution SXI Engine
Categorizes dataset columns into 5 domain buckets:
1. Marketing (campaigns, promotions, channel)
2. Demographic (customer ID, location, profile)
3. Engagement (order history, days since last order, velocity)
4. KPI / Behaviour (target outcomes, risk scores, delivery milestones)
5. Transaction (quantities, values, warehouse stock, pricing, inventory)
"""

import os
import json
import re
import pandas as pd
from typing import Dict, List, Any, Tuple

class FeatureClassifier:
    CATEGORIES = {
        "marketing": {
            "name": "Marketing",
            "description": "Marketing attribution, promotional campaigns, channel sources, discounts",
            "keywords": ["promo", "campaign", "channel", "marketing", "source", "medium", "referral", "discount", "coupon", "ad_", "utm_"]
        },
        "demographic": {
            "name": "Demographic",
            "description": "Customer profile, location, regional cluster, account attributes",
            "keywords": ["customer_id", "customer_ref", "location", "home_location", "city", "region", "country", "cluster", "zip", "address", "state", "gender", "age", "demographic", "profile"]
        },
        "engagement": {
            "name": "Engagement",
            "description": "Order velocity, reorder frequency, interaction metrics, historical engagement",
            "keywords": ["order_history", "days_since", "retry_count", "interaction", "frequency", "velocity", "visits", "sessions", "activity", "engagement", "login", "clicks"]
        },
        "kpi_behaviour": {
            "name": "KPI/Behaviour",
            "description": "Target outcome, stage milestones, risk scores, performance flags",
            "keywords": ["outcome", "target", "s1_", "s2_", "s3_", "s4_", "s5_", "s6_", "s7_", "s8_", "s9_", "s10_", "risk_score", "flag", "anomaly", "quality_check", "compliance", "delta", "score", "lead_time", "accuracy", "mismatch", "observed", "is_buyer"]
        },
        "transaction": {
            "name": "Transaction",
            "description": "Purchase values, item quantities, inventory units, pricing, warehouse metrics",
            "keywords": ["quantity", "items_per_order", "average_order_value", "stock_level", "capacity", "price", "cost", "weight", "volume", "warehouse", "transit", "duration", "latency", "dock", "distance", "order_date", "product_id", "product_ref", "reservation", "demand", "route", "eta"]
        }
    }

    def __init__(self):
        pass

    def classify_column(self, col_name: str, sample_series: pd.Series = None) -> str:
        name_lower = col_name.lower().strip()
        
        # Priority rules
        if re.search(r'^(s\d+_|outcome|target|lead_time|kpi|_event)', name_lower):
            return 'kpi_behaviour'
        if re.search(r'(customer_id|customer_ref|location|address|demographic)', name_lower):
            return 'demographic'
        if re.search(r'(promo|campaign|channel|ad_|marketing|utm_)', name_lower):
            return 'marketing'
        if re.search(r'(order_history|days_since|retry|frequency|engagement)', name_lower):
            return 'engagement'
        if re.search(r'(quantity|value|price|cost|stock|warehouse|weight|volume|product_id|amount|item)', name_lower):
            return 'transaction'

        scores = {}
        for cat_key, cat_data in self.CATEGORIES.items():
            score = 0
            for kw in cat_data["keywords"]:
                if kw in name_lower:
                    score += 2
            scores[cat_key] = score

        best_cat = max(scores, key=scores.get)
        if scores[best_cat] > 0:
            return best_cat

        if sample_series is not None and pd.api.types.is_numeric_dtype(sample_series):
            return "transaction"
        return "kpi_behaviour"

    def classify_and_divide_dataset(self, df: pd.DataFrame, session_id: str, output_dir: str) -> Dict[str, Any]:
        columns = list(df.columns)
        print(f"[FeatureClassifier] Classifying {len(columns)} columns...")
        
        mapping = {}
        for col in columns:
            mapping[col] = self.classify_column(col, df[col] if col in df else None)

        print(f"[FeatureClassifier] Dividing dataset into 5 categories...")
        print(f"[FeatureClassifier] Generating mapping JSON...")

        division_folder = os.path.join(output_dir, "feature_divisions", f"session_{session_id}")
        os.makedirs(division_folder, exist_ok=True)

        total_path = os.path.join(division_folder, "total.csv")
        df.to_csv(total_path, index=False)
        print(f"[FeatureClassifier] Saved original user uploading data (total): {len(columns)} columns")

        divided_files = {"total": total_path}
        category_counts = {}

        for cat_key in self.CATEGORIES.keys():
            cat_cols = [col for col, cat in mapping.items() if cat == cat_key]
            sub_df = df[cat_cols] if cat_cols else pd.DataFrame()
            cat_file = os.path.join(division_folder, f"{cat_key}.csv")
            sub_df.to_csv(cat_file, index=False)
            divided_files[cat_key] = cat_file
            category_counts[cat_key] = len(cat_cols)
            print(f"[FeatureClassifier] Saved {cat_key}: {len(cat_cols)} columns")

        json_path = os.path.join(division_folder, "feature_mapping.json")
        mapping_data = {
            "total_columns": len(columns),
            "categories": category_counts,
            "column_mapping": mapping
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2)

        return {
            "division_folder": division_folder,
            "mapping": mapping_data,
            "files": divided_files
        }
