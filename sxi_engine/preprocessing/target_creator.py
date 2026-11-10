import os
import re
import pandas as pd
from typing import Tuple

def create_is_buyer_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Creates a binary target column 'is_buyer' in the DataFrame.
    Deletes the source column used to prevent data leakage.
    
    Creation Logic:
    1. Transaction IDs: transaction_id, order_id, purchase_id, invoice_id.
    2. Transaction columns: transaction, purchase, order, payment, sale, booking.
    3. Fallback indicators: revenue, purchase_amount, payment_status, order_status, etc.
    """
    df = df.copy()
    
    # --- Match 1: Transaction IDs ---
    id_kws = ["transaction_id", "order_id", "purchase_id", "invoice_id", "transactionid", "orderid", "purchaseid", "invoiceid"]
    id_cols = [c for c in df.columns if any(kw in c.lower() for kw in id_kws)]
    
    if id_cols:
        source_col = id_cols[0]
        series_str = df[source_col].astype(str).str.lower().str.strip()
        invalid_vals = ["none", "null", "nan", "0", "false", "", "undefined"]
        
        df["is_buyer"] = (df[source_col].notna() & (~series_str.isin(invalid_vals))).astype(int)
        df = df.drop(columns=[source_col])
        # explanation = f"Created target column <b>is_buyer</b> from transaction ID column <b>{source_col}</b> (deleted to prevent data leakage)."
        explanation = ""
        return df, explanation

    # --- Match 2: Transaction columns ---
    tx_kws = ["transaction", "purchase", "order", "payment", "sale", "booking"]
    tx_cols = [c for c in df.columns if any(kw in c.lower() for kw in tx_kws)]
    
    if tx_cols:
        source_col = tx_cols[0]
        series = df[source_col]
        
        if pd.api.types.is_numeric_dtype(series):
            df["is_buyer"] = (series.notna() & (series > 0)).astype(int)
        else:
            series_str = series.astype(str).str.lower().str.strip()
            invalid_vals = ["failed", "canceled", "cancelled", "0", "false", "none", "null", "nan", "", "pending"]
            df["is_buyer"] = (series.notna() & (~series_str.isin(invalid_vals))).astype(int)
            
        df = df.drop(columns=[source_col])
        # explanation = f"Created target column <b>is_buyer</b> from transaction-related column <b>{source_col}</b> (deleted to prevent data leakage)."
        explanation = ""
        return df, explanation

    # --- Match 3: Fallback indicators ---
    fallback_kws = ["revenue", "purchase_amount", "payment_status", "order_status", "checkout", "cart", "subscription", "price"]
    fb_cols = [c for c in df.columns if any(kw in c.lower() for kw in fallback_kws)]
    
    if fb_cols:
        source_col = fb_cols[0]
        series = df[source_col]
        
        if pd.api.types.is_numeric_dtype(series):
            df["is_buyer"] = (series.notna() & (series > 0)).astype(int)
        else:
            series_str = series.astype(str).str.lower().str.strip()
            if "status" in source_col.lower():
                valid_vals = ["completed", "success", "paid", "active", "completed_checkout", "checkout", "subscribed", "yes", "true", "1"]
                df["is_buyer"] = (series.notna() & series_str.isin(valid_vals)).astype(int)
            else:
                invalid_vals = ["false", "0", "no", "none", "nan", "", "null"]
                df["is_buyer"] = (series.notna() & (~series_str.isin(invalid_vals))).astype(int)
                
        df = df.drop(columns=[source_col])
        # explanation = f"Created target column <b>is_buyer</b> using purchase indicator column <b>{source_col}</b> (deleted to prevent data leakage)."
        explanation = ""
        return df, explanation

    # --- Match 4: LLM Analysis Fallback ---
    try:
        from agent2.utils.llm_provider import request_llm
        columns_str = ", ".join(df.columns)
        prompt = f"""
        You are an expert data scientist. 
        Analyze the following column names from a dataset:
        [{columns_str}]
        
        Determine the best way to create a binary 'is_buyer' (1 for buyer, 0 for non-buyer) column.
        If a column represents purchase, transaction, revenue, or successful status, use it.
        If multiple columns need to be combined, do so.
        Return ONLY valid pandas python code that modifies the dataframe `df`.
        The code MUST assign `df['is_buyer']`.
        Do not include markdown blocks, just the raw python code.
        Example: df['is_buyer'] = (df['Status'] == 'Paid').astype(int)
        """
        
        code_str = request_llm([{"role": "user", "content": prompt}], max_tokens=200).strip()
        code_str = code_str.replace("```python", "").replace("```", "").strip()
        
        local_env = {"df": df, "pd": pd}
        exec(code_str, {}, local_env)
        df = local_env["df"]
        explanation = ""
    except Exception as e:
        df["is_buyer"] = 0
        explanation = ""
        
    return df, explanation
