# import shap
# import numpy as np
# import pandas as pd


# # ----------------------------------------------------------
# # Helper: normalize shap_values into consistent list
# # ----------------------------------------------------------
# def normalize_shap_output(shap_values):
#     """
#     Converts SHAP output into a flat Python list of floats.
#     Handles: scalar, 1D, 2D, list-of-arrays.
#     """

#     # Multi-class (TreeExplainer sometimes returns list)
#     if isinstance(shap_values, list):
#         shap_values = shap_values[0]  # take class 0 for now

#     arr = np.array(shap_values)

#     try:
#         return arr[0].tolist()   # case: shape (1, n_features)
#     except Exception:
#         return arr.flatten().tolist()


# # ----------------------------------------------------------
# # Helper: convert expected_value safely to float
# # ----------------------------------------------------------
# def normalize_base_value(value):
#     """
#     SHAP expected_value formats:
#     - float
#     - array([float])
#     - array([[float]])
#     - list of floats
#     """

#     try:
#         return float(np.array(value).flatten()[0])
#     except Exception:
#         return None


# # ----------------------------------------------------------
# # LOCAL SHAP
# # ----------------------------------------------------------
# def explain_local_shap(model, row_df, background_df):

#     # 1. Try fast TreeExplainer
#     try:
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(row_df)
#         return {
#             "engine": "tree",
#             "expected_value": normalize_base_value(explainer.expected_value),
#             "shap_values": normalize_shap_output(shap_values),
#             "features": row_df.iloc[0].to_dict()
#         }
#     except:
#         pass

#     # 2. Try LinearExplainer
#     try:
#         explainer = shap.LinearExplainer(model, background_df)
#         shap_values = explainer.shap_values(row_df)
#         return {
#             "engine": "linear",
#             "expected_value": normalize_base_value(explainer.expected_value),
#             "shap_values": normalize_shap_output(shap_values),
#             "features": row_df.iloc[0].to_dict()
#         }
#     except:
#         pass

#     # 3. Slow fallback: KernelExplainer
#     background_small = background_df.sample(min(25, len(background_df)))
#     explainer = shap.KernelExplainer(model.predict, background_small)
#     shap_values = explainer.shap_values(row_df, nsamples=50)

#     return {
#         "engine": "kernel",
#         "expected_value": normalize_base_value(explainer.expected_value),
#         "shap_values": normalize_shap_output(shap_values),
#         "features": row_df.iloc[0].to_dict()
#     }


# # ----------------------------------------------------------
# # GLOBAL SHAP
# # ----------------------------------------------------------
# def explain_global_shap(model, df):

#     # 1. TreeExplainer
#     try:
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(df)
#         return {
#             "engine": "tree",
#             "expected_value": normalize_base_value(explainer.expected_value),
#             "shap_values": normalize_shap_output(shap_values),
#             "feature_names": df.columns.tolist()
#         }
#     except:
#         pass

#     # 2. LinearExplainer
#     try:
#         explainer = shap.LinearExplainer(model, df)
#         shap_values = explainer.shap_values(df)
#         return {
#             "engine": "linear",
#             "expected_value": normalize_base_value(explainer.expected_value),
#             "shap_values": normalize_shap_output(shap_values),
#             "feature_names": df.columns.tolist()
#         }
#     except:
#         pass

#     # 3. Kernel fallback
#     df_small = df.sample(min(25, len(df)))
#     explainer = shap.KernelExplainer(model.predict, df_small)
#     shap_values = explainer.shap_values(df_small, nsamples=50)

#     return {
#         "engine": "kernel",
#         "expected_value": normalize_base_value(explainer.expected_value),
#         "shap_values": normalize_shap_output(shap_values),
#         "feature_names": df.columns.tolist()
#     }

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DEBUG_SHAP = True
SHAP_PLOT_DIR = "shap_plots"


def debug_print(*args):
    if DEBUG_SHAP:
        print("\n[SHAP DEBUG]", *args)


# ----------------------------------------------------------
# Convert SHAP from log-odds → probability space
# ----------------------------------------------------------
def convert_logodds_to_probability_shap(shap_vals, base_value, pred_proba):
    """
    LinearExplainer gives log-odds SHAP values.
    Convert them into probability-space SHAP using:
    
        prob_shap = logodds_shap * (p * (1 - p))

    This makes SHAP meaningful & non-zero for logistic regression.
    """
    scale = pred_proba * (1 - pred_proba)
    return [v * scale for v in shap_vals]


# ----------------------------------------------------------
# Utility: Normalize SHAP output
# ----------------------------------------------------------
def normalize_shap_output(shap_values):
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    arr = np.array(shap_values)
    try:
        return arr[0].tolist()
    except:
        return arr.flatten().tolist()


def normalize_base_value(value):
    try:
        return float(np.array(value).flatten()[0])
    except:
        return None


# ----------------------------------------------------------
# Combine SHAP values + ranking
# ----------------------------------------------------------
def package_shap_output(shap_values_list, row_df, expected_value):
    features = row_df.iloc[0].to_dict()
    names = list(features.keys())

    pairs = [
        {"feature": names[i], "value": features[names[i]], "shap": shap_values_list[i]}
        for i in range(len(names))
    ]

    sorted_abs = sorted(pairs, key=lambda x: abs(x["shap"]), reverse=True)
    top_pos = [p for p in sorted_abs if p["shap"] > 0][:10]
    top_neg = [p for p in sorted_abs if p["shap"] < 0][:10]

    # Print summary for backend console
    debug_print("=== SHAP SUMMARY (probability space) ===")
    debug_print("Expected value:", expected_value)

    debug_print("\nTop Positive Contributors (+):")
    for p in top_pos:
        print(f"  {p['feature']} → +{p['shap']} (value={p['value']})")

    debug_print("\nTop Negative Contributors (–):")
    for p in top_neg:
        print(f"  {p['feature']} → {p['shap']} (value={p['value']})")

    return {
        "expected_value": expected_value,
        "feature_shap_pairs": pairs,
        "top_positive": top_pos,
        "top_negative": top_neg,
        "sorted_by_magnitude": sorted_abs[:20]
    }


# ----------------------------------------------------------
# Plot saver
# ----------------------------------------------------------
def save_shap_plots(explainer, shap_values, row_df, background_df):
    os.makedirs(SHAP_PLOT_DIR, exist_ok=True)
    paths = []

    shap_np = np.array(shap_values)

    # Force plot
    try:
        plt.figure()
        shap.force_plot(
            explainer.expected_value,
            shap_np[0],
            row_df,
            matplotlib=True,
            show=False
        )
        force_path = os.path.join(SHAP_PLOT_DIR, "force_plot.png")
        plt.savefig(force_path, bbox_inches="tight")
        plt.close()
        debug_print("Saved force plot:", force_path)
        paths.append(force_path)
    except Exception as e:
        debug_print("Force plot error:", e)

    # Summary plot
    try:
        plt.figure()
        shap.summary_plot(shap_np, background_df, show=False)
        summary_path = os.path.join(SHAP_PLOT_DIR, "summary_plot.png")
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()
        debug_print("Saved summary plot:", summary_path)
        paths.append(summary_path)
    except Exception as e:
        debug_print("Summary plot error:", e)

    return paths


# ----------------------------------------------------------
# LOCAL SHAP (probability-space)
# ----------------------------------------------------------
def explain_local_shap(model, row_df, background_df):

    debug_print("=== LOCAL SHAP START ===")

    # 1. LinearExplainer for Logistic Regression
    explainer = shap.LinearExplainer(model, background_df)
    shap_values_logodds = explainer.shap_values(row_df)

    # Normalize
    shap_logodds = normalize_shap_output(shap_values_logodds)
    base_logodds = normalize_base_value(explainer.expected_value)

    # 2. Convert to probability SHAP
    pred_proba = float(model.predict_proba(row_df)[0][1])   # class 1 probability
    shap_prob = convert_logodds_to_probability_shap(shap_logodds, base_logodds, pred_proba)

    # Save plots (still based on log-odds internally)
    plot_paths = save_shap_plots(explainer, shap_values_logodds, row_df, background_df)

    # Package
    packaged = package_shap_output(shap_prob, row_df, pred_proba)
    packaged["engine"] = "linear-probability"
    packaged["plot_paths"] = plot_paths
    packaged["predicted_probability"] = pred_proba

    return packaged




# ----------------------------------------------------------
# GLOBAL SHAP
# ----------------------------------------------------------
# ----------------------------------------------------------
# GLOBAL SHAP (probability-space, LLM-ready)
# ----------------------------------------------------------
def explain_global_shap(model, df):

    debug_print("=== GLOBAL SHAP START ===")

    # Always use LinearExplainer for LogisticRegression
    explainer = shap.LinearExplainer(model, df)
    shap_values_logodds = explainer.shap_values(df)

    # Expected value
    base_logodds = normalize_base_value(explainer.expected_value)

    # Raw SHAP array
    shap_logodds_array = np.array(shap_values_logodds)

    # Convert to probability-space SHAP
    pred_proba = model.predict_proba(df)[:, 1]              # per-sample probs
    scale_factors = pred_proba * (1 - pred_proba)           # derivative scale
    shap_prob = (shap_logodds_array.T * scale_factors).T    # rowwise multiply

    shap_prob_list = shap_prob.tolist()
    feature_names = df.columns.tolist()

    # GLOBAL abs mean importance
    mean_signed = np.mean(shap_prob, axis=0)

    global_ranking = sorted(
        [{"feature": feature_names[i], "shap": float(mean_signed[i])}
        for i in range(len(feature_names))],
        key=lambda x: abs(x["shap"]),
        reverse=True
    )

    feature_shap_pairs = [
        {"feature": feature_names[i], "shap": float(mean_signed[i])}
        for i in range(len(feature_names))
    ]

    # DEBUG PRINT SIGNED GLOBAL SHAP
    debug_print("Top GLOBAL signed SHAP contributors:")
    for item in global_ranking[:10]:
        debug_print(f"{item['feature']} → {item['shap']}")

    # ------------------------------------------------------
    # Save global summary plot
    # ------------------------------------------------------
    plot_paths = []
    try:
        plt.figure()
        shap.summary_plot(shap_prob, df, show=False)
        summary_path = os.path.join(SHAP_PLOT_DIR, "global_summary_plot.png")
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()
        debug_print("Saved global summary plot:", summary_path)
        plot_paths.append(summary_path)
    except Exception as e:
        debug_print("Global summary plot error:", e)

    # ------------------------------------------------------
    # ✔ FINAL LLM-ready output
    # ------------------------------------------------------
    return {
        "engine": "linear-probability-global",
        "expected_value": base_logodds,
        "feature_shap_pairs": feature_shap_pairs,       # <-- REQUIRED for LLM
        "global_feature_importance": global_ranking[:20],
        "shap_values_per_sample": shap_prob_list,
        "feature_names": feature_names,
        "plot_paths": plot_paths,
    }

