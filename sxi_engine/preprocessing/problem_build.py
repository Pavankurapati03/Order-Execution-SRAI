"""
Orchestrate problem lock → FE → master rebuild → Agent-2 target_info.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from django.conf import settings

from agent2.utils.target_info_store import persist_final_target_info
from pipeline.metadata_extractors.csv_extractor import (
    build_master_dataset,
    save_master_from_dataframe,
)
from pipeline.metadata_extractors.subindex_masters import build_subindex_masters_from_total
from pipeline.problem_contracts import get_group, get_problem, target_info_from_problem
from pipeline.problem_fe import (
    FeatureGapError,
    ProblemFEResult,
    apply_problem_fe,
    is_non_feature_column,
)
from pipeline.purchase_history import extract_commerce_bundle, persist_commerce_tables


def _features_from_fe_frame(df: pd.DataFrame, target: str) -> list[str]:
    """Locked model features only — exclude target + identity pass-through columns."""
    skip_extra: set[str] = set()
    if str(target or "") == "is_full_price_buyer":
        # Conversion label + discount flag used to build P1 — never train on them.
        skip_extra = {
            "is_buyer",
            "bought_with_discount",
            "p1_discount_source",
        }
        from pipeline.problem_contracts import get_problem

        allowed = set((get_problem(1) or {}).get("features") or [])
        return [
            c
            for c in df.columns
            if c in allowed and c not in skip_extra
        ]
    return [
        c
        for c in df.columns
        if c != target
        and c not in skip_extra
        and not is_non_feature_column(str(c))
    ]


def _master_frame_without_identity(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Master used by SXI must be numeric features + target only.

    user_pseudo_id stays on the FE CSV for CRM retargeting exports; including it
    in master can break SXI (non-numeric) because SXI hardcodes primary key to i_n_d_e_x.
    P4 keeps the original session grain (do not drop censored rows) so regression
    SXI receives a dense numeric frame like the original proxy path.
    """
    work = df

    cols = _features_from_fe_frame(work, target)
    # purchase_frequency is a P5 dashboard pass-through name but a P4 Stage-2 training feature
    if target == "days_until_next_purchase" and "purchase_frequency" in work.columns:
        if "purchase_frequency" not in cols:
            cols.append("purchase_frequency")
    if target in work.columns:
        cols = cols + [target]
    # Preserve column order from FE when possible
    ordered = [c for c in work.columns if c in cols]
    master = work[ordered].copy()
    if target == "days_until_next_purchase":
        for c in ordered:
            if c == target:
                continue
            master[c] = pd.to_numeric(master[c], errors="coerce").fillna(0)
        master[target] = pd.to_numeric(master[target], errors="coerce")
        master[target] = master[target].fillna(master[target].median())
    return master


def activate_problem(request, problem_id: int, *, fe_path: str | None = None) -> dict[str, Any]:
    """Switch session master/target to an already-FE'd problem without re-reading raw."""
    problem = get_problem(problem_id)
    if not problem:
        return {"ok": False, "error": f"Unknown problem id: {problem_id}"}

    run_id = str(request.session.get("run_id") or request.session.session_key or "session")
    if not fe_path:
        fe_path = os.path.join(
            settings.MEDIA_ROOT, "problem_builds", f"session_{run_id}", f"problem_{problem_id}_fe.csv"
        )
    if not os.path.exists(fe_path):
        return {"ok": False, "error": f"FE file missing for problem {problem_id}: {fe_path}"}

    try:
        df = pd.read_csv(fe_path)
    except Exception as exc:
        return {"ok": False, "error": f"Could not read FE file: {exc}"}

    pk = str(request.session.get("primary_key") or "")
    task_type = problem["task_type"]
    master_df = _master_frame_without_identity(df, problem["target"])
    master_info = save_master_from_dataframe(master_df, run_id)
    if not master_info:
        # Fall back: write a temp master-safe CSV then build
        safe_path = fe_path.replace("_fe.csv", "_master_safe.csv")
        try:
            master_df.to_csv(safe_path, index=False)
        except Exception:
            safe_path = fe_path
        master_info = build_master_dataset(
            run_id=run_id,
            preprocessing_mode="auto",
            custom_config={
                "target_column": problem["target"],
                "primary_key": pk,
                "task_type": task_type,
            },
            input_csv_path=safe_path,
            output_name=f"problem_{problem['id']}_master.csv",
            request=request,
        )
    if not master_info:
        return {"ok": False, "error": "Failed to activate master for problem."}

    features_used = _features_from_fe_frame(df, problem["target"])
    pos = None
    if task_type == "classification" and problem["target"] in df.columns:
        try:
            s = pd.to_numeric(df[problem["target"]], errors="coerce").fillna(0)
            pos = float((s > 0).mean()) if len(s) else None
        except Exception:
            pos = None

    request.session["csv_master_info"] = master_info
    request.session["selected_problem_id"] = problem["id"]
    request.session["build_mode"] = "problem_contract"
    request.session["task_type"] = task_type
    request.session["target_column"] = problem["target"]
    request.session["use_case"] = f"Predict {problem['target']} — {problem['name']}"
    request.session["locked_features"] = features_used
    request.session["problem_positive_rate"] = pos
    request.session["problem_fe_path"] = fe_path
    request.session["actionable_feature_mode"] = "contract"
    request.session["manual_selected_features"] = features_used
    request.session["show_decision_tree"] = True
    request.session["show_top_features"] = True
    request.session["show_target_features"] = True
    request.session["show_correlation"] = True
    request.session["show_improvement_levels"] = True

    quality_map = dict(request.session.get("problem_data_quality") or {})
    quality = quality_map.get(str(problem["id"])) or request.session.get("data_quality") or {}
    source_map = dict(request.session.get("problem_target_source") or {})
    target_source = source_map.get(str(problem["id"])) or request.session.get("target_source") or "unknown"
    request.session["data_quality"] = quality
    request.session["target_source"] = target_source

    target_info = target_info_from_problem(problem)
    target_info["Locked Features"] = features_used
    target_info["Positive Rate"] = pos
    target_info["Row Count"] = int(len(df))
    target_info["Show Decision Tree"] = True
    target_info["Show Top Features"] = True
    if "user_pseudo_id" in df.columns:
        target_info["Retargeting ID"] = "user_pseudo_id"
    target_info["Target Source"] = target_source
    target_info["Data Quality"] = quality
    persist_final_target_info(request, target_info)
    request.session["subindex_masters"] = {}
    request.session.modified = True
    from pipeline.session_utils import safe_session_save

    safe_session_save(request.session, label="activate_problem")
    return {
        "ok": True,
        "problem": problem,
        "features_used": features_used,
        "positive_rate": pos,
        "row_count": int(len(df)),
        "fe_path": fe_path,
        "master_info": master_info,
    }


def lock_and_build_group(request, group_id: int) -> dict[str, Any]:
    """
    FE + lock both problems in a $1000 group ($500 × 2).
    Activates the first problem for the upcoming SXI run; queues the second.
    """
    group = get_group(group_id)
    if not group:
        return {"ok": False, "error": f"Unknown group id: {group_id}"}

    results = []
    for pid in group["problem_ids"]:
        result = lock_and_build_problem(request, int(pid))
        if not result.get("ok"):
            return {
                "ok": False,
                "error": result.get("error") or f"Failed on problem {pid}",
                "missing": result.get("missing") or [],
                "failed_problem_id": pid,
                "group": group,
                "partial_results": results,
            }
        results.append(result)

    first_id = int(group["problem_ids"][0])
    activated = activate_problem(request, first_id, fe_path=results[0].get("fe_path"))
    if not activated.get("ok"):
        return activated

    request.session["selected_group_id"] = group["id"]
    request.session["group_problem_ids"] = list(group["problem_ids"])
    request.session["group_queue"] = list(group["problem_ids"][1:])
    request.session["group_dashboard_urls"] = {}
    request.session["group_dashboard_paths"] = {}
    request.session["group_name"] = group["name"]
    request.session.modified = True
    from pipeline.session_utils import safe_session_save

    safe_session_save(request.session, label="lock_and_build_group")
    return {
        "ok": True,
        "group": group,
        "results": results,
        "active_problem": activated.get("problem"),
        "queued_problem_ids": list(group["problem_ids"][1:]),
    }


def _resolve_event_rows_csv(request, run_id: str) -> str | None:
    """Event-level (pre user-collapse) CSV for timestamp sequences."""
    from agent2.utils.path_utils import resolve_media_path

    raw_info = request.session.get("csv_raw_info") or {}
    event_csv = raw_info.get("event_rows_csv") if isinstance(raw_info, dict) else None
    if event_csv:
        path = resolve_media_path(event_csv)
        if path and os.path.exists(path):
            return path
    session_dir = os.path.join(settings.MEDIA_ROOT, "feature_divisions", f"session_{run_id}")
    if os.path.isdir(session_dir):
        for name in sorted(os.listdir(session_dir), reverse=True):
            if "_event_rows_" in name and name.endswith(".csv"):
                return os.path.join(session_dir, name)
    return None


def _resolve_source_csv(request, run_id: str) -> str | None:
    """Prefer feature-division total.csv, else merged raw."""
    from agent2.utils.path_utils import resolve_media_path

    division_total = os.path.join(
        settings.MEDIA_ROOT, "feature_divisions", f"session_{run_id}", "total.csv"
    )
    if os.path.exists(division_total):
        return division_total

    raw_info = request.session.get("csv_raw_info") or {}
    raw_csv = raw_info.get("csv") if isinstance(raw_info, dict) else None
    if raw_csv:
        path = resolve_media_path(raw_csv)
        if path and os.path.exists(path):
            return path

    master = request.session.get("csv_master_info") or {}
    if isinstance(master, dict) and master.get("csv"):
        path = resolve_media_path(master["csv"])
        if path and os.path.exists(path):
            return path
    return None


def lock_and_build_problem(request, problem_id: int) -> dict[str, Any]:
    """
    Apply locked contract FE, write problem CSV, rebuild master, persist target_info.

    Returns dict with ok/error, fe notes, positive_rate, problem meta.
    """
    problem = get_problem(problem_id)
    if not problem:
        return {"ok": False, "error": f"Unknown problem id: {problem_id}"}

    run_id = str(request.session.get("run_id") or request.session.session_key or "session")
    source = _resolve_source_csv(request, run_id)
    if not source:
        return {"ok": False, "error": "No uploaded/merged dataset found for Build."}

    try:
        df = pd.read_csv(source)
    except Exception as exc:
        return {"ok": False, "error": f"Could not read dataset: {exc}"}

    if df.empty:
        return {"ok": False, "error": "Dataset is empty."}

    # Upload merge collapses duplicate user_pseudo_id → one row/user. P6/P7/P8 (and P2/P4
    # commerce) need the pre-collapse event/session rows so SXI sees the full upload.
    if int(problem["id"]) in (6, 7, 8):
        event_source = _resolve_event_rows_csv(request, run_id)
        if event_source:
            try:
                event_df = pd.read_csv(event_source)
                if not event_df.empty and len(event_df) >= len(df):
                    print(
                        f"[problem-build] P{problem['id']} using event-level rows "
                        f"({len(event_df)} rows vs collapsed {len(df)}) {event_source}"
                    )
                    df = event_df
            except Exception as exc:
                print(
                    f"[problem-build] P{problem['id']} event-level read failed, "
                    f"using merged frame: {exc}"
                )

    commerce_bundle = None
    commerce_paths = {}
    if int(problem["id"]) in (2, 4):
        try:
            event_source = _resolve_event_rows_csv(request, run_id)
            commerce_df = df
            if event_source:
                try:
                    commerce_df = pd.read_csv(event_source)
                    print(
                        f"[problem-build] P{problem['id']} commerce extract from event-level "
                        f"rows ({len(commerce_df)} rows) {event_source}"
                    )
                except Exception as exc:
                    print(f"[problem-build] event-level read failed, using merged frame: {exc}")
                    commerce_df = df
            commerce_bundle = extract_commerce_bundle(commerce_df)
            commerce_paths = persist_commerce_tables(run_id, commerce_bundle, settings.MEDIA_ROOT)
        except Exception as exc:
            commerce_bundle = None
            print(f"[problem-build] commerce extract failed for problem {problem['id']}: {exc}")

    input_row_count = int(len(df))

    try:
        inventory_df = None
        if int(problem["id"]) == 6:
            inv_path = request.session.get("p6_inventory_path")
            if inv_path and os.path.exists(str(inv_path)):
                try:
                    from pipeline.p6_inventory import read_inventory_csv

                    inventory_df = read_inventory_csv(str(inv_path))
                    print(f"[problem-build] P6 inventory loaded from {inv_path} ({len(inventory_df)} SKUs)")
                except Exception as exc:
                    print(f"[problem-build] P6 inventory read failed: {exc}")
                    inventory_df = None

        fe: ProblemFEResult = apply_problem_fe(
            df,
            problem["id"],
            commerce_bundle=commerce_bundle,
            inventory_df=inventory_df,
        )
    except FeatureGapError as gap:
        return {
            "ok": False,
            "error": gap.message,
            "missing": list(gap.missing),
            "problem": problem,
        }

    # Persist FE output for audit / dashboard
    out_dir = os.path.join(settings.MEDIA_ROOT, "problem_builds", f"session_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    fe_path = os.path.join(out_dir, f"problem_{problem['id']}_fe.csv")
    fe.df.to_csv(fe_path, index=False)

    pk = str(request.session.get("primary_key") or "")
    task_type = problem["task_type"]

    # Master for SXI: features + target only (FE CSV keeps user_pseudo_id for exports).
    master_df = _master_frame_without_identity(fe.df, problem["target"])
    master_info = save_master_from_dataframe(master_df, run_id)
    if not master_info:
        safe_path = fe_path.replace("_fe.csv", "_master_safe.csv")
        try:
            master_df.to_csv(safe_path, index=False)
        except Exception:
            safe_path = fe_path
        master_info = build_master_dataset(
            run_id=run_id,
            preprocessing_mode="auto",
            custom_config={
                "target_column": problem["target"],
                "primary_key": pk,
                "task_type": task_type,
            },
            input_csv_path=safe_path,
            output_name=f"problem_{problem['id']}_master.csv",
            request=request,
        )

    if not master_info:
        return {"ok": False, "error": "Failed to write master dataset for this problem."}

    features_used = list(fe.features_used) or _features_from_fe_frame(fe.df, problem["target"])
    request.session["csv_master_info"] = master_info
    request.session["selected_problem_id"] = problem["id"]
    request.session["build_mode"] = "problem_contract"
    request.session["task_type"] = task_type
    request.session["target_column"] = problem["target"]
    request.session["use_case"] = f"Predict {problem['target']} — {problem['name']}"
    request.session["locked_features"] = features_used
    request.session["problem_positive_rate"] = fe.positive_rate
    request.session["problem_fe_path"] = fe_path
    request.session["actionable_feature_mode"] = "contract"
    request.session["manual_selected_features"] = features_used
    request.session["show_decision_tree"] = True
    request.session["show_top_features"] = True
    request.session["show_target_features"] = True
    request.session["show_correlation"] = True
    request.session["show_improvement_levels"] = True
    # Individual lock — clear any prior group orchestration so SXI runs once.
    request.session.pop("selected_group_id", None)
    request.session["group_problem_ids"] = [problem["id"]]
    request.session["group_queue"] = []
    request.session["group_dashboard_urls"] = {}
    request.session["group_dashboard_paths"] = {}
    request.session["group_name"] = problem["name"]
    quality = dict(getattr(fe, "data_quality", None) or {})
    target_source = str(getattr(fe, "target_source", "") or quality.get("target_source") or "unknown")
    request.session["target_source"] = target_source
    request.session["data_quality"] = quality
    request.session["commerce_paths"] = commerce_paths
    quality_map = dict(request.session.get("problem_data_quality") or {})
    quality_map[str(problem["id"])] = quality
    request.session["problem_data_quality"] = quality_map
    source_map = dict(request.session.get("problem_target_source") or {})
    source_map[str(problem["id"])] = target_source
    request.session["problem_target_source"] = source_map
    request.session.modified = True

    target_info = target_info_from_problem(problem)
    target_info["Locked Features"] = features_used
    target_info["Positive Rate"] = fe.positive_rate
    if int(problem["id"]) == 6:
        target_info["Row Count"] = input_row_count
        target_info["Panel Row Count"] = int(len(fe.df))
    else:
        target_info["Row Count"] = int(len(fe.df))
    target_info["Show Decision Tree"] = True
    target_info["Show Top Features"] = True
    target_info["Show Target Features"] = True
    target_info["Show Correlation"] = True
    target_info["Show Improvement Levels"] = True
    if "user_pseudo_id" in fe.df.columns:
        target_info["Retargeting ID"] = "user_pseudo_id"
    # Persist FE notes so dashboards can show raw vs derived for this run
    if getattr(fe, "notes", None):
        target_info["FE Notes"] = list(fe.notes)
        request.session["problem_fe_notes"] = list(fe.notes)
    target_info["Target Source"] = target_source
    target_info["Data Quality"] = quality
    if quality and int(problem["id"]) in (2, 4) and not quality.get("has_repeat_history"):
        target_info["Proxy Warning"] = (
            "Insufficient repeat purchase history for a true P"
            f"{problem['id']} label. SXI ran on a disclosed proxy target."
        )
    persist_final_target_info(request, target_info)

    # Best-effort subindex slice — skipped for problem-contract (A/B don't need it)
    if request.session.get("build_mode") != "problem_contract":
        try:
            subindex_result = build_subindex_masters_from_total(
                run_id=run_id,
                main_master_info=master_info,
                primary_key=pk or "",
                request=request,
                target_column=problem["target"],
            )
            request.session["subindex_masters"] = subindex_result.get("subindex_masters") or {}
        except Exception:
            request.session["subindex_masters"] = {}
    else:
        request.session["subindex_masters"] = {}
        request.session["dxi_run_plan"] = {
            "total": True,
            "subindexes": [],
            "skipped": True,
            "reason": "problem_contract_build",
        }
    from pipeline.session_utils import safe_session_save

    safe_session_save(request.session, label="lock_and_build_problem")

    return {
        "ok": True,
        "problem": problem,
        "features_used": fe.features_used,
        "target": fe.target,
        "positive_rate": fe.positive_rate,
        "row_count": input_row_count if int(problem["id"]) == 6 else int(len(fe.df)),
        "panel_row_count": int(len(fe.df)) if int(problem["id"]) == 6 else None,
        "notes": fe.notes,
        "fe_path": fe_path,
        "master_info": master_info,
        "target_info": target_info,
        "target_source": target_source,
        "data_quality": quality,
        "commerce_paths": commerce_paths,
    }
