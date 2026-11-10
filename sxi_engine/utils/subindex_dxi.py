"""
Run real subindex DXI analyses (no PDF) and export a structured Excel/CSV summary.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from django.conf import settings
from django.utils import timezone

from agent2.models import DxiRunResult, ModelExecutionResult, UserUpload
from agent2.utils.path_utils import resolve_media_path
from agent2.utils.sxi_executor import SXIExecutor

SUBINDEX_CATEGORIES = (
    "marketing",
    "demographic",
    "engagement",
    "kpi_behaviour",
    "transaction",
)

DASHBOARD_KEY = {
    "marketing": "marketing",
    "demographic": "demographic",
    "engagement": "engagement",
    "kpi_behaviour": "kpi",
    "transaction": "transactions",
}


def _to_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_scores(result: dict) -> Dict[str, Optional[float]]:
    if not isinstance(result, dict):
        return {
            "current_dxi": None,
            "immediate_dxi": None,
            "mid_dxi": None,
            "long_dxi": None,
        }
    immediate = result.get("immediate") if isinstance(result.get("immediate"), dict) else {}
    midterm = result.get("midterm") if isinstance(result.get("midterm"), dict) else {}
    longterm = result.get("longterm") if isinstance(result.get("longterm"), dict) else {}
    return {
        "current_dxi": _to_float(
            result.get("sxi_score")
            or result.get("sxi_avg")
            or result.get("current_sxi")
            or result.get("current_dxi")
        ),
        "immediate_dxi": _to_float(immediate.get("sxi") or result.get("tgsxi")),
        "mid_dxi": _to_float(midterm.get("sxi") or result.get("midsxi")),
        "long_dxi": _to_float(longterm.get("sxi") or result.get("ltx")),
    }


def _deterministic_fulldata_path(run_id: str, category: str) -> str:
    if not run_id or not category:
        return ""
    path = (
        Path(settings.MEDIA_ROOT)
        / "files"
        / "chatbot"
        / str(run_id)
        / "csv"
        / f"fulldatarl_{category}_{run_id}.csv"
    )
    return str(path) if path.is_file() else ""


def _compute_composite_dxi_correlation(
    csv_path: str,
    target_col: str = "is_buyer",
) -> Optional[float]:
    if not csv_path or not os.path.isfile(str(csv_path)):
        return None
    try:
        df = pd.read_csv(csv_path)
        if "composite_dxi" not in df.columns or target_col not in df.columns:
            return None
        work = df[["composite_dxi", target_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(work) < 3:
            return None
        if work["composite_dxi"].std() == 0 or work[target_col].std() == 0:
            return None
        return round(float(work["composite_dxi"].corr(work[target_col])), 4)
    except Exception as exc:
        print(f"[SUBINDEX DXI] correlation failed for {csv_path}: {exc}")
        return None


def _extract_top_features(result: dict, limit: int = 5) -> List[dict]:
    if not isinstance(result, dict):
        return []
    fi = result.get("feature_importance") or {}
    current = fi.get("current") if isinstance(fi, dict) else None
    rows: List[dict] = []
    if isinstance(current, dict):
        items = sorted(
            ((str(k), _to_float(v) or 0.0) for k, v in current.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        for rank, (name, score) in enumerate(items[:limit], start=1):
            rows.append({"rank": rank, "feature": name, "importance": round(score, 6)})
        return rows
    if isinstance(current, list):
        parsed = []
        for item in current:
            if isinstance(item, dict):
                name = str(
                    item.get("feature")
                    or item.get("Feature")
                    or item.get("name")
                    or item.get("Base_Feature")
                    or ""
                ).strip()
                score = _to_float(
                    item.get("importance")
                    or item.get("Importance")
                    or item.get("score")
                    or item.get("SXI_Weights")
                ) or 0.0
                if name:
                    parsed.append((name, score))
            else:
                parsed.append((str(item), 0.0))
        parsed.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, score) in enumerate(parsed[:limit], start=1):
            rows.append({"rank": rank, "feature": name, "importance": round(score, 6)})
    return rows


def _upsert_dxi_row(*, run_id: str, session_key: str, category: str, defaults: dict) -> DxiRunResult:
    obj, _ = DxiRunResult.objects.update_or_create(
        run_id=str(run_id),
        category=category,
        defaults={
            "session_key": session_key or "",
            **defaults,
        },
    )
    return obj


def save_total_dxi_from_main_result(
    *,
    request,
    run_id: str,
    result: dict,
    execution_id=None,
    master_path: str = "",
) -> Optional[DxiRunResult]:
    if not run_id:
        return None
    scores = _extract_scores(result if isinstance(result, dict) else {})
    session_key = getattr(request.session, "session_key", "") or ""
    execution = None
    if execution_id:
        try:
            execution = ModelExecutionResult.objects.filter(id=execution_id).first()
        except Exception:
            execution = None

    return _upsert_dxi_row(
        run_id=run_id,
        session_key=session_key,
        category="total",
        defaults={
            "master_path": master_path or "",
            "execution": execution,
            "current_dxi": scores["current_dxi"],
            "immediate_dxi": scores["immediate_dxi"],
            "mid_dxi": scores["mid_dxi"],
            "long_dxi": scores["long_dxi"],
            "top_features": _extract_top_features(result if isinstance(result, dict) else {}),
            "status": "completed" if scores["current_dxi"] is not None else "failed",
            "error_message": "" if scores["current_dxi"] is not None else "Missing current DXI",
            "started_at": timezone.now(),
            "completed_at": timezone.now(),
            "extras": {"source": "main_sxi"},
        },
    )


def _append_chat(request, text: str) -> None:
    if not request or not hasattr(request, "session"):
        return
    history = request.session.get("chat_history", [])
    history.append({"sender": "bot", "text": text})
    request.session["chat_history"] = history
    request.session.modified = True


def _save_subindex_execution(
    *,
    session,
    upload: UserUpload,
    category: str,
    master_path: str,
    target_info: dict,
    result: dict,
    task_type: str = "classification",
) -> Optional[ModelExecutionResult]:
    try:
        from agent2.utils.json_utils import make_json_safe

        normalized_task = str(task_type or "classification").strip().lower()
        if normalized_task in {"numeric", "continuous"}:
            normalized_task = "regression"
        elif normalized_task in {"categorical"}:
            normalized_task = "classification"

        return ModelExecutionResult.objects.create(
            session=session,
            upload=upload,
            agent_name="agent2",
            model_name=f"SXI_subindex_{category}",
            modality="tabular",
            task_type=normalized_task,
            input_schema=make_json_safe(
                {
                    "target_info": target_info,
                    "category": category,
                    "master_path": master_path,
                    "subindex": True,
                    "generate_reports": False,
                }
            ),
            execution_metadata=make_json_safe(
                {
                    "category": category,
                    "pdf_skipped": True,
                    "encoding_mode": "inherited_5_plus_1_from_total",
                    "task_type": normalized_task,
                }
            ),
            metrics=make_json_safe(
                {
                    "sxi_score": result.get("sxi_score") or result.get("sxi_avg"),
                    "immediate": result.get("immediate"),
                    "midterm": result.get("midterm"),
                    "longterm": result.get("longterm"),
                }
            ),
            artifacts=make_json_safe({"plots": result.get("plots") or {}}),
            raw_output=make_json_safe(result),
            explanation_ready=False,
        )
    except Exception as exc:
        print(f"[SUBINDEX DXI] Failed to save ModelExecutionResult for {category}: {exc}")
        return None


def export_subindex_summary(
    *,
    run_id: str,
    rows: List[dict],
    feature_rows: List[dict],
    metadata: dict,
) -> Dict[str, str]:
    out_dir = Path(settings.MEDIA_ROOT) / "files" / "chatbot" / str(run_id) / "subindex_dxi"
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = out_dir / "subindex_dxi_summary.xlsx"
    csv_path = out_dir / "subindex_dxi_summary.csv"
    features_csv = out_dir / "subindex_feature_importance.csv"
    meta_csv = out_dir / "subindex_metadata.csv"

    summary_df = pd.DataFrame(rows)
    features_df = pd.DataFrame(feature_rows or [{"note": "no feature importance captured"}])
    meta_df = pd.DataFrame([metadata])

    summary_df.to_csv(csv_path, index=False)
    features_df.to_csv(features_csv, index=False)
    meta_df.to_csv(meta_csv, index=False)

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            features_df.to_excel(writer, sheet_name="feature_importance", index=False)
            meta_df.to_excel(writer, sheet_name="metadata", index=False)
    except Exception as exc:
        print(f"[SUBINDEX DXI] Excel export failed, CSV still saved: {exc}")
        xlsx_path = Path("")

    return {
        "xlsx": str(xlsx_path) if xlsx_path else "",
        "csv": str(csv_path),
        "features_csv": str(features_csv),
        "metadata_csv": str(meta_csv),
        "xlsx_url": f"/media/files/chatbot/{run_id}/subindex_dxi/subindex_dxi_summary.xlsx" if xlsx_path else "",
        "csv_url": f"/media/files/chatbot/{run_id}/subindex_dxi/subindex_dxi_summary.csv",
    }


def run_subindex_dxi_analyses(
    *,
    request,
    session,
    upload: UserUpload,
    target_info: dict,
    run_id: str,
) -> Dict[str, Any]:
    """
    Run SXI on each ready subindex master with generate_reports=False.
    Persist DxiRunResult rows and write one Excel/CSV summary.
    """
    subindex_masters = request.session.get("subindex_masters") or {}
    if not subindex_masters:
        # Fall back to filesystem discovery from manifest / master folders
        for category in SUBINDEX_CATEGORIES:
            path = Path(settings.MEDIA_ROOT) / "master" / category / f"master_{run_id}_{category}.csv"
            if path.exists():
                subindex_masters[category] = {
                    "category": category,
                    "path": str(path),
                    "csv": f"/media/master/{category}/{path.name}",
                    "status": "ready",
                    "feature_count": 0,
                    "rows": 0,
                }

    if not subindex_masters:
        return {
            "completed": 0,
            "total": 0,
            "export": {},
            "message": "No subindex masters found",
            "all_complete": False,
        }

    session_key = getattr(request.session, "session_key", "") or ""
    task_type = str(
        (target_info or {}).get("Task Type")
        or request.session.get("task_type")
        or "classification"
    ).strip().lower()
    if task_type in {"numeric", "continuous"}:
        task_type = "regression"
    elif task_type in {"categorical"}:
        task_type = "classification"

    status = {
        "status": "running",
        "completed": 0,
        "total": len(subindex_masters),
        "categories": {},
        "export_path": "",
        "task_type": task_type,
        "updated_at": timezone.now().isoformat(),
    }
    request.session["subindex_dxi_status"] = status
    request.session.modified = True
    try:
        request.session.save()
    except Exception:
        pass

    _append_chat(
        request,
        "📊 Computing <b>real subindex DXI</b> scores (4 categories, no PDF — Excel/CSV export).",
    )

    summary_rows: List[dict] = []
    feature_rows: List[dict] = []
    completed = 0

    previous_mode = request.session.get("sxi_execution_mode")
    # Avoid comparison_only early-return so we get target DXI fields without PDF.
    request.session.pop("sxi_execution_mode", None)
    request.session.modified = True

    try:
        for category in SUBINDEX_CATEGORIES:
            info = subindex_masters.get(category) or {}
            if not info:
                continue

            master_url = info.get("path") or info.get("master_df_encoded") or info.get("csv") or ""
            master_path = resolve_media_path(master_url) if master_url else ""
            if master_path and not os.path.isabs(str(master_path)):
                master_path = resolve_media_path(master_path)

            feature_count = int(info.get("feature_count") or 0)
            row_count = int(info.get("rows") or 0)
            skip = str(info.get("status") or "").startswith("skipped") or feature_count < 2

            _upsert_dxi_row(
                run_id=run_id,
                session_key=session_key,
                category=category,
                defaults={
                    "master_path": str(master_path or ""),
                    "feature_count": feature_count,
                    "row_count": row_count,
                    "status": "running" if not skip else "skipped",
                    "error_message": info.get("skip_reason") or ("Insufficient features" if skip else ""),
                    "started_at": timezone.now(),
                    "completed_at": timezone.now() if skip else None,
                },
            )

            if skip or not master_path or not os.path.isfile(str(master_path)):
                err = info.get("skip_reason") or f"Master missing for {category}"
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "category": category,
                        "dashboard_key": DASHBOARD_KEY.get(category, category),
                        "master_path": str(master_path or ""),
                        "row_count": row_count,
                        "feature_count": feature_count,
                        "current_dxi": None,
                        "immediate_target_dxi": None,
                        "mid_target_dxi": None,
                        "long_target_dxi": None,
                        "dxi_gap_immediate": None,
                        "dxi_gap_mid": None,
                        "dxi_gap_long": None,
                        "status": "skipped",
                        "error": err,
                        "execution_id": None,
                    }
                )
                status["categories"][category] = {"status": "skipped", "error": err}
                completed += 1
                status["completed"] = completed
                request.session["subindex_dxi_status"] = status
                request.session.modified = True
                _append_chat(request, f"⚠️ Subindex DXI skipped: <b>{category}</b> — {err}")
                continue

            _append_chat(
                request,
                f"⏳ Subindex DXI {completed + 1}/{len(subindex_masters)}: <b>{category}</b>…",
            )
            try:
                request.session.save()
            except Exception:
                pass

            try:
                # Refresh row/feature counts from file if missing
                if row_count <= 0 or feature_count <= 0:
                    try:
                        df_probe = pd.read_csv(master_path, nrows=5)
                        feature_count = max(0, len(df_probe.columns) - 3)
                        row_count = sum(1 for _ in open(master_path, "r", encoding="utf-8", errors="ignore")) - 1
                    except Exception:
                        pass

                iteration_target = copy.deepcopy(target_info or {})
                iteration_target["subindex_category"] = category
                iteration_target["iteration_selected"] = f"subindex:{category}"

                executor = SXIExecutor(
                    request=request,
                    dataframe_path=str(master_path),
                    target_info=iteration_target,
                )
                result = executor.execute(
                    generate_reports=False,
                    include_report_context=False,
                )

                if result.get("error"):
                    raise RuntimeError(result.get("message") or result.get("error") or "SXI failed")

                scores = _extract_scores(result)
                top_feats = _extract_top_features(result)
                execution = _save_subindex_execution(
                    session=session,
                    upload=upload,
                    category=category,
                    master_path=str(master_path),
                    target_info=iteration_target,
                    result=result,
                    task_type=task_type,
                )

                cur = scores["current_dxi"]
                imm = scores["immediate_dxi"]
                mid = scores["mid_dxi"]
                lng = scores["long_dxi"]

                fulldata_path = str(result.get("fulldata_path") or _deterministic_fulldata_path(run_id, category))
                fulldata_url = str(result.get("fulldata_url") or "")
                if fulldata_path and not fulldata_url:
                    rel = fulldata_path.replace(str(settings.MEDIA_ROOT), "").replace("\\", "/").lstrip("/")
                    fulldata_url = f"/media/{rel}" if rel else ""
                target_col = str((target_info or {}).get("Target Outcome") or "is_buyer")
                target_correlation = _compute_composite_dxi_correlation(fulldata_path, target_col)

                _upsert_dxi_row(
                    run_id=run_id,
                    session_key=session_key,
                    category=category,
                    defaults={
                        "master_path": str(master_path),
                        "execution": execution,
                        "current_dxi": cur,
                        "immediate_dxi": imm,
                        "mid_dxi": mid,
                        "long_dxi": lng,
                        "feature_count": feature_count,
                        "row_count": row_count,
                        "top_features": top_feats,
                        "status": "completed" if cur is not None else "failed",
                        "error_message": "" if cur is not None else "Missing current DXI in SXI result",
                        "completed_at": timezone.now(),
                        "extras": {
                            "dashboard_key": DASHBOARD_KEY.get(category),
                            "pdf_skipped": True,
                            "task_type": task_type,
                            "fulldata_path": fulldata_path,
                            "fulldata_url": fulldata_url,
                            "target_correlation": target_correlation,
                            "target_column": target_col,
                        },
                    },
                )

                gap_imm = (imm - cur) if cur is not None and imm is not None else None
                gap_mid = (mid - cur) if cur is not None and mid is not None else None
                gap_long = (lng - cur) if cur is not None and lng is not None else None

                summary_rows.append(
                    {
                        "run_id": run_id,
                        "category": category,
                        "dashboard_key": DASHBOARD_KEY.get(category, category),
                        "master_path": str(master_path),
                        "row_count": row_count,
                        "feature_count": feature_count,
                        "current_dxi": cur,
                        "immediate_target_dxi": imm,
                        "mid_target_dxi": mid,
                        "long_target_dxi": lng,
                        "dxi_gap_immediate": round(gap_imm, 4) if gap_imm is not None else None,
                        "dxi_gap_mid": round(gap_mid, 4) if gap_mid is not None else None,
                        "dxi_gap_long": round(gap_long, 4) if gap_long is not None else None,
                        "status": "completed" if cur is not None else "failed",
                        "error": "",
                        "execution_id": getattr(execution, "id", None),
                    }
                )
                for feat in top_feats:
                    feature_rows.append(
                        {
                            "category": category,
                            "feature": feat.get("feature"),
                            "importance": feat.get("importance"),
                            "rank": feat.get("rank"),
                        }
                    )

                status["categories"][category] = {
                    "status": "completed",
                    "current_dxi": cur,
                    "immediate_dxi": imm,
                    "execution_id": getattr(execution, "id", None),
                }
                _append_chat(
                    request,
                    f"✅ Subindex DXI complete: <b>{category}</b> — current "
                    f"<b>{cur if cur is not None else 'N/A'}</b>"
                    + (f", immediate target <b>{imm}</b>" if imm is not None else ""),
                )
            except Exception as exc:
                print(f"[SUBINDEX DXI] {category} failed: {exc}")
                _upsert_dxi_row(
                    run_id=run_id,
                    session_key=session_key,
                    category=category,
                    defaults={
                        "master_path": str(master_path or ""),
                        "status": "failed",
                        "error_message": str(exc),
                        "completed_at": timezone.now(),
                        "feature_count": feature_count,
                        "row_count": row_count,
                    },
                )
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "category": category,
                        "dashboard_key": DASHBOARD_KEY.get(category, category),
                        "master_path": str(master_path or ""),
                        "row_count": row_count,
                        "feature_count": feature_count,
                        "current_dxi": None,
                        "immediate_target_dxi": None,
                        "mid_target_dxi": None,
                        "long_target_dxi": None,
                        "dxi_gap_immediate": None,
                        "dxi_gap_mid": None,
                        "dxi_gap_long": None,
                        "status": "failed",
                        "error": str(exc),
                        "execution_id": None,
                    }
                )
                status["categories"][category] = {"status": "failed", "error": str(exc)}
                _append_chat(request, f"❌ Subindex DXI failed: <b>{category}</b> — {exc}")

            completed += 1
            status["completed"] = completed
            status["updated_at"] = timezone.now().isoformat()
            request.session["subindex_dxi_status"] = status
            request.session.modified = True
            try:
                request.session.save()
            except Exception:
                pass
    finally:
        if previous_mode is None:
            request.session.pop("sxi_execution_mode", None)
        else:
            request.session["sxi_execution_mode"] = previous_mode
        request.session.modified = True

    export_paths = export_subindex_summary(
        run_id=run_id,
        rows=summary_rows,
        feature_rows=feature_rows,
        metadata={
            "run_id": run_id,
            "task_type": task_type,
            "encoding_mode": "inherited_5_plus_1_from_total",
            "pdf_for_subindex": False,
            "created_at": timezone.now().isoformat(),
            "categories_attempted": list(subindex_masters.keys()),
        },
    )

    # Stamp export path on all category rows
    export_file = export_paths.get("xlsx") or export_paths.get("csv") or ""
    DxiRunResult.objects.filter(run_id=str(run_id), category__in=SUBINDEX_CATEGORIES).update(
        export_path=export_file
    )

    ok_count = sum(1 for r in summary_rows if r.get("status") == "completed")
    all_done = completed >= len(subindex_masters)
    status.update(
        {
            "status": "completed" if all_done else "partial",
            "completed": completed,
            "successful": ok_count,
            "export_path": export_file,
            "export_urls": {
                "xlsx": export_paths.get("xlsx_url") or "",
                "csv": export_paths.get("csv_url") or "",
            },
            "updated_at": timezone.now().isoformat(),
        }
    )
    request.session["subindex_dxi_status"] = status
    request.session["dxi_pipeline_status"] = "complete" if all_done else "partial"
    request.session.modified = True
    try:
        request.session.save()
    except Exception:
        pass

    _append_chat(
        request,
        f"✅ Subindex DXI finished ({ok_count}/{len(summary_rows)} successful).",
    )

    return {
        "completed": completed,
        "successful": ok_count,
        "total": len(subindex_masters),
        "export": export_paths,
        "rows": summary_rows,
        "all_complete": all_done,
        "message": f"Subindex DXI complete ({ok_count}/{len(summary_rows)})",
    }


def load_category_scores_from_db(run_id: str) -> Dict[str, dict]:
    """Map dashboard keys -> {current, target, immediate, mid, long} from DxiRunResult."""
    if not run_id:
        return {}
    scores: Dict[str, dict] = {}
    qs = DxiRunResult.objects.filter(run_id=str(run_id), status="completed")
    for row in qs:
        if row.category == "total":
            key = "total"
        else:
            key = DASHBOARD_KEY.get(row.category)
        if not key:
            continue
        target = row.immediate_dxi if row.immediate_dxi is not None else row.long_dxi
        scores[key] = {
            "current": row.current_dxi,
            "target": target,
            "immediate": row.immediate_dxi,
            "mid": row.mid_dxi,
            "long": row.long_dxi,
            "status": row.status,
            "category": row.category,
        }
    return scores
