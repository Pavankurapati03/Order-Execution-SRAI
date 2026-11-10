"""Persist Agent-2 target configuration outside the Django session cookie.

Concurrent requests (report status polling, batch runner) can overwrite the
session and drop ``final_target_info``. Disk backup makes execution resilient.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.conf import settings


def _session_dir(request) -> Path | None:
    """Prefer run_id so target info survives Django session races/reloads."""
    run_id = str(request.session.get("run_id") or "").strip()
    if run_id:
        path = Path(settings.MEDIA_ROOT) / "sessions" / f"run_{run_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    key = getattr(request.session, "session_key", None) or ""
    if not key:
        try:
            request.session.save()
        except Exception:
            return None
        key = request.session.session_key or ""
    if not key:
        return None
    path = Path(settings.MEDIA_ROOT) / "sessions" / str(key)
    path.mkdir(parents=True, exist_ok=True)
    return path


def persist_final_target_info(request, target_info: dict[str, Any]) -> dict[str, Any]:
    info = dict(target_info or {})
    request.session["final_target_info"] = info
    request.session["target_info"] = info
    request.session.modified = True
    try:
        request.session.save()
    except Exception:
        pass

    folder = _session_dir(request)
    if folder is not None:
        try:
            (folder / "final_target_info.json").write_text(
                json.dumps(info, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[target_info] disk persist failed: {exc}")
    return info


def load_final_target_info(request) -> dict[str, Any] | None:
    for key in ("final_target_info", "target_info"):
        value = request.session.get(key)
        if isinstance(value, dict) and value.get("Target Outcome"):
            return dict(value)

    folder = _session_dir(request)
    if folder is None:
        return None
    path = folder / "final_target_info.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and data.get("Target Outcome"):
        request.session["final_target_info"] = data
        request.session.modified = True
        return dict(data)
    return None


def build_target_info_from_pipeline(request) -> dict[str, Any] | None:
    """Last-resort reconstruction from Agent-1 / pipeline session keys."""
    target = (
        request.session.get("target_column")
        or request.session.get("selected_text_target")
        or ""
    )
    target = str(target).strip()
    if not target:
        return None

    task_type = str(request.session.get("task_type") or "").strip().lower()
    is_regression = task_type in {"regression", "numeric", "continuous"}
    if is_regression:
        return {
            "Target Outcome": target,
            "Target Outcome Type": "Numeric",
            "Task Type": "regression",
            "Selected Outcome": "Above_mean",
            "Selected Outcome Meaning": "Good",
            "Good Outcome Value": "Above_mean",
            "Bad Outcome Value": "Below_mean",
            "Good Outcome": "Above_mean",
            "Bad Outcome": "Below_mean",
            "Target Outcome Improvement": 20,
            "Target Outcome change": "increasing",
        }

    return {
        "Target Outcome": target,
        "Target Outcome Type": "Categorical",
        "Task Type": "classification",
        "Selected Outcome": "1",
        "Selected Outcome Meaning": "Yes",
        "Good Outcome Label": "1",
        "Bad Outcome Label": "0",
        "Good Outcome Value": 1,
        "Bad Outcome Value": 0,
        "Good Outcome": "1",
        "Bad Outcome": "0",
        "Target Outcome Improvement": 20,
        "Target Outcome change": "increasing",
    }


def resolve_final_target_info(request) -> dict[str, Any] | None:
    info = load_final_target_info(request)
    if info:
        return info
    built = build_target_info_from_pipeline(request)
    if built:
        return persist_final_target_info(request, built)
    return None
