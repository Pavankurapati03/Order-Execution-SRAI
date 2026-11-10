import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"}
DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_SCALE = 2
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 900
AUTO_FIX_STATE = {
    "kaleido_install_attempted": False,
    "chrome_install_attempted": False,
    "fallback_kaleido_attempted": False,
}


def _is_frozen_app() -> bool:
    """True inside the PyInstaller EXE (sys.executable is the app, not python.exe)."""
    return bool(getattr(sys, "frozen", False)) or bool(getattr(sys, "_MEIPASS", None))


def _log(logger: Optional[logging.Logger], level: int, message: str) -> None:
    active_logger = logger or LOGGER
    active_logger.log(level, message)


def _safe_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _major_version(version_text: Optional[str]) -> int:
    if not version_text:
        return 0
    try:
        return int(str(version_text).split(".", 1)[0])
    except Exception:
        return 0


def _find_chrome_path() -> Optional[str]:
    env_candidates = [
        os.environ.get("PLOTLY_CHROME_PATH"),
        os.environ.get("CHROME_PATH"),
        os.environ.get("CHROMIUM_PATH"),
    ]
    for candidate in env_candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))

    which_candidates = [
        "chrome",
        "chromium",
        "chromium-browser",
        "google-chrome",
        "msedge",
        "microsoft-edge",
    ]
    for candidate in which_candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    if os.name == "nt":
        windows_candidates = [
            Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
        ]
        for candidate in windows_candidates:
            if candidate.exists():
                return str(candidate)

    return None


@lru_cache(maxsize=1)
def inspect_plotly_export_environment() -> Dict[str, Any]:
    plotly_version = _safe_version("plotly")
    kaleido_version = _safe_version("kaleido")
    chrome_path = _find_chrome_path()
    python_version = platform.python_version()

    issues: List[str] = []
    if not plotly_version:
        issues.append("Plotly is not installed.")
    if not kaleido_version:
        issues.append("Kaleido is not installed.")
    if _major_version(kaleido_version) >= 1 and not chrome_path:
        issues.append("Kaleido 1.x requires Chrome/Chromium, but no browser executable was found.")
    if _major_version(plotly_version) and _major_version(plotly_version) < 6 and _major_version(kaleido_version) >= 1:
        issues.append("Older Plotly with Kaleido 1.x can be unstable; consider Kaleido 0.2.1 fallback.")
    if platform.system().lower() == "windows" and _major_version(kaleido_version) >= 1:
        issues.append("Windows + Kaleido 1.x may require extra fallback protection.")

    return {
        "python_version": python_version,
        "platform": platform.platform(),
        "plotly_version": plotly_version,
        "kaleido_version": kaleido_version,
        "chrome_path": chrome_path,
        "issues": issues,
    }


def _clear_environment_cache() -> None:
    inspect_plotly_export_environment.cache_clear()


def _run_subprocess(args: List[str], timeout_seconds: int, logger: Optional[logging.Logger]) -> Tuple[bool, str]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout_seconds}s: {' '.join(args[:4])}"
    except Exception as exc:
        return False, f"Command failed to start: {exc}"

    combined = "\n".join(filter(None, [completed.stdout.strip(), completed.stderr.strip()])).strip()
    if completed.returncode != 0:
        return False, combined or f"Command exited with code {completed.returncode}"
    return True, combined


def _pip_install(package_spec: str, timeout_seconds: int, logger: Optional[logging.Logger]) -> bool:
    _log(logger, logging.INFO, f"Attempting dependency install: {package_spec}")
    ok, output = _run_subprocess(
        [sys.executable, "-m", "pip", "install", package_spec],
        timeout_seconds=timeout_seconds,
        logger=logger,
    )
    if ok:
        _log(logger, logging.INFO, f"Dependency install success: {package_spec}")
        _clear_environment_cache()
        return True
    _log(logger, logging.WARNING, f"Dependency install failed for {package_spec}: {output}")
    return False


def _attempt_chrome_install(timeout_seconds: int, logger: Optional[logging.Logger]) -> Optional[str]:
    script = (
        "import json, plotly.io as pio\n"
        "result = {'path': None}\n"
        "try:\n"
        "    result['path'] = pio.get_chrome()\n"
        "except Exception as first_error:\n"
        "    result['plotly_error'] = repr(first_error)\n"
        "    try:\n"
        "        import kaleido\n"
        "        if hasattr(kaleido, 'get_chrome_sync'):\n"
        "            result['path'] = kaleido.get_chrome_sync()\n"
        "    except Exception as second_error:\n"
        "        result['kaleido_error'] = repr(second_error)\n"
        "print(json.dumps(result))\n"
    )
    ok, output = _run_subprocess([sys.executable, "-c", script], timeout_seconds, logger)
    if not ok:
        _log(logger, logging.WARNING, f"Chrome install helper failed: {output}")
        return None

    try:
        payload = json.loads(output.splitlines()[-1])
    except Exception:
        _log(logger, logging.WARNING, f"Chrome install helper returned unreadable output: {output}")
        return None

    path = payload.get("path")
    if path and Path(path).exists():
        _clear_environment_cache()
        _log(logger, logging.INFO, f"Chrome found or installed: {path}")
        return str(path)

    _log(logger, logging.WARNING, f"Chrome install was attempted but no executable was produced: {payload}")
    return None


def ensure_plotly_export_ready(
    logger: Optional[logging.Logger] = None,
    timeout_seconds: int = 180,
    allow_auto_fix: bool = True,
) -> Dict[str, Any]:
    info = inspect_plotly_export_environment()
    _log(
        logger,
        logging.INFO,
        "Plotly export environment: "
        f"python={info['python_version']} plotly={info['plotly_version']} "
        f"kaleido={info['kaleido_version']} chrome={info['chrome_path'] or 'missing'}",
    )

    # Never pip-install or spawn sys.executable from the frozen EXE — that
    # re-launches sriya_chatbot.exe and dumps the user back on /license/.
    if _is_frozen_app():
        _log(
            logger,
            logging.INFO,
            "Frozen EXE: skipping Kaleido/Chrome auto-fix; using matplotlib PNG export.",
        )
        return info

    if info["kaleido_version"]:
        try:
            import kaleido  # noqa: F401

            _log(logger, logging.INFO, "Kaleido initialized")
        except Exception as exc:
            _log(logger, logging.WARNING, f"Kaleido import failed during initialization: {exc}")

    if not allow_auto_fix:
        return inspect_plotly_export_environment()

    if not info["kaleido_version"] and not AUTO_FIX_STATE["kaleido_install_attempted"]:
        AUTO_FIX_STATE["kaleido_install_attempted"] = True
        if not _pip_install("kaleido==1.2.0", timeout_seconds, logger):
            _pip_install("kaleido==0.2.1", timeout_seconds, logger)
        info = inspect_plotly_export_environment()
        if info["kaleido_version"]:
            try:
                import kaleido  # noqa: F401

                _log(logger, logging.INFO, "Kaleido initialized")
            except Exception as exc:
                _log(logger, logging.WARNING, f"Kaleido import failed after install: {exc}")

    info = inspect_plotly_export_environment()
    if (
        _major_version(info["kaleido_version"]) >= 1
        and not info["chrome_path"]
        and not AUTO_FIX_STATE["chrome_install_attempted"]
    ):
        AUTO_FIX_STATE["chrome_install_attempted"] = True
        _attempt_chrome_install(timeout_seconds, logger)
        info = inspect_plotly_export_environment()

    needs_kaleido_fallback = (
        _major_version(info["kaleido_version"]) >= 1
        and (
            not info["chrome_path"]
            or _major_version(info["plotly_version"]) < 6
        )
    )
    if needs_kaleido_fallback and not AUTO_FIX_STATE["fallback_kaleido_attempted"]:
        AUTO_FIX_STATE["fallback_kaleido_attempted"] = True
        _log(
            logger,
            logging.WARNING,
            "Detected a potentially unstable Plotly/Kaleido setup; attempting fallback to kaleido==0.2.1",
        )
        if _pip_install("kaleido==0.2.1", timeout_seconds, logger):
            info = inspect_plotly_export_environment()

    return info


def _resolve_output_paths(filename: str, html_path: Optional[str] = None) -> Tuple[Path, Path]:
    output_path = Path(filename)
    suffix = output_path.suffix.lower()

    if html_path:
        resolved_html = Path(html_path)
    elif suffix == ".html":
        resolved_html = output_path
    else:
        resolved_html = output_path.with_suffix(".html")

    if suffix in IMAGE_SUFFIXES:
        resolved_image = output_path
    else:
        resolved_image = resolved_html.with_suffix(".png")

    return resolved_html, resolved_image


def _serialize_figure_json(fig: Any) -> Path:
    handle, temp_path = tempfile.mkstemp(prefix="plotly_export_", suffix=".json")
    os.close(handle)
    temp_file = Path(temp_path)
    temp_file.write_text(fig.to_json(), encoding="utf-8")
    return temp_file


def _child_export_script() -> str:
    return (
        "import json, os, sys, traceback\n"
        "from pathlib import Path\n"
        "import plotly.io as pio\n"
        "fig_path, image_path, image_format, width, height, scale, method = sys.argv[1:8]\n"
        "width = int(width)\n"
        "height = int(height)\n"
        "scale = float(scale)\n"
        "result = {'ok': False, 'method': method, 'image_path': image_path}\n"
        "try:\n"
        "    fig = pio.from_json(Path(fig_path).read_text(encoding='utf-8'))\n"
        "    if method == 'write_image':\n"
        "        fig.write_image(image_path, format=image_format, width=width, height=height, scale=scale)\n"
        "    else:\n"
        "        image_bytes = fig.to_image(format=image_format, width=width, height=height, scale=scale)\n"
        "        Path(image_path).write_bytes(image_bytes)\n"
        "    image_file = Path(image_path)\n"
        "    result['ok'] = image_file.exists() and image_file.stat().st_size > 0\n"
        "    result['bytes'] = image_file.stat().st_size if image_file.exists() else 0\n"
        "except Exception as exc:\n"
        "    result['error'] = repr(exc)\n"
        "    result['traceback'] = traceback.format_exc()\n"
        "print(json.dumps(result))\n"
        "sys.exit(0 if result.get('ok') else 1)\n"
    )


def _export_with_subprocess(
    fig: Any,
    image_path: Path,
    image_format: str,
    width: int,
    height: int,
    scale: int,
    timeout_seconds: int,
    method: str,
    logger: Optional[logging.Logger],
) -> Tuple[bool, Dict[str, Any]]:
    # Critical: in the PyInstaller EXE, sys.executable is sriya_chatbot.exe.
    # Spawning it with -c restarts the whole app (new Django server + /license/).
    if _is_frozen_app():
        message = (
            f"{method} skipped in frozen EXE (sys.executable would relaunch the app); "
            "using matplotlib fallback"
        )
        _log(logger, logging.WARNING, message)
        return False, {"ok": False, "error": message, "method": method}

    temp_json = _serialize_figure_json(fig)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _child_export_script(),
                str(temp_json),
                str(image_path),
                image_format,
                str(width),
                str(height),
                str(scale),
                method,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        message = f"{method} timed out after {timeout_seconds}s"
        _log(logger, logging.WARNING, message)
        return False, {"ok": False, "error": message, "method": method}
    except Exception as exc:
        message = f"{method} subprocess failed to start: {exc}"
        _log(logger, logging.WARNING, message)
        return False, {"ok": False, "error": message, "method": method}
    finally:
        temp_json.unlink(missing_ok=True)

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    payload: Dict[str, Any]
    try:
        payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except Exception:
        payload = {"ok": False, "error": stdout or stderr or f"{method} failed with code {completed.returncode}"}

    if completed.returncode == 0 and image_path.exists() and image_path.stat().st_size > 0:
        _log(logger, logging.INFO, f"{method} export success: {image_path}")
        return True, payload

    error_text = payload.get("error") or stderr or stdout or f"{method} failed with code {completed.returncode}"
    _log(logger, logging.WARNING, f"{method} export failed: {error_text}")
    return False, payload


def _strip_html(text: Any) -> str:
    raw = str(text or "")
    # Plotly uses HTML in titles/legend; matplotlib shows tags literally.
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", "", raw)
    return raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()


def _decode_plotly_binary(values: Any) -> List[Any]:
    """Decode Plotly binary-encoded arrays ({dtype, bdata}) used for long traces."""
    if not isinstance(values, dict):
        return []
    bdata = values.get("bdata")
    dtype = values.get("dtype")
    if not bdata or not dtype:
        return []
    try:
        import base64
        import numpy as np

        raw = base64.b64decode(bdata)
        arr = np.frombuffer(raw, dtype=np.dtype(dtype))
        shape = values.get("shape")
        if shape:
            try:
                if isinstance(shape, str):
                    shape = tuple(int(part) for part in shape.replace(",", " ").split() if part)
                arr = arr.reshape(shape)
            except Exception:
                pass
        return arr.tolist() if hasattr(arr, "tolist") else list(arr)
    except Exception:
        return []


def _trace_values_to_list(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, dict) and ("bdata" in values or "dtype" in values):
        decoded = _decode_plotly_binary(values)
        if decoded:
            return decoded
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if hasattr(values, "tolist"):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [values]


def _render_matplotlib_fallback(fig: Any, image_path: Path, title: str, logger: Optional[logging.Logger]) -> str:
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dict = fig.to_dict()
    traces = figure_dict.get("data", []) or []
    layout = figure_dict.get("layout", {}) or {}
    layout_title = layout.get("title", {}) or {}
    chart_title = layout_title.get("text") if isinstance(layout_title, dict) else str(layout_title or title)
    chart_title = _strip_html(chart_title)

    xaxis = layout.get("xaxis", {}) or {}
    yaxis = layout.get("yaxis", {}) or {}
    xaxis_title = xaxis.get("title", {}) or {}
    yaxis_title = yaxis.get("title", {}) or {}
    x_label = xaxis_title.get("text") if isinstance(xaxis_title, dict) else str(xaxis_title or "")
    y_label = yaxis_title.get("text") if isinstance(yaxis_title, dict) else str(yaxis_title or "")
    x_label = _strip_html(x_label)
    y_label = _strip_html(y_label)

    fig_mat, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    fig_mat.patch.set_facecolor("white")
    ax.set_facecolor("#eff1fe")
    for spine in ax.spines.values():
        spine.set_color("#9585e6")
        spine.set_linewidth(1.0)
    rendered = False

    for idx, trace in enumerate(traces):
        trace_type = str(trace.get("type") or "scatter").lower()
        trace_name = _strip_html(trace.get("name") or f"Series {idx + 1}")
        marker = trace.get("marker", {}) or {}
        line = trace.get("line", {}) or {}
        color = marker.get("color") or line.get("color") or "#1f77b4"
        if isinstance(color, (list, tuple, np.ndarray, pd.Series, dict)) or color is None:
            color = "#1f77b4"

        if trace_type == "pie":
            labels = [_strip_html(value) for value in _trace_values_to_list(trace.get("labels"))]
            values = pd.to_numeric(pd.Series(_trace_values_to_list(trace.get("values"))), errors="coerce").fillna(0)
            if labels and float(values.sum()) > 0:
                # Plotly pie uses marker.colors (list). Preserve intended palette
                # (e.g. darkgreen/darkred from SXI tv_eda) instead of matplotlib defaults.
                raw_colors = marker.get("colors")
                if raw_colors is None:
                    raw_colors = marker.get("color")
                pie_colors = _trace_values_to_list(raw_colors) if raw_colors is not None else []
                pie_colors = [c for c in pie_colors if c not in (None, "")]
                if len(pie_colors) < len(labels):
                    # Safe buyer/non-buyer style defaults when colors were dropped in export.
                    defaults = ["darkgreen", "darkred", "#4F81BD", "#F5B041"]
                    while len(pie_colors) < len(labels):
                        pie_colors.append(defaults[len(pie_colors) % len(defaults)])
                ax.clear()
                wedges, _texts, autotexts = ax.pie(
                    values.tolist(),
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=pie_colors[: len(labels)],
                )
                for autotext in autotexts:
                    autotext.set_color("white")
                ax.axis("equal")
                rendered = True
                break
            continue

        if trace_type == "histogram":
            raw_values = trace.get("x")
            if raw_values is None:
                raw_values = trace.get("y")
            values = pd.to_numeric(pd.Series(_trace_values_to_list(raw_values)), errors="coerce").dropna()
            if not values.empty:
                bins = min(30, max(10, int(np.sqrt(len(values)))))
                ax.hist(values.tolist(), bins=bins, alpha=0.7, label=trace_name, color=color)
                rendered = True
            continue

        if trace_type == "heatmap":
            z_data = trace.get("z")
            if z_data is not None:
                if isinstance(z_data, dict) and "bdata" in z_data:
                    import base64
                    dtype_map = {"i1": np.int8, "i2": np.int16, "i4": np.int32,
                                 "u1": np.uint8, "u2": np.uint16, "f4": np.float32, "f8": np.float64}
                    dt = dtype_map.get(z_data.get("dtype", "f8"), np.float64)
                    raw = base64.b64decode(z_data["bdata"])
                    flat = np.frombuffer(raw, dtype=dt)
                    shape = tuple(int(s) for s in z_data["shape"].split(","))
                    z_arr = flat.reshape(shape).astype(float)
                else:
                    z_arr = np.array(z_data, dtype=float)
                text_data = trace.get("text")
                cs_raw = trace.get("colorscale") or [[0, "red"], [0.5, "white"], [1, "lightgreen"]]
                from matplotlib.colors import LinearSegmentedColormap
                cs_colors = []
                for entry in cs_raw:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        cs_colors.append(str(entry[1]))
                    elif isinstance(entry, str):
                        cs_colors.append(entry)
                if len(cs_colors) < 2:
                    cs_colors = ["red", "white", "lightgreen"]
                cmap = LinearSegmentedColormap.from_list("custom", cs_colors, N=256)
                ax.clear()
                im = ax.imshow(z_arr, cmap=cmap, aspect="auto")
                fig_mat.colorbar(im, ax=ax, shrink=0.8)
                x_ticks = trace.get("x") or list(range(z_arr.shape[1]))
                y_ticks = trace.get("y") or list(range(z_arr.shape[0]))
                ax.set_xticks(range(len(x_ticks)))
                ax.set_xticklabels([str(t) for t in x_ticks], fontsize=11)
                ax.set_yticks(range(len(y_ticks)))
                ax.set_yticklabels([str(t) for t in y_ticks], fontsize=11)
                for i in range(z_arr.shape[0]):
                    for j in range(z_arr.shape[1]):
                        label = ""
                        if text_data and i < len(text_data) and j < len(text_data[i]):
                            label = str(text_data[i][j]).replace("<br>", "\n")
                        else:
                            label = str(int(z_arr[i, j]))
                        ax.text(j, i, label, ha="center", va="center",
                                fontsize=13, fontweight="bold", color="black")
                rendered = True
            continue

        if trace_type in {"bar", "scatter", "box", "violin"}:
            x_values = _trace_values_to_list(trace.get("x"))
            y_values = _trace_values_to_list(trace.get("y"))

            if trace_type in {"box", "violin"} and not y_values:
                y_values = x_values
                x_values = list(range(1, len(y_values) + 1))

            if not y_values:
                continue

            y_numeric = pd.to_numeric(pd.Series(y_values), errors="coerce")
            x_numeric = pd.to_numeric(pd.Series(x_values), errors="coerce")
            filtered_x: List[Any] = []
            filtered_y: List[float] = []
            for position, y_value in enumerate(y_numeric.tolist()):
                if pd.isna(y_value):
                    continue
                if position < len(x_numeric) and not pd.isna(x_numeric.iloc[position]):
                    x_value = float(x_numeric.iloc[position])
                else:
                    x_value = x_values[position] if position < len(x_values) else position
                    if x_value is None or (isinstance(x_value, float) and pd.isna(x_value)):
                        x_value = position
                filtered_x.append(x_value)
                filtered_y.append(float(y_value))

            if not filtered_y:
                continue

            if trace_type == "bar":
                ax.bar(filtered_x, filtered_y, label=trace_name, color=color, alpha=0.85)
            else:
                mode = str(trace.get("mode") or "lines").lower()
                symbol = str(marker.get("symbol") or "circle").lower()
                marker_style = "x" if "x" in symbol else ("o" if "circle" in symbol or symbol == "circle" else "o")
                line_width = float(line.get("width") or 2)
                dash = str(line.get("dash") or "solid").lower()
                linestyle = {
                    "solid": "-",
                    "dash": "--",
                    "dot": ":",
                    "dashdot": "-.",
                    "longdash": "--",
                    "longdashdot": "-.",
                }.get(dash, "-")
                # Skip invisible legend-only annotation traces
                if not filtered_x and not filtered_y:
                    continue
                if "markers" in mode and "lines" not in mode:
                    ax.scatter(
                        filtered_x,
                        filtered_y,
                        label=trace_name,
                        color=color,
                        s=60,
                        marker=marker_style,
                        zorder=5,
                        alpha=float(marker.get("opacity") or trace.get("opacity") or 0.75),
                    )
                elif "lines" in mode and "markers" not in mode:
                    # Perfect-prediction / ±error / trend lines — draw behind markers
                    ax.plot(
                        filtered_x,
                        filtered_y,
                        label=trace_name,
                        color=color,
                        linewidth=line_width,
                        linestyle=linestyle,
                        zorder=2,
                    )
                else:
                    ax.plot(
                        filtered_x,
                        filtered_y,
                        label=trace_name,
                        color=color,
                        linewidth=line_width,
                        linestyle=linestyle,
                        marker=marker_style if "markers" in mode else None,
                        markersize=6,
                        zorder=3,
                    )
            rendered = True

    if not rendered:
        raise ValueError("No supported Plotly traces were available for matplotlib fallback.")

    ax.set_title(str(chart_title or title), fontsize=13, fontweight="bold", pad=12)
    if x_label:
        ax.set_xlabel(str(x_label), fontsize=10, fontweight="bold")
    if y_label:
        ax.set_ylabel(str(y_label), fontsize=10, fontweight="bold")
    if len(traces) > 1 and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, which="major", color="#9585e6", linestyle="-", linewidth=0.6, alpha=0.55)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#9585e6", linestyle="-", linewidth=0.3, alpha=0.25)
    plt.tight_layout()
    fig_mat.savefig(image_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig_mat)
    _log(logger, logging.INFO, f"Matplotlib fallback export success: {image_path}")
    return "matplotlib_fallback"


def _render_placeholder_image(image_path: Path, title: str, logger: Optional[logging.Logger]) -> str:
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig_placeholder, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.40, "Chart preview could not be rendered automatically.", ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.28, "The interactive HTML chart was still saved for this run.", ha="center", va="center", fontsize=11)
    fig_placeholder.savefig(image_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_placeholder)
    _log(logger, logging.WARNING, f"Placeholder image created: {image_path}")
    return "placeholder_image"


def _should_try_kaleido_021(env_info: Dict[str, Any], error_chain: List[str]) -> bool:
    if _major_version(env_info.get("kaleido_version")) < 1:
        return False

    combined = " | ".join(error_chain).lower()
    instability_markers = [
        "browserfailederror",
        "the browser seemed to close immediately",
        "chrome",
        "chromium",
        "choreo_get_chrome",
        "kaleido",
    ]
    return any(marker in combined for marker in instability_markers)


def _write_html_with_fallback(fig: Any, html_path: Path, logger: Optional[logging.Logger]) -> None:
    try:
        fig.write_html(
            str(html_path),
            config={"responsive": True},
            full_html=True,
            include_plotlyjs=True,
        )
        _log(logger, logging.INFO, f"HTML export success: {html_path}")
        return
    except Exception as exc:
        _log(logger, logging.WARNING, f"Native HTML export failed for {html_path}: {exc}")

    html_path.write_text(
        "<html><body><h2>Plotly HTML export fallback</h2>"
        "<p>The interactive chart could not be written normally for this run.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    _log(logger, logging.WARNING, f"HTML fallback file created: {html_path}")


def save_plot(
    fig: Any,
    filename: str,
    html_path: Optional[str] = None,
    title: str = "Plotly Chart",
    logger: Optional[logging.Logger] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    scale: int = DEFAULT_SCALE,
) -> Dict[str, Any]:
    resolved_html_path, resolved_image_path = _resolve_output_paths(filename, html_path=html_path)
    resolved_html_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_image_path.parent.mkdir(parents=True, exist_ok=True)

    logs: List[str] = []

    def remember(message: str, level: int = logging.INFO) -> None:
        logs.append(message)
        _log(logger, level, message)

    env_info = ensure_plotly_export_ready(logger=logger, timeout_seconds=timeout_seconds, allow_auto_fix=True)
    remember(
        "Environment check complete: "
        f"python={env_info['python_version']}, plotly={env_info['plotly_version']}, "
        f"kaleido={env_info['kaleido_version']}, chrome={env_info['chrome_path'] or 'missing'}"
    )
    if env_info.get("chrome_path"):
        remember(f"Chrome found / installed: {env_info['chrome_path']}")
    else:
        remember("Chrome not found; static export may require fallback handling", logging.WARNING)

    _write_html_with_fallback(fig, resolved_html_path, logger)
    remember(f"HTML file ready: {resolved_html_path}")

    image_format = resolved_image_path.suffix.lower().lstrip(".") or "png"
    fallback_used = "none"
    export_ok = False
    error_chain: List[str] = []

    # Frozen EXE: never attempt Kaleido subprocess (relaunches the app).
    if _is_frozen_app():
        remember(
            "Frozen EXE detected — skipping Kaleido write_image/to_image; "
            "exporting via matplotlib",
            logging.WARNING,
        )
        resolved_image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fallback_used = _render_matplotlib_fallback(fig, resolved_image_path, title, logger)
            export_ok = resolved_image_path.exists() and resolved_image_path.stat().st_size > 0
            if export_ok:
                remember("PNG export success via matplotlib (frozen EXE path)")
        except Exception as exc:
            error_chain.append(repr(exc))
            remember(f"Matplotlib fallback failed: {exc}", logging.ERROR)
    else:
        for attempt in range(1, 3):
            remember(f"Attempting PNG export via write_image (try {attempt}/2)")
            export_ok, payload = _export_with_subprocess(
                fig=fig,
                image_path=resolved_image_path,
                image_format=image_format,
                width=width,
                height=height,
                scale=scale,
                timeout_seconds=timeout_seconds,
                method="write_image",
                logger=logger,
            )
            if export_ok:
                remember(f"PNG export success via write_image on try {attempt}")
                break
            err = str(payload.get("error") or "write_image failed")
            error_chain.append(err)
            # A hung Kaleido process will not recover on immediate retry — skip to fallbacks.
            if "timed out" in err.lower():
                remember("write_image timed out; skipping second attempt and moving to fallbacks", logging.WARNING)
                break

        if (
            not export_ok
            and _should_try_kaleido_021(env_info, error_chain)
            and not AUTO_FIX_STATE["fallback_kaleido_attempted"]
        ):
            AUTO_FIX_STATE["fallback_kaleido_attempted"] = True
            remember("Native export looked unstable; attempting kaleido==0.2.1 compatibility fallback", logging.WARNING)
            if _pip_install("kaleido==0.2.1", timeout_seconds, logger):
                env_info = inspect_plotly_export_environment()
                remember(
                    "Environment refreshed after fallback install: "
                    f"plotly={env_info['plotly_version']}, kaleido={env_info['kaleido_version']}, "
                    f"chrome={env_info['chrome_path'] or 'missing'}"
                )
                export_ok, payload = _export_with_subprocess(
                    fig=fig,
                    image_path=resolved_image_path,
                    image_format=image_format,
                    width=width,
                    height=height,
                    scale=scale,
                    timeout_seconds=timeout_seconds,
                    method="write_image",
                    logger=logger,
                )
                if export_ok:
                    fallback_used = "kaleido_0_2_1"
                    remember("PNG export success after switching to kaleido==0.2.1")
                else:
                    error_chain.append(str(payload.get("error") or "kaleido==0.2.1 retry failed"))

        if not export_ok:
            remember("Fallback triggered: attempting to_image export", logging.WARNING)
            resolved_image_path.parent.mkdir(parents=True, exist_ok=True)
            export_ok, payload = _export_with_subprocess(
                fig=fig,
                image_path=resolved_image_path,
                image_format=image_format,
                width=width,
                height=height,
                scale=scale,
                timeout_seconds=timeout_seconds,
                method="to_image",
                logger=logger,
            )
            if export_ok:
                fallback_used = "to_image"
                remember("PNG export success via to_image fallback")
            else:
                error_chain.append(str(payload.get("error") or "to_image failed"))

        if not export_ok:
            remember("Fallback triggered: attempting matplotlib reconstruction", logging.WARNING)
            resolved_image_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fallback_used = _render_matplotlib_fallback(fig, resolved_image_path, title, logger)
                export_ok = resolved_image_path.exists() and resolved_image_path.stat().st_size > 0
                if export_ok:
                    remember("PNG export success via matplotlib fallback")
            except Exception as exc:
                error_chain.append(str(exc))
                remember(f"Matplotlib fallback failed: {exc}", logging.WARNING)

    if not export_ok:
        resolved_image_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_used = _render_placeholder_image(resolved_image_path, title, logger)
        export_ok = resolved_image_path.exists() and resolved_image_path.stat().st_size > 0
        if export_ok:
            remember("PNG guarantee satisfied with placeholder image", logging.WARNING)

    explanation = (
        f"HTML saved to {resolved_html_path}. PNG saved to {resolved_image_path} via {fallback_used}."
        if export_ok and fallback_used != "none"
        else f"HTML saved to {resolved_html_path}. PNG saved to {resolved_image_path} via native Plotly export."
    )

    if error_chain:
        explanation += f" Issues observed: {' | '.join(error_chain)}"

    return {
        "ok": export_ok,
        "html_path": str(resolved_html_path),
        "image_path": str(resolved_image_path),
        "png_path": str(resolved_image_path),
        "message": explanation,
        "fallback_used": fallback_used,
        "logs": logs,
        "environment": env_info,
        "errors": error_chain,
    }
