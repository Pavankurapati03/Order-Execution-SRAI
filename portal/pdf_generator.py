"""
Robust Headless Browser PDF Generator for Quantellix Automated EDA Reports.
Uses Chrome / Edge headless Blink engine for pixel-perfect CSS Grid & Flexbox rendering,
with automatic fallback to PyMuPDF.
"""

import os
import shutil
import subprocess
import logging
from typing import Optional

logger = logging.getLogger("quantellix.pdf_generator")

def find_browser_executable() -> Optional[str]:
    """Finds Chrome, Edge, or Chromium on the host system."""
    candidates = [
        # Standard 64-bit Chrome & Edge on Windows
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        # Local AppData installations
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        # PATH-based executables
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("msedge"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None

def inline_images_as_base64(html_content: str, html_dir: str) -> str:
    """
    Replaces local image URLs in HTML with base64 data URIs so that headless Chrome/Edge
    renders all images in the PDF without needing network access or file:// resolution.
    """
    import base64
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def repl(match):
        src = match.group(1)
        if src.startswith("data:"):
            return match.group(0)

        clean_src = src.split("?")[0]
        candidates = []
        if clean_src.startswith("/portal/uploads/"):
            candidates.append(os.path.join(project_root, "portal", "uploads", clean_src[len("/portal/uploads/"):]))
        elif clean_src.startswith("/uploads/"):
            candidates.append(os.path.join(project_root, "portal", "uploads", clean_src[len("/uploads/"):]))
        elif clean_src.startswith("/static/"):
            candidates.append(os.path.join(project_root, "portal", "static", clean_src[len("/static/"):]))
        elif clean_src.startswith("file:///"):
            candidates.append(clean_src[8:].replace("/", os.sep))
        else:
            candidates.append(os.path.join(html_dir, clean_src))
            candidates.append(os.path.join(project_root, clean_src.lstrip("/\\")))

        for cand in candidates:
            if os.path.exists(cand) and os.path.isfile(cand):
                try:
                    with open(cand, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    ext = os.path.splitext(cand)[1].lower()
                    mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in [".jpg", ".jpeg"] else "image/svg+xml")
                    return f'src="data:{mime};base64,{b64}"'
                except Exception as e:
                    logger.warning(f"Failed to inline image {cand}: {e}")
        return match.group(0)

    import re
    return re.sub(r'src=["\']([^"\']+)["\']', repl, html_content)


def convert_html_to_pdf(html_path: str, pdf_path: str, timeout_sec: int = 35) -> bool:
    """
    Converts an HTML file to PDF using headless Chrome/Edge with base64 image inlining.
    Falls back to PyMuPDF if no browser is available or if execution fails.
    """
    html_abs = os.path.abspath(html_path)
    pdf_abs = os.path.abspath(pdf_path)

    if not os.path.exists(html_abs):
        logger.error(f"HTML source file not found: {html_abs}")
        return False

    os.makedirs(os.path.dirname(pdf_abs), exist_ok=True)

    # Inline all images to base64 so headless Chrome/Edge renders all visuals without net/path errors
    temp_html_path = html_abs + ".inlined.html"
    target_html_abs = html_abs
    try:
        with open(html_abs, "r", encoding="utf-8") as f:
            raw_html = f.read()
        inlined_html = inline_images_as_base64(raw_html, os.path.dirname(html_abs))
        with open(temp_html_path, "w", encoding="utf-8") as f:
            f.write(inlined_html)
        target_html_abs = temp_html_path
    except Exception as e:
        logger.warning(f"Image inlining notice: {e}")

    try:
        browser = find_browser_executable()
        if browser:
            try:
                cmd = [
                    browser,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--allow-file-access-from-files",
                    "--enable-local-file-accesses",
                    "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=3000",
                    f"--print-to-pdf={pdf_abs}",
                    target_html_abs
                ]
                logger.info(f"Generating PDF with headless browser: {browser}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec
                )
                if os.path.exists(pdf_abs) and os.path.getsize(pdf_abs) > 5000:
                    logger.info(f"Successfully generated PDF ({os.path.getsize(pdf_abs)} bytes) via {browser}")
                    return True
                else:
                    logger.warning(f"Browser PDF generation produced invalid file (code {result.returncode}): {result.stderr}")
            except Exception as e:
                logger.error(f"Headless browser PDF conversion failed: {e}")

        # Fallback to PyMuPDF
        logger.info("Falling back to PyMuPDF HTML conversion...")
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            doc = fitz.open(target_html_abs)
            pdf_bytes = doc.convert_to_pdf()
            with open(pdf_abs, "wb") as f:
                f.write(pdf_bytes)
            doc.close()
            return os.path.exists(pdf_abs) and os.path.getsize(pdf_abs) > 0
        except Exception as e:
            logger.error(f"PyMuPDF fallback failed: {e}")
            return False
    finally:
        if os.path.exists(temp_html_path):
            try:
                os.remove(temp_html_path)
            except Exception:
                pass
