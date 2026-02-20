from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Set


def is_allowed_file(filename: str, allowed_extensions: Set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions


def extract_text_from_path(path: Path) -> str:
    ext = path.suffix.lower()

    if ext in {".txt", ".md", ".csv", ".xml"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".json":
        raw = path.read_text(encoding="utf-8", errors="ignore")
        try:
            obj = json.loads(raw)
        except Exception:
            return raw
        return json.dumps(obj, indent=2, ensure_ascii=True)

    if ext in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
        raw = re.sub(r"(?s)<[^>]+>", " ", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:
            raise RuntimeError("PDF support requires pypdf. Install: pip install pypdf") from exc

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"[page {i + 1}]\n{text}")
        return "\n\n".join(pages)

    if ext == ".docx":
        try:
            from docx import Document as DocxDocument  # type: ignore
        except Exception as exc:
            raise RuntimeError("DOCX support requires python-docx. Install: pip install python-docx") from exc

        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs if (p.text or "").strip())

    return path.read_text(encoding="utf-8", errors="ignore")
