"""Convert extracted text into a lightweight, stable layout representation."""
from __future__ import annotations
import re
from typing import Any


class LayoutService:
    @staticmethod
    def from_text(text: str) -> dict[str, Any]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        headings = [line for line in lines if len(line) <= 80 and (line.isupper() or line.endswith(":"))]
        return {"paragraphs": paragraphs, "headings": list(dict.fromkeys(headings)), "line_count": len(lines)}
