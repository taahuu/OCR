"""Best-effort reconstruction of simple tables from OCR word coordinates.

This intentionally only returns a table when repeated horizontal rows and
column starts are detected. It is not a guarantee for borderless/complex tables.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any


class TableExtractionService:
    def from_ocr_words(self, words: list[Any], page_number: int = 1) -> list[dict[str, Any]]:
        lines: dict[tuple[int, int, int], list[Any]] = defaultdict(list)
        for word in words:
            get = (lambda k: word.get(k)) if isinstance(word, dict) else (lambda k: getattr(word, k, None))
            key = (int(get("block_number") or 0), int(get("paragraph_number") or 0), int(get("line_number") or 0))
            lines[key].append(word)

        row_cells: list[tuple[int, list[str]]] = []
        for line_words in lines.values():
            line_words.sort(key=lambda w: int((w.get("left") if isinstance(w, dict) else w.left) or 0))
            cells: list[str] = []
            current = ""
            previous_right: int | None = None
            for word in line_words:
                get = (lambda k: word.get(k)) if isinstance(word, dict) else (lambda k: getattr(word, k, None))
                left = int(get("left") or 0)
                width = int(get("width") or 0)
                token = str(get("text") or "").strip()
                if not token:
                    continue
                gap = left - previous_right if previous_right is not None else 0
                if current and gap > 45:
                    cells.append(current)
                    current = token
                else:
                    current = f"{current} {token}".strip()
                previous_right = left + width
            if current:
                cells.append(current)
            if len(cells) >= 2:
                row_cells.append((min(int((w.get("top") if isinstance(w, dict) else w.top) or 0) for w in line_words), cells))

        if len(row_cells) < 3:
            return []
        row_cells.sort(key=lambda item: item[0])
        # Require repeated column counts; otherwise ordinary prose is too likely.
        counts = [len(cells) for _, cells in row_cells]
        common_count = max(set(counts), key=counts.count)
        candidates = [cells for _, cells in row_cells if len(cells) == common_count]
        if len(candidates) < 3 or common_count < 2:
            return []
        return [{"table_number": 1, "page_number": page_number, "headers": candidates[0], "rows": candidates[1:], "detection_method": "ocr_coordinate_heuristic"}]
