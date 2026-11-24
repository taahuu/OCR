"""Extract paragraphs and tables from Office Open XML DOCX files locally."""
from __future__ import annotations
from io import BytesIO
from typing import Any
import zipfile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from app.core.exceptions import FileTooLargeError
from app.core.exceptions import CorruptDocumentError


class DocxService:
    def extract(self, data: bytes) -> tuple[str, list[dict[str, Any]]]:
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                members = archive.infolist()
                if len(members) > 2000 or sum(item.file_size for item in members) > 100 * 1024 * 1024:
                    raise FileTooLargeError("The expanded DOCX content exceeds the allowed processing limit.")
                names = set(archive.namelist())
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise CorruptDocumentError("The DOCX file is invalid or unreadable.")
            document = Document(BytesIO(data))
        except (CorruptDocumentError, FileTooLargeError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise CorruptDocumentError("The DOCX file is corrupted or unreadable.") from exc

        text_parts: list[str] = []
        tables: list[dict[str, Any]] = []
        body = document.element.body
        for child in body.iterchildren():
            if isinstance(child, CT_P):
                value = Paragraph(child, document).text.strip()
                if value:
                    text_parts.append(value)
            elif isinstance(child, CT_Tbl):
                table = Table(child, document)
                rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                if rows:
                    table_data = {
                        "table_number": len(tables) + 1,
                        "headers": rows[0],
                        "rows": rows[1:],
                    }
                    tables.append(table_data)
                    text_parts.append("\n".join(" | ".join(row) for row in rows))

        # Headers and footers are part of the visible document content too.
        for section in document.sections:
            for region in (section.header, section.footer):
                region_text = "\n".join(p.text.strip() for p in region.paragraphs if p.text.strip())
                if region_text:
                    text_parts.append(region_text)
        return "\n\n".join(text_parts), tables
