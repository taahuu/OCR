"""Local, explainable document classification and field extraction.

Invoice extraction is tailored to the Shades tax-invoice layout. It remains
heuristic/local (no LLM or cloud API), preserves the other document extractors,
and returns None or [] when a requested value cannot be located.
"""
from __future__ import annotations

import re
from typing import Any


class InformationExtractionService:
    KEY_VALUE_RE = re.compile(r"^\s*([^:\n]{2,60})\s*:\s*(.*?)\s*$", re.MULTILINE)
    EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
    PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
    DATE_RE = re.compile(r"\b(?:\d{1,2}[-/.][A-Za-z]{3}[-/.]\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b")
    MONEY_RE = re.compile(r"(?:₹|INR|Rs\.?|\$|USD|€|£)\s?[\d,]+(?:\.\d{1,2})?", re.I)
    AMOUNT_RE = r"(?:₹\s*)?\d[\d,]*(?:\.\d{1,2})?"

    def extract(self, text: str, tables: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        clean = text or ""
        kind = self.classify(clean)
        key_values = self._key_values(clean)
        common: dict[str, Any] = {"document_type": kind}
        if kind == "invoice":
            # Invoice response is deliberately restricted to the requested fields.
            common["invoice"] = self._invoice(clean)
        else:
            common.update({
                "key_value_pairs": key_values,
                "emails": list(dict.fromkeys(self.EMAIL_RE.findall(clean))),
                "phone_numbers": list(dict.fromkeys(x.strip() for x in self.PHONE_RE.findall(clean))),
                "dates": list(dict.fromkeys(self.DATE_RE.findall(clean))),
                "monetary_values": list(dict.fromkeys(self.MONEY_RE.findall(clean))),
            })
            if kind == "resume":
                common["resume"] = self._resume(clean, key_values)
            elif kind == "bank_statement":
                common["bank_statement"] = self._bank_statement(clean, key_values, tables or [])
            elif kind == "form":
                common["form"] = {"fields": key_values, "checkbox_markers": self._checkboxes(clean)}
            else:
                common["general"] = {"headings": self._headings(clean), "paragraph_count": len([p for p in re.split(r"\n\s*\n", clean) if p.strip()])}
        common["tables"] = tables or []
        return common

    @staticmethod
    def classify(text: str) -> str:
        t = text.lower()
        if any(x in t for x in ("resume", "curriculum vitae", "professional summary", "work experience", "technical skills")):
            return "resume"
        if any(x in t for x in ("invoice", "invoice number", "bill to", "subtotal", "amount due", "tax invoice")):
            return "invoice"
        if any(x in t for x in ("bank statement", "account statement", "opening balance", "closing balance", "transaction details")):
            return "bank_statement"
        if len(re.findall(r"(?:_{3,}|\.{3,}|\[\s*[xX ]?\s*\])", text)) >= 2 or "please fill" in t:
            return "form"
        return "general_document"

    def _key_values(self, text: str) -> list[dict[str, str]]:
        pairs = []
        for key, value in self.KEY_VALUE_RE.findall(text):
            key, value = key.strip(), value.strip()
            if value and len(key) <= 60:
                pairs.append({"key": key, "value": value})
        return pairs

    @staticmethod
    def _clean_value(value: str | None) -> str | None:
        if value is None:
            return None
        value = re.sub(r"\s+", " ", value).strip(" :.-\t")
        return value or None

    @classmethod
    def _first(cls, patterns: list[str], text: str, flags: int = re.I | re.M) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags)
            if match:
                value = cls._clean_value(match.group(1))
                if value:
                    return value
        return None

    @classmethod
    def _invoice(cls, text: str) -> dict[str, Any]:
        """Extract the selected Shades invoice fields from native text or OCR text."""
        lines = [line.strip() for line in text.splitlines()]
        nonempty = [line for line in lines if line]

        # Invoice header values are frequently placed in adjacent PDF columns.
        invoice_number = cls._first([r"\b(\d{4}-\d{2}/[A-Z0-9-]+)\b", r"Invoice\s*(?:No\.?|Number|#)\s*[:#-]?\s*([A-Z0-9/-]+)"], text)
        invoice_date = cls._first([rf"{re.escape(invoice_number or '___')}\s+((?:\d{{1,2}}[-/.][A-Za-z]{{3}}[-/.]\d{{2,4}})|(?:\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}))", r"Invoice\s*Date\s*[:#-]?\s*([\w./-]+)"], text)
        delivery_note = cls._first([r"\b(sp:ku)\b", r"Delivery\s*Note[^\n]*\n\s*([A-Z0-9:/-]+)"], text)
        dispatch_doc = cls._first([r"\b(Ch:[A-Z0-9/-]+)\b", r"Dispatch\s*Doc(?:ument)?\s*No\.?[^\n]*\n\s*([A-Z0-9:/-]+)"], text)
        # Seller is the issuer named immediately below the Tax Invoice heading.
        seller_name = None
        for i, line in enumerate(nonempty[:8]):
            if line.strip().lower() == "tax invoice":
                candidate = nonempty[i + 1] if i + 1 < len(nonempty) else ""
                seller_name = re.split(r"\s{2,}", candidate.strip())[0].strip() or None
                break
        seller_address = None
        if seller_name:
            try:
                idx = next(i for i, line in enumerate(nonempty) if line.strip().startswith(seller_name))
                address_lines = []
                for line in nonempty[idx + 1:]:
                    if re.search(r"GSTIN/UIN|State Name|Contact\s*:|E-Mail\s*:|Consignee", line, re.I):
                        break
                    # Native PDF text can merge the seller's left column with invoice metadata.
                    left_column = re.split(r"\s{2,}", line.strip())[0].strip()
                    left_column = re.sub(r"\s+sp:ku\s*$", "", left_column, flags=re.I)
                    left_column = left_column.strip(" ,")
                    if left_column:
                        address_lines.append(left_column)
                seller_address = ", ".join(address_lines) or None
            except StopIteration:
                pass

        gstins = re.findall(r"\b\d{2}[A-Z0-9]{10,13}\b", text, re.I)
        seller_gstin = gstins[0].upper() if gstins else None
        seller_contact = cls._first([r"Contact\s*:\s*(\+?[\d ()-]{8,})"], text)
        seller_email = cls._first([r"E-Mail\s*:\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})"], text)

        # Product rows: serial, description, HSN, GST %, quantity/unit, rate,
        # per-unit label, discount %, line amount. Keep the original printed values.
        items: list[dict[str, Any]] = []
        item_re = re.compile(
            rf"^\s*(\d{{1,3}})\s+(.+?)\s+(\d{{4,8}})\s+(\d{{1,2}}\s*%)\s+"
            rf"([\d,.]+)\s+([A-Za-z]+)\s+({cls.AMOUNT_RE})\s+([A-Za-z]+)\s+"
            rf"([\d,.]+\s*%)\s+({cls.AMOUNT_RE})\s*$", re.I
        )
        for line in text.splitlines():
            match = item_re.match(line)
            if match:
                serial, description, hsn, gst, qty, unit, rate, rate_unit, discount, amount = match.groups()
                items.append({
                    "serial_number": int(serial), "description": cls._clean_value(description),
                    "hsn_sac": hsn, "gst_rate": gst.replace(" ", ""),
                    "quantity": qty.replace(",", ""), "unit": unit,
                    "rate": rate.replace("₹", "").strip(), "rate_unit": rate_unit,
                    "discount": discount.replace(" ", ""), "amount": amount.replace("₹", "").strip(),
                })

        # Summary values use the label on the left and printed amount on the right.
        def amount_after(label: str) -> str | None:
            return cls._first([rf"{label}\s+(?:₹\s*)?({cls.AMOUNT_RE})"], text)

        taxable_value = cls._first([rf"^\s*\d{{4,8}}\s+({cls.AMOUNT_RE})\s+\d{{1,2}}%\s+{cls.AMOUNT_RE}\s+\d{{1,2}}%"], text, re.I | re.M)
        # The invoice's item subtotal is the amount immediately before CGST Output.
        item_subtotal = cls._first([rf"^\s*({cls.AMOUNT_RE})\s*$"], "\n".join(lines[lines.index(next((x for x in lines if re.search(r"CGST Output", x, re.I)), ""))-1:]) if any(re.search(r"CGST Output", x, re.I) for x in lines) else "", re.M)
        cgst = amount_after(r"CGST\s+Output")
        sgst = amount_after(r"SGST\s+Output")
        igst = amount_after(r"IGST\s+Output")
        rounded = amount_after(r"Rounded\s+Off")
        grand_total = cls._first([
            rf"Total\s+\d+\s+[A-Za-z]+\s+₹\s*({cls.AMOUNT_RE})",
            rf"Total\s+₹\s*({cls.AMOUNT_RE})",
        ], text)
        amount_words = cls._first([r"Amount\s+Chargeable\s*\(in\s*words\)\s*(?:E\.?\s*&\.?\s*O\.?\s*E\.?)?\s*\n\s*(?:INR\s+)?(.+?\s+Only)"], text)
        total_tax = cls._first([rf"Total\s+Tax\s+Amount\s*[: ]+({cls.AMOUNT_RE})"], text)
        # In this layout total tax is also printed in the HSN summary row.
        if not total_tax:
            tax_values = re.findall(rf"\b\d{{4,8}}\s+{cls.AMOUNT_RE}\s+\d{{1,2}}%\s+{cls.AMOUNT_RE}\s+\d{{1,2}}%\s+{cls.AMOUNT_RE}\s+({cls.AMOUNT_RE})", text, re.I)
            total_tax = tax_values[0] if tax_values else None
        if not total_tax and cgst and sgst:
            try:
                total_tax = f"{float(cgst.replace(',', '')) + float(sgst.replace(',', '')):,.2f}"
            except ValueError:
                pass

        account_holder = cls._first([r"A/c\s*Holder'?s\s*Name\s*:\s*(.+)"], text)
        bank_name = cls._first([r"Bank\s*Name\s*:\s*(.+)"], text)
        account_number = cls._first([r"A/c\s*No\.?\s*:\s*([\dXx* -]+)"], text)
        branch_ifsc = cls._first([r"Branch\s*&\s*IFS\s*Code\s*:\s*(.+)"], text)
        branch, ifsc = (None, None)
        if branch_ifsc:
            parts = re.split(r"\s*&\s*", branch_ifsc, maxsplit=1)
            branch = cls._clean_value(parts[0])
            if len(parts) > 1:
                ifsc = cls._clean_value(parts[1])

        return {
            "invoice_number": invoice_number,
            "invoice_date": invoice_date,
            "delivery_note": delivery_note,
            "dispatch_document_number": dispatch_doc,
            "seller": {
                "name": seller_name, "address": seller_address, "gstin": seller_gstin,
                "contact": seller_contact, "email": seller_email,
            },
            "product_items": items,
            "tax_summary": {
                "taxable_value": taxable_value or item_subtotal,
                "cgst": cgst, "sgst": sgst, "igst": igst,
                "total_tax_amount": total_tax, "rounded_off": rounded,
            },
            "grand_total": grand_total,
            "amount_in_words": amount_words,
            "bank_details": {
                "account_holder_name": account_holder, "bank_name": bank_name,
                "account_number": account_number, "branch": branch, "ifsc": ifsc,
            },
        }

    def _resume(self, text: str, pairs: list[dict[str, str]]) -> dict[str, Any]:
        lines = [x.strip(" •\t") for x in text.splitlines() if x.strip()]
        email = self.EMAIL_RE.search(text)
        phone = self.PHONE_RE.search(text)
        name = next((x for x in lines[:8] if not self.EMAIL_RE.search(x) and not self.PHONE_RE.search(x) and len(x.split()) <= 5 and not re.search(r"resume|curriculum vitae", x, re.I)), None)
        sections = self._sections(text, ("professional summary", "technical skills", "skills", "work experience", "experience", "projects", "education", "certifications"))
        skill_text = sections.get("technical_skills") or sections.get("skills") or ""
        skills = [part.strip(" •-\t") for part in re.split(r"[,|;\n]", skill_text) if part.strip(" •-\t")]
        return {"name": name, "email": email.group(0) if email else None, "phone": phone.group(0).strip() if phone else None, "skills": skills, "experience_entries": [x.strip() for x in sections.get("work_experience", sections.get("experience", "")).splitlines() if x.strip()], "education_entries": [x.strip() for x in sections.get("education", "").splitlines() if x.strip()], "sections": sections, "key_value_pairs": pairs}

    def _bank_statement(self, text: str, pairs: list[dict[str, str]], tables: list[dict[str, Any]]) -> dict[str, Any]:
        account = re.search(r"(?:account\s*(?:number|no)?|a/c\s*(?:no)?)\s*[:#-]?\s*([Xx*\d -]{4,})", text, re.I)
        transactions = []
        for table in tables:
            headers = [str(x).lower() for x in table.get("headers", [])]
            if any(token in " ".join(headers) for token in ("date", "description", "debit", "credit", "balance")):
                for row in table.get("rows", []):
                    transactions.append({headers[i] if i < len(headers) else f"column_{i + 1}": value for i, value in enumerate(row)})
        return {"account_number_as_printed": account.group(1).strip() if account else None, "transactions": transactions, "tables_detected": len(tables), "key_value_pairs": pairs}

    @staticmethod
    def _checkboxes(text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if re.search(r"(?:☒|☑|☐|\[[xX ]\])", line)]

    @staticmethod
    def _headings(text: str) -> list[str]:
        result = []
        for line in text.splitlines():
            value = line.strip()
            if value and len(value) < 80 and (value.isupper() or value.endswith(":")):
                result.append(value.rstrip(":"))
        return list(dict.fromkeys(result))

    @staticmethod
    def _sections(text: str, names: tuple[str, ...]) -> dict[str, str]:
        lines = text.splitlines()
        found: dict[str, str] = {}
        current = None
        chunks: dict[str, list[str]] = {}
        for line in lines:
            stripped = line.strip().lower().rstrip(":")
            matched = next((n for n in names if stripped == n), None)
            if matched:
                current = matched
                chunks.setdefault(current, [])
            elif current:
                chunks[current].append(line.strip())
        for key, values in chunks.items():
            found[key.replace(" ", "_")] = "\n".join(x for x in values if x)
        return found
