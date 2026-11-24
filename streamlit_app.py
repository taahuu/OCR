"""Streamlit UI for the OCR Document Extraction API.

Run the FastAPI backend separately, then launch:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Document Intelligence | OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("OCR_API_URL", "http://127.0.0.1:8000").rstrip("/")
EXTRACT_URL = f"{API_BASE_URL}/api/v1/ocr/extract"
HEALTH_URL = f"{API_BASE_URL}/api/v1/health"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1420px;}
      .app-header {padding: 1.35rem 1.6rem; border-radius: 18px; margin-bottom: 1.1rem;
        background: linear-gradient(115deg, #102a43 0%, #176b87 58%, #22a6a1 100%);
        color: #fff; box-shadow: 0 8px 24px rgba(16,42,67,.16);}
      .app-header h1 {color:#fff; font-size:1.8rem; margin:0 0 .25rem 0; font-weight:750;}
      .app-header p {color:rgba(255,255,255,.88); margin:0; font-size:.96rem;}
      .header-chip {display:inline-block; margin-top:.8rem; padding:.25rem .65rem;
        border:1px solid rgba(255,255,255,.35); border-radius:99px; font-size:.78rem;}
      .section-card {padding:1rem 1.15rem; border:1px solid rgba(128,128,128,.22);
        border-radius:14px; margin:.5rem 0 1rem 0;}
      .muted {color: #858b98; font-size: .92rem;}
      .step-strip {display:flex; gap:.5rem; flex-wrap:wrap; margin:.6rem 0 1.1rem 0;}
      .step {padding:.4rem .75rem; border-radius:99px; background:rgba(100,116,139,.12);
        font-size:.82rem; color:inherit;}
      .app-footer {margin-top:2.5rem; padding:1rem .25rem; border-top:1px solid rgba(128,128,128,.25);
        color:#858b98; font-size:.82rem; text-align:center;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2);
          padding: 12px; border-radius: 12px;}
      div[data-testid="stFileUploader"] {border:1px dashed rgba(34,166,161,.65); border-radius:14px; padding:.5rem;}
      div.stButton > button, div.stDownloadButton > button {border-radius:10px; font-weight:600;}
      div[data-testid="stFormSubmitButton"] button {border-radius:10px; font-weight:650; min-height:2.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
      <h1>Invoice Intelligence</h1>
      <p>Smart document processing · OCR extraction · Editable invoice review</p>
      <span class="header-chip">POWERED BY YOUR OCR API</span>
    </div>
    <div class="step-strip">
      <span class="step">① Upload document</span>
      <span class="step">② Extract information</span>
      <span class="step">③ Review & edit</span>
      <span class="step">④ Download JSON</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Workspace")
    st.caption("Invoice and document extraction")
    st.write("**Backend connection**")
    st.code(API_BASE_URL, language=None)
    try:
        health = requests.get(HEALTH_URL, timeout=4)
        if health.ok:
            st.success("Backend is reachable")
        else:
            st.warning(f"Backend responded with HTTP {health.status_code}")
    except requests.RequestException:
        st.error("Backend is not reachable")
        st.caption("Start FastAPI before uploading a document.")
    st.divider()
    st.subheader("How to use")
    st.markdown("1. Upload one document\n2. Wait for extraction\n3. Review and correct fields\n4. Save and download JSON")
    st.divider()
    st.subheader("Supported files")
    st.write("PDF · DOCX · PNG · JPG · JPEG · TIFF · BMP · WEBP")
    st.caption("The API applies its own upload, page and image-size validation.")
    st.divider()
    st.caption("Extracted fields are best-effort. Always verify important information before saving.")

st.markdown(
    """
    <div class="section-card">
      <h3 style="margin:0 0 .3rem 0">Upload your document</h3>
      <div class="muted">Choose a PDF or image. The OCR service will extract the content and prepare fields for review.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Browse and select your invoice or document",
    type=["pdf", "docx", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    help="Supported formats: PDF, DOCX, PNG, JPG, JPEG, TIFF, BMP and WEBP. Extraction starts automatically after upload.",
)

def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)

def add_editable_field(label: str, value: Any, key: str, multiline: bool = False) -> str:
    text_value = as_text(value)
    if multiline or "\n" in text_value or len(text_value) > 100:
        return st.text_area(label, value=text_value, key=key, height=110)
    return st.text_input(label, value=text_value, key=key)

def make_form_fields(structured: dict[str, Any]) -> list[tuple[str, str, Any, bool]]:
    """Create human-readable editable fields from the API's structured_data."""
    kind = structured.get("document_type", "general_document")
    section_name = {
        "resume": "resume",
        "invoice": "invoice",
        "bank_statement": "bank_statement",
        "form": "form",
    }.get(kind)
    section = structured.get(section_name, {}) if section_name else {}
    fields: list[tuple[str, str, Any, bool]] = []

    # Common fields are useful across document categories.
    # Invoice documents show only the invoice-specific fields selected for this workflow.
    common_fields = [] if kind == "invoice" else [
        ("Emails", "emails", structured.get("emails", []), False),
        ("Phone numbers", "phone_numbers", structured.get("phone_numbers", []), False),
        ("Dates found", "dates", structured.get("dates", []), False),
        ("Monetary values", "monetary_values", structured.get("monetary_values", []), False),
    ]
    fields.extend((label, name, value, multi) for label, name, value, multi in common_fields)

    if isinstance(section, dict):
        for name, value in section.items():
            if name in {"key_value_pairs", "sections", "transactions", "tables_detected"}:
                continue
            label = name.replace("_", " ").strip().title()
            multiline = isinstance(value, list) or isinstance(value, dict) or (
                isinstance(value, str) and ("\n" in value or len(value) > 100)
            )
            fields.append((label, f"{section_name}_{name}", value, multiline))

    # Preserve arbitrary labelled fields detected by the API.
    pairs = [] if kind == "invoice" else structured.get("key_value_pairs", [])
    if isinstance(pairs, list):
        for index, pair in enumerate(pairs):
            if isinstance(pair, dict):
                label = str(pair.get("key") or f"Detected field {index + 1}")
                fields.append((label, f"pair_{index}", pair.get("value", ""), False))

    # Form extraction nests key/value pairs under form.fields.
    if kind == "form" and isinstance(section, dict):
        nested_pairs = section.get("fields", [])
        if isinstance(nested_pairs, list):
            for index, pair in enumerate(nested_pairs):
                if isinstance(pair, dict):
                    label = str(pair.get("key") or f"Form field {index + 1}")
                    fields.append((label, f"form_pair_{index}", pair.get("value", ""), False))

    # Remove duplicate widget keys while retaining the first occurrence.
    seen = set()
    unique = []
    for item in fields:
        if item[1] not in seen:
            seen.add(item[1])
            unique.append(item)
    return unique

def extract_document(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    files = {"file": (file_name, file_bytes)}
    response = requests.post(EXTRACT_URL, files=files, timeout=(10, 360))
    if not response.ok:
        try:
            detail = response.json().get("error", {})
            message = detail.get("message") or response.text
            code = detail.get("code", "request_failed")
        except (ValueError, AttributeError):
            message, code = response.text, "request_failed"
        raise RuntimeError(f"{code}: {message}")
    return response.json()

if uploaded is None:
    st.info("👆 Upload a document to get started. Your extracted information will appear here in an editable review form.")
    st.markdown("#### What you can do")
    a, b, c = st.columns(3)
    a.markdown("**📄 Extract**\n\nRead text and invoice fields from supported documents.")
    b.markdown("**✏️ Review**\n\nCorrect OCR mistakes in individual editable fields.")
    c.markdown("**📥 Export**\n\nDownload reviewed information as structured JSON.")
    st.markdown('<div class="app-footer">Invoice Intelligence · Review extracted values before using them for accounting.</div>', unsafe_allow_html=True)
    st.stop()

file_bytes = uploaded.getvalue()
file_fingerprint = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.get("processed_fingerprint") != file_fingerprint:
    with st.spinner("Processing document and extracting information..."):
        try:
            result = extract_document(uploaded.name, file_bytes)
            st.session_state["ocr_result"] = result
            st.session_state["processed_fingerprint"] = file_fingerprint
            st.session_state["reviewed_data"] = None
            # Clear prior widget values so the new document populates the new form.
            for state_key in list(st.session_state.keys()):
                if state_key.startswith("field_"):
                    del st.session_state[state_key]
        except requests.ConnectionError:
            st.error("Could not connect to the OCR API. Check that FastAPI is running and OCR_API_URL is correct.")
            st.stop()
        except requests.Timeout:
            st.error("The extraction request timed out. Try a smaller document or increase the client timeout.")
            st.stop()
        except Exception as exc:
            st.error(f"Document extraction failed: {exc}")
            st.stop()

result = st.session_state.get("ocr_result")
if not result:
    st.stop()

structured = result.get("structured_data") or {}
metadata = result.get("metadata") or {}
doc_kind = structured.get("document_type", "general_document")
status = result.get("status", "unknown")

st.divider()
st.subheader("Extraction overview")
st.caption("A quick summary of the document processing result.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Document type", str(doc_kind).replace("_", " ").title())
m2.metric("Pages", result.get("total_pages", 0))
m3.metric("OCR pages", metadata.get("ocr_pages", 0))
m4.metric("Processing time", f'{metadata.get("processing_time_ms", 0)} ms')

if status == "success":
    st.success(f"Extraction completed: {result.get('filename', uploaded.name)}")
elif status == "partial":
    st.warning("Extraction completed with some page failures. Review the page results and extracted fields.")
else:
    st.error(f"Extraction status: {status}")

if status in {"success", "partial"}:
    raw_ocr_text = result.get("full_text", "")
    if isinstance(raw_ocr_text, str):
        st.download_button(
            "Download OCR Text (.txt)",
            data=raw_ocr_text.encode("utf-8"),
            file_name=f"{os.path.splitext(uploaded.name)[0]}_ocr_text.txt",
            mime="text/plain",
        )

pages = result.get("pages", [])
with st.expander("Page-by-page processing details", expanded=False):
    for page in pages:
        confidence = page.get("confidence")
        confidence_text = f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else "Not provided"
        st.markdown(
            f"**Page {page.get('page_number')}** · Method: `{page.get('method')}` · "
            f"Status: `{page.get('status')}` · Confidence: `{confidence_text}`"
        )
        if page.get("preprocessing_applied"):
            st.caption("Preprocessing: " + ", ".join(page["preprocessing_applied"]))
        if page.get("error"):
            st.warning(page["error"])

st.divider()
st.subheader("Review and edit extracted information")
st.caption("All fields are editable. Check spelling, numbers, GST values and totals before saving.")

fields = make_form_fields(structured)


def invoice_text(value: Any) -> str:
    """Convert one extracted invoice value to an editable text value."""
    return "" if value is None else str(value)


def invoice_input(label: str, value: Any, key: str, *, multiline: bool = False) -> str:
    """Render a single invoice value, never a whole dict/list in one widget."""
    initial = invoice_text(value)
    if multiline:
        return st.text_area(label, value=initial, key=key, height=85)
    return st.text_input(label, value=initial, key=key)


if doc_kind == "invoice":
    invoice = structured.get("invoice") or {}
    seller = invoice.get("seller") or {}
    tax = invoice.get("tax_summary") or {}
    bank = invoice.get("bank_details") or {}
    products = invoice.get("product_items") or []
    if not isinstance(products, list):
        products = []

    with st.form("review_invoice_fields"):
        st.markdown("### 🧾 Invoice Details")
        c1, c2 = st.columns(2)
        with c1:
            reviewed_invoice_number = invoice_input("Invoice Number", invoice.get("invoice_number"), "inv_invoice_number")
            reviewed_delivery_note = invoice_input("Delivery Note", invoice.get("delivery_note"), "inv_delivery_note")
        with c2:
            reviewed_invoice_date = invoice_input("Invoice Date", invoice.get("invoice_date"), "inv_invoice_date")
            reviewed_dispatch_number = invoice_input("Dispatch Document Number", invoice.get("dispatch_document_number"), "inv_dispatch_document_number")

        st.divider()
        st.markdown("### 🏢 Seller Details")
        c1, c2 = st.columns(2)
        with c1:
            reviewed_seller_name = invoice_input("Seller Name", seller.get("name"), "inv_seller_name")
            reviewed_seller_gstin = invoice_input("Seller GSTIN", seller.get("gstin"), "inv_seller_gstin")
            reviewed_seller_contact = invoice_input("Seller Contact", seller.get("contact"), "inv_seller_contact")
        with c2:
            reviewed_seller_address = invoice_input("Seller Address", seller.get("address"), "inv_seller_address", multiline=True)
            reviewed_seller_email = invoice_input("Seller Email", seller.get("email"), "inv_seller_email")

        st.divider()
        st.markdown("### 📦 Product Details")
        st.caption(f"{len(products)} product line(s) detected. Expand each product to review its values.")
        reviewed_products = []
        if products:
            for idx, product in enumerate(products):
                product = product if isinstance(product, dict) else {}
                with st.expander(f"Product {idx + 1} — {invoice_text(product.get('description')) or 'Description not detected'}", expanded=(idx == 0)):
                    c1, c2 = st.columns(2)
                    with c1:
                        description = invoice_input("Product Description", product.get("description"), f"inv_product_{idx}_description", multiline=True)
                        hsn = invoice_input("HSN/SAC", product.get("hsn_sac"), f"inv_product_{idx}_hsn")
                        quantity = invoice_input("Quantity", product.get("quantity"), f"inv_product_{idx}_quantity")
                        rate = invoice_input("Rate", product.get("rate"), f"inv_product_{idx}_rate")
                    with c2:
                        gst_rate = invoice_input("GST Rate", product.get("gst_rate"), f"inv_product_{idx}_gst_rate")
                        unit = invoice_input("Unit", product.get("unit"), f"inv_product_{idx}_unit")
                        discount = invoice_input("Discount", product.get("discount"), f"inv_product_{idx}_discount")
                        amount = invoice_input("Amount", product.get("amount"), f"inv_product_{idx}_amount")
                    reviewed_products.append({
                        "serial_number": product.get("serial_number", idx + 1),
                        "description": description, "hsn_sac": hsn, "gst_rate": gst_rate,
                        "quantity": quantity, "unit": unit, "rate": rate,
                        "discount": discount, "amount": amount,
                    })
        else:
            st.info("No product line items were detected by OCR.")

        st.divider()
        st.markdown("### 🧮 Tax Summary")
        c1, c2 = st.columns(2)
        with c1:
            reviewed_taxable_value = invoice_input("Taxable Value", tax.get("taxable_value"), "inv_taxable_value")
            reviewed_cgst = invoice_input("CGST", tax.get("cgst"), "inv_cgst")
            reviewed_igst = invoice_input("IGST", tax.get("igst"), "inv_igst")
        with c2:
            reviewed_sgst = invoice_input("SGST", tax.get("sgst"), "inv_sgst")
            reviewed_total_tax = invoice_input("Total Tax Amount", tax.get("total_tax_amount"), "inv_total_tax_amount")
            reviewed_rounded_off = invoice_input("Rounded Off", tax.get("rounded_off"), "inv_rounded_off")

        st.divider()
        st.markdown("### 💰 Invoice Total")
        c1, c2 = st.columns(2)
        with c1:
            reviewed_grand_total = invoice_input("Grand Total", invoice.get("grand_total"), "inv_grand_total")
        with c2:
            reviewed_amount_words = invoice_input("Amount in Words", invoice.get("amount_in_words"), "inv_amount_in_words")

        st.divider()
        st.markdown("### 🏦 Bank Details")
        c1, c2 = st.columns(2)
        with c1:
            reviewed_account_holder = invoice_input("Account Holder Name", bank.get("account_holder_name"), "inv_account_holder_name")
            reviewed_account_number = invoice_input("Account Number", bank.get("account_number"), "inv_account_number")
            reviewed_ifsc = invoice_input("IFSC", bank.get("ifsc"), "inv_ifsc")
        with c2:
            reviewed_bank_name = invoice_input("Bank Name", bank.get("bank_name"), "inv_bank_name")
            reviewed_branch = invoice_input("Branch", bank.get("branch"), "inv_branch")

        submitted = st.form_submit_button("Save reviewed information", type="primary", use_container_width=True)

    if submitted:
        reviewed_invoice = {
            "invoice_number": reviewed_invoice_number,
            "invoice_date": reviewed_invoice_date,
            "delivery_note": reviewed_delivery_note,
            "dispatch_document_number": reviewed_dispatch_number,
            "seller": {
                "name": reviewed_seller_name, "address": reviewed_seller_address,
                "gstin": reviewed_seller_gstin, "contact": reviewed_seller_contact,
                "email": reviewed_seller_email,
            },
            "product_items": reviewed_products,
            "tax_summary": {
                "taxable_value": reviewed_taxable_value, "cgst": reviewed_cgst,
                "sgst": reviewed_sgst, "igst": reviewed_igst,
                "total_tax_amount": reviewed_total_tax, "rounded_off": reviewed_rounded_off,
            },
            "grand_total": reviewed_grand_total,
            "amount_in_words": reviewed_amount_words,
            "bank_details": {
                "account_holder_name": reviewed_account_holder, "bank_name": reviewed_bank_name,
                "account_number": reviewed_account_number, "branch": reviewed_branch,
                "ifsc": reviewed_ifsc,
            },
        }
        st.session_state["reviewed_data"] = {
            "filename": result.get("filename", uploaded.name),
            "document_type": doc_kind,
            "invoice": reviewed_invoice,
            "tables": result.get("tables", []),
            "full_text": result.get("full_text", ""),
        }
        st.success("Reviewed invoice information saved. You can now download the JSON file below.")
else:
    if not fields:
        st.info("No structured fields were detected. You can still review and export the raw extracted text below.")

    with st.form("review_extracted_fields"):
        if fields:
            columns = st.columns(2)
            for index, (label, name, value, multiline) in enumerate(fields):
                with columns[index % 2]:
                    add_editable_field(label, value, f"field_{name}", multiline)
        else:
            st.write("No editable structured fields are available for this document.")
        submitted = st.form_submit_button("Save reviewed information", type="primary", use_container_width=True)

    if submitted:
        reviewed = {}
        for label, name, original, multiline in fields:
            reviewed[label] = st.session_state.get(f"field_{name}", as_text(original))
        st.session_state["reviewed_data"] = {
            "filename": result.get("filename", uploaded.name),
            "document_type": doc_kind,
            "fields": reviewed,
            "tables": result.get("tables", []),
            "full_text": result.get("full_text", ""),
        }

reviewed_data = st.session_state.get("reviewed_data")
if reviewed_data:
    st.markdown("### 📤 Export reviewed information")
    st.caption("Your saved edits are included in this download.")
    st.download_button(
        "⬇️ Download reviewed data (JSON)",
        data=json.dumps(reviewed_data, ensure_ascii=False, indent=2),
        file_name=f"{os.path.splitext(uploaded.name)[0]}_reviewed.json",
        mime="application/json",
        use_container_width=True,
    )

tables = [] if doc_kind == "invoice" else (result.get("tables") or [])
if tables:
    st.divider()
    st.subheader("Detected tables")
    for index, table in enumerate(tables, start=1):
        st.markdown(f"**Table {index}**")
        headers = table.get("headers") or []
        rows = table.get("rows") or []
        if headers:
            try:
                st.dataframe(pd.DataFrame(rows, columns=headers), use_container_width=True, hide_index=True)
            except (ValueError, TypeError):
                st.json(table)
        else:
            st.json(table)

with st.expander("View raw OCR text", expanded=False):
    st.text_area("Extracted document text", value=result.get("full_text", ""), height=320, disabled=True)

with st.expander("View complete API response (JSON)", expanded=False):
    st.json(result)
    st.download_button(
        "Download complete extraction response",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name=f"{os.path.splitext(uploaded.name)[0]}_extraction.json",
        mime="application/json",
    )


st.markdown(
    '<div class="app-footer">© 2026 Invoice Intelligence · Built with Streamlit and FastAPI · Always verify extracted financial information before final use.</div>',
    unsafe_allow_html=True,
)
