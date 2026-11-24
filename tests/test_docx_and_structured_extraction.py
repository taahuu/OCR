from io import BytesIO

from docx import Document

from app.services.docx_service import DocxService
from app.services.information_extraction_service import InformationExtractionService
from app.utils.file_validation import DocumentFormat, detect_format, validate_upload


def make_docx() -> bytes:
    document = Document()
    document.add_paragraph("Akanksha Marathe")
    document.add_paragraph("Email: akanksha@example.com")
    document.add_paragraph("PROFESSIONAL SUMMARY")
    document.add_paragraph("Python backend developer")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Skill"
    table.cell(0, 1).text = "Level"
    table.cell(1, 0).text = "Python"
    table.cell(1, 1).text = "Advanced"
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_docx_signature_and_validation():
    data = make_docx()
    assert detect_format(data) is DocumentFormat.DOCX
    upload = validate_upload("resume.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", data)
    assert upload.format is DocumentFormat.DOCX


def test_docx_extracts_paragraphs_and_tables():
    text, tables = DocxService().extract(make_docx())
    assert "Akanksha Marathe" in text
    assert tables == [{"table_number": 1, "headers": ["Skill", "Level"], "rows": [["Python", "Advanced"]]}]


def test_resume_classification_and_basic_fields():
    text = """Akanksha Marathe
Email: akanksha@example.com
PROFESSIONAL SUMMARY
Python Backend Developer
TECHNICAL SKILLS
Python, FastAPI
WORK EXPERIENCE
Backend Developer at Example
EDUCATION
B.Tech"""
    result = InformationExtractionService().extract(text)
    assert result["document_type"] == "resume"
    assert result["resume"]["name"] == "Akanksha Marathe"
    assert result["resume"]["email"] == "akanksha@example.com"
    assert "technical_skills" in result["resume"]["sections"]


def test_invoice_fields_are_extracted_without_inventing_missing_values():
    result = InformationExtractionService().extract("INVOICE\nInvoice No: INV-1024\nInvoice Date: 24-11-2026\nTotal: INR 5,000")
    assert result["document_type"] == "invoice"
    assert result["invoice"]["invoice_number"] == "INV-1024"
    assert result["invoice"]["invoice_date"] == "24-11-2026"
    assert "due_date" not in result["invoice"]  # not part of the selected invoice fields


def test_bank_statement_transactions_are_mapped_from_detected_tables():
    text = "BANK STATEMENT\nAccount Number: 1234567890"
    tables = [{"headers": ["Date", "Description", "Debit", "Credit", "Balance"], "rows": [["01-01-2026", "ATM", "500", "", "9500"]]}]
    result = InformationExtractionService().extract(text, tables)
    assert result["document_type"] == "bank_statement"
    assert result["bank_statement"]["transactions"][0]["description"] == "ATM"


def test_shades_tax_invoice_extracts_selected_fields_and_all_line_items():
    text = """Tax Invoice
Shades                                                   Invoice No.                Dated
Green House Nursery Premises,                   2026-27/678            12-Nov-26
Behind Pushpkunj Hospital, Khandwa Road            Delivery Note             Mode/Terms of Payment
Indore, Madhya Pradesh                         sp:ku
GSTIN/UIN: 23DKJPM8261F1Z1
State Name : Madhya Pradesh, Code : 23
Contact : 9039462121
E-Mail : shadestheluxurygardenstore@gmail.com
Consignee (Ship to)
                                                       Dispatch Doc No.            Delivery Note DateRudraksha Construction
GR Garden Services, In Front of Shehnai Garden,    Ch:S1/30               12-Nov-26
 1 Tokyo 40 WS                    3923       18 %    2 nos   1,923.73 nos  45 %       2,116.10
 2 Tokyo 40 CS                    3923       18 %    2 nos   1,923.73 nos  45 %       2,116.10
 3 Rome Big 35 Eco Cream         3923       18 %    1 nos     977.96 nos  45 %        537.88
 4 Havana 30 White                3923       18 %    6 nos     737.20 nos  45 %      2,432.76
 5 Havana 30 Grey                 3923       18 %    6 nos     737.20 nos  45 %      2,432.76
 6 York 11 Mocca                  3923       18 %    1 nos      42.37 nos  45 %         23.30
 7 York 16 Mocca                  3923       18 %    1 nos      97.46 nos  45 %         53.60
 8 York 16 Eco Brown              3923       18 %    1 nos     110.20 nos  45 %         60.61
 9 York Eco 35 Brown              3923       18 %    1 nos     890.00 nos  45 %        489.50
10 York 35 Mocca                  3923       18 %    1 nos     762.71 nos  45 %        419.49
10,682.10
CGST Output 961.39
SGST Output 961.39
Rounded Off 0.12
Total 22 nos ₹ 12,605.00
Amount Chargeable (in words) E. & O.E
INR Twelve Thousand Six Hundred Five Only
3923 10,682.10 9% 961.39 9% 961.39 1,922.78
A/c Holder's Name : Shades
Bank Name : ICICI Bank
A/c No. : 094505002801
Branch & IFS Code: Ratlam Kothi & ICIC0000916
"""
    invoice = InformationExtractionService().extract(text)["invoice"]
    assert invoice["invoice_number"] == "2026-27/678"
    assert invoice["invoice_date"] == "12-Nov-26"
    assert invoice["delivery_note"] == "sp:ku"
    assert invoice["dispatch_document_number"] == "Ch:S1/30"
    assert "delivery_note_date" not in invoice
    assert invoice["seller"]["name"] == "Shades"
    assert invoice["seller"]["gstin"] == "23DKJPM8261F1Z1"
    assert invoice["seller"]["email"] == "shadestheluxurygardenstore@gmail.com"
    assert len(invoice["product_items"]) == 10
    assert invoice["product_items"][0]["description"] == "Tokyo 40 WS"
    assert invoice["tax_summary"]["taxable_value"] == "10,682.10"
    assert invoice["tax_summary"]["cgst"] == "961.39"
    assert invoice["tax_summary"]["sgst"] == "961.39"
    assert invoice["tax_summary"]["total_tax_amount"] == "1,922.78"
    assert invoice["grand_total"] == "12,605.00"
    assert invoice["bank_details"]["ifsc"] == "ICIC0000916"
