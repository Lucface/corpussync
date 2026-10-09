"""source: extract reads notes and captions, and skips pdf or docx when the extra is missing."""

import builtins
import sys
from pathlib import Path

from corpussync.extract import extract_file
from corpussync.vtt import clean_srt, clean_vtt

FIXTURES = Path(__file__).resolve().parent / "fixtures"

_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 73 >>
stream
BT
/F1 12 Tf
72 720 Td
(Hello from the pdf fixture.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000390 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
467
%%EOF
"""


def test_markdown_txt_html_vtt_and_srt_fixtures(tmp_path):
    """source: md, txt, html, vtt, and srt extract to text, with a heading title when one exists."""
    md_text, md_title = extract_file(FIXTURES / "sample.md")
    assert md_title == "Markdown Title"
    assert "linen cloth" in md_text

    copy = tmp_path / "sample.markdown"
    copy.write_text((FIXTURES / "sample.md").read_text(encoding="utf-8"), encoding="utf-8")
    _text, title = extract_file(copy)
    assert title == "Markdown Title"

    txt, txt_title = extract_file(FIXTURES / "sample.txt")
    assert txt_title is None
    assert "Plain text notes" in txt

    html, html_title = extract_file(FIXTURES / "sample.html")
    assert html_title == "HTML Title"
    assert "Hello from html." in html
    assert "secret script" not in html
    assert "Page Title" not in html

    vtt, _title = extract_file(FIXTURES / "rolling.vtt")
    assert vtt == clean_vtt((FIXTURES / "rolling.vtt").read_text(encoding="utf-8"))
    srt, _title = extract_file(FIXTURES / "sample.srt")
    assert srt == clean_srt((FIXTURES / "sample.srt").read_text(encoding="utf-8"))


def test_pdf_and_docx_extract_when_installed(tmp_path):
    """source: pdf and docx are extracted when the extra is installed."""
    pdf_path = tmp_path / "note.pdf"
    pdf_path.write_bytes(_PDF)
    pdf_text, pdf_title = extract_file(pdf_path)
    assert pdf_title is None
    assert "Hello from the pdf fixture." in pdf_text

    from docx import Document

    docx_path = tmp_path / "note.docx"
    document = Document()
    document.add_heading("Doc Title", level=1)
    document.add_paragraph("Hello from docx body")
    document.save(docx_path)
    docx_text, docx_title = extract_file(docx_path)
    assert docx_title == "Doc Title"
    assert "Hello from docx body" in docx_text


def _block_import(monkeypatch, root_name):
    real = builtins.__import__
    for key in list(sys.modules):
        if key == root_name or key.startswith(root_name + "."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == root_name or name.startswith(root_name + "."):
            raise ImportError(f"{root_name} is not installed")
        return real(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_pdf_skipped_when_extra_missing(tmp_path, monkeypatch, capsys):
    """source: a missing pdf extra skips the file with one line naming the extra."""
    _block_import(monkeypatch, "pypdf")
    path = tmp_path / "note.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    assert extract_file(path) is None
    captured = capsys.readouterr()
    assert "[pdf]" in captured.out
    assert "Traceback" not in captured.err


def test_docx_skipped_when_extra_missing(tmp_path, monkeypatch, capsys):
    """source: a missing docx extra skips the file with one line naming the extra."""
    _block_import(monkeypatch, "docx")
    path = tmp_path / "note.docx"
    path.write_bytes(b"not a document")
    assert extract_file(path) is None
    captured = capsys.readouterr()
    assert "[docx]" in captured.out
    assert "Traceback" not in captured.err
