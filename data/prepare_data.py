import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from bs4 import BeautifulSoup

"""
Data preparation script for RAG Chunking Challenge.

Converts raw PDF and HTML files to clean text format for chunking evaluation.

This script:
1. Extracts text from ArXiv PDF using PyMuPDF (fitz) for better structure
   and pdfplumber for table extraction.
2. Parses Tesla 10-K HTML using BeautifulSoup.
3. Cleans, normalizes, and writes results to data/processed/.

Usage:
    python data/prepare_data.py
"""


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def normalize_pdf_text(text: str) -> str:
    """Clean ligatures, encoding artifacts, and spacing issues."""
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "￿": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Normalize Unicode & whitespace
    text = re.sub(r"\s+", " ", text)
    # Merge hyphenated line breaks (e.g., "in-\ntelligence" → "intelligence")
    text = re.sub(r"(?<=\w)- (?=\w)", "", text)
    # Remove citation brackets like [12]
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


def format_table_as_text(table):
    """Convert extracted table to simple text format."""
    if not table:
        return ""
    lines = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
        lines.append(" | ".join(cleaned_row))
    return "\n".join(lines)


def postprocess_text(full_text: str) -> str:
    """
    Optional final cleanup before chunking.

    Removes duplicates, stray control characters, and fixes newlines.
    """
    # Remove stray non-ASCII characters
    full_text = re.sub(r"[^\x00-\x7F]+", " ", full_text)
    # Collapse multiple spaces/newlines
    full_text = re.sub(r"\s{2,}", " ", full_text)
    # Trim extra whitespace after newlines
    full_text = re.sub(r"(?<=\n)\s+", "", full_text)
    # Remove any accidental duplicated "Page X" markers
    full_text = re.sub(r"(--- Page \d+ ---\s*){2,}", r"\1", full_text)
    return full_text.strip()


# -----------------------------------------------------------------------------
# PDF (ArXiv) conversion
# -----------------------------------------------------------------------------
def convert_arxiv_pdf():
    """
    Convert ArXiv PDF to clean text.

    Uses PyMuPDF for high-quality text extraction and pdfplumber
    for table extraction. Saves results to data/processed/.
    """
    print("Converting ArXiv paper (via PyMuPDF + pdfplumber)...")

    script_dir = Path(__file__).parent
    pdf_path = script_dir / "raw" / "arxiv_paper.pdf"
    output_path = script_dir / "processed" / "arxiv_paper.txt"

    if not pdf_path.exists():
        print(f"  ❌ {pdf_path} not found")
        return False

    try:
        text_parts = []

        # ---------- Main text extraction ----------
        doc = fitz.open(pdf_path)
        print(f"  📄 Processing {len(doc)} pages...")

        for page_num, page in enumerate(doc, 1):
            # Extract blocks of text (layout-aware)
            blocks = page.get_text("blocks")
            blocks = sorted(
                blocks, key=lambda b: (b[1], b[0])
            )  # sort top-down, left-right

            page_text = "\n".join(block[4] for block in blocks if block[4].strip())
            page_text = normalize_pdf_text(page_text)

            text_parts.append(f"\n--- Page {page_num} ---\n")
            text_parts.append(page_text)

        doc.close()

        # ---------- Table extraction ----------
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if tables:
                    print(f"  📊 Found {len(tables)} table(s) on page {page_num}")
                    for i, table in enumerate(tables, 1):
                        text_parts.append(f"\n[TABLE {page_num}.{i}]\n")
                        text_parts.append(format_table_as_text(table))
                        text_parts.append("\n[END TABLE]\n")

        # ---------- Write output ----------
        full_text = "\n".join(text_parts)
        # full_text = postprocess_text(full_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_text, encoding="utf-8")

        file_size_kb = len(full_text) / 1024
        print(f"  ✓ Extracted {file_size_kb:.1f} KB of text")
        print(f"  ✓ Saved to {output_path.relative_to(script_dir.parent)}")
        return True

    except Exception as e:
        print(f"  ❌ Error processing PDF: {e}")
        import traceback

        traceback.print_exc()
        return False


# -----------------------------------------------------------------------------
# HTML (Tesla 10-K) conversion
# -----------------------------------------------------------------------------
def extract_table_from_html(table_element):
    """Extract table data from HTML <table>."""
    rows = []
    for row in table_element.find_all("tr"):
        cells = []
        for cell in row.find_all(["th", "td"]):
            text = " ".join(cell.get_text(strip=True).split())
            cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows) if rows else ""


def convert_tesla_10k():
    """Convert Tesla 10-K HTML to clean text using BeautifulSoup."""
    print("Converting Tesla 10-K (HTML)...")

    script_dir = Path(__file__).parent
    html_path = script_dir / "raw" / "tesla_10K.html"
    output_path = script_dir / "processed" / "tesla_10k_excerpt.txt"

    if not html_path.exists():
        print(f"  ❌ {html_path} not found")
        return False

    try:
        html_content = html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "lxml")

        # Remove scripts, styles, and non-content elements
        for tag in soup(
            ["script", "style", "meta", "link", "noscript", "header", "footer", "nav"]
        ):
            tag.decompose()

        text_parts = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "table", "div"]):
            if element.name in ["h1", "h2", "h3", "h4"]:
                heading = element.get_text(strip=True)
                if heading:
                    text_parts.append(f"\n\n[{element.name.upper()}] {heading}\n")
            elif element.name == "table":
                table_text = extract_table_from_html(element)
                if table_text:
                    text_parts.append("\n[TABLE START]\n")
                    text_parts.append(table_text)
                    text_parts.append("\n[TABLE END]\n")
            else:
                text = element.get_text(strip=True)
                if text and len(text) > 20:
                    text_parts.append(text)

        # Fallback if minimal content extracted
        if len(text_parts) < 10:
            print(
                "  ⚠️ Structured extraction yielded little content; using fallback mode..."
            )
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines()]
            full_text = "\n".join(line for line in lines if line)
        else:
            full_text = "\n".join(text_parts)

        # Clean up excessive newlines
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_text, encoding="utf-8")

        size_kb = len(full_text) / 1024
        num_tables = full_text.count("[TABLE START]")
        print(f"  ✓ Extracted text ({size_kb:.1f} KB, {num_tables} tables)")
        print(f"  ✓ Saved to {output_path.relative_to(script_dir.parent)}")
        return True

    except Exception as e:
        print(f"  ❌ Error processing HTML: {e}")
        import traceback

        traceback.print_exc()
        return False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" RAG Chunking Challenge - Data Preparation")
    print("=" * 70)
    print()

    tasks = [
        ("ArXiv Paper (PDF)", convert_arxiv_pdf),
        ("Tesla 10-K (HTML)", convert_tesla_10k),
    ]

    results = []
    for name, func in tasks:
        success = func()
        results.append((name, success))
        print()

    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    success_count = sum(1 for _, s in results if s)
    print(f"\nCompleted: {success_count}/{len(results)} documents processed\n")

    if success_count == len(results):
        print("✅ All documents ready for chunking!")
        print("\nNext steps:")
        print("  1. Review processed files in data/processed/")
        print("  2. Implement chunking strategies in src/chunkers/")
        print("  3. Create evaluation queries in evaluation/queries.json")
        return 0
    else:
        print("⚠️ Some documents failed to process.")
        print("   Check error messages above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
