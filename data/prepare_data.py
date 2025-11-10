import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

"""
Data preparation script for RAG Chunking Challenge.

Converts Tesla 10-K HTML to clean text format for chunking evaluation.

This script:
1. Parses Tesla 10-K HTML using BeautifulSoup.
2. Cleans, normalizes, and writes results to data/processed/.

Usage:
    python data/prepare_data.py
"""


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
    print(" RAG Chunking Challenge - Data Preparation (Tesla 10-K only)")
    print("=" * 70)
    print()

    tasks = [("Tesla 10-K (HTML)", convert_tesla_10k)]

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
