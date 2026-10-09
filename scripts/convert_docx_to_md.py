import os
import sys
import re
import argparse
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ingestion.docx_parser import docx_parser


def docx_to_markdown(docx_path: str) -> str:
    """
    Converts a docx file to Markdown by leveraging docx_parser's rich extraction.
    """
    chunks = docx_parser.parse_docx(docx_path)
    parent_chunks = [c for c in chunks if not c.metadata.is_child]

    md_sections = []
    for chunk in parent_chunks:
        # Convert '### Title' header to top-level '# Title'
        sec_text = re.sub(r"^###\s+", "# ", chunk.text_content)
        md_sections.append(sec_text.strip())

    return "\n\n".join(md_sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Convert DOCX to Markdown mirroring Bussiness_Rules/docs/markdown format")
    parser.add_argument("--input", "-i", type=str, help="Path to input .docx file")
    parser.add_argument("--output", "-o", type=str, help="Path to output .md file")
    parser.add_argument("--test", action="store_true", help="Run conversion self-check")
    args = parser.parse_args()

    project_root = backend_dir.parent

    if args.test:
        test_file = project_root / "data" / "raw_docs" / "_AI_HDSD_ATLĐ (DN).v1_HCM_2026 (1).docx"
        if not test_file.exists():
            print(f"[FAIL] Test file not found: {test_file}")
            sys.exit(1)
        res = docx_to_markdown(str(test_file))
        assert "# ĐĂNG KÝ" in res, "Missing '# ĐĂNG KÝ' section"
        assert "[IMAGE_1]" in res, "Missing image placeholder"
        print(f"[PASS] Self-test passed. Generated {len(res.splitlines())} lines of Markdown.")
        return

    default_pairs = [
        (
            project_root / "data" / "raw_docs" / "_AI_HDSD_ATLĐ (DN).v1_HCM_2026 (1).docx",
            project_root / "Bussiness_Rules" / "docs" / "markdown" / "HDSD_DN_v1_2026.md"
        ),
        (
            project_root / "data" / "raw_docs" / "HDSD ATLĐ (Phường).docx",
            project_root / "Bussiness_Rules" / "docs" / "markdown" / "HDSD_Phuong_v2_2026.md"
        )
    ]

    targets = [(Path(args.input), Path(args.output))] if (args.input and args.output) else default_pairs

    for in_path, out_path in targets:
        if not in_path.exists():
            print(f"[SKIP] Input not found: {in_path}")
            continue
        print(f"[CONVERTING] {in_path.name} -> {out_path.name}...")
        md_content = docx_to_markdown(str(in_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_content, encoding="utf-8")
        print(f"[DONE] Saved {len(md_content.splitlines())} lines to {out_path}")


if __name__ == "__main__":
    main()
