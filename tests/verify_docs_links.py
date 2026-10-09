import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
files_to_check = [
    root / "README.md",
    root / "Bussiness_Rules" / "docs" / "PROJECT_CANVAS_ARCHITECTURE.md",
]

link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

all_ok = True
for f in files_to_check:
    print(f"\n--- Checking links in: {f.name} ---")
    content = f.read_text(encoding="utf-8")

    # Check for forbidden file:/// links
    if "file:///" in content:
        print(f"  [ERROR] Found local file:/// link in {f.name}!")
        all_ok = False
    else:
        print("  [OK] No file:/// links found.")

    links = link_pattern.findall(content)
    rel_links = [
        (text, url)
        for text, url in links
        if not url.startswith("http://")
        and not url.startswith("https://")
        and not url.startswith("#")
    ]

    for text, url in rel_links:
        clean_url = url.split("#")[0]
        if not clean_url:
            continue

        target_path = (f.parent / clean_url).resolve()
        if target_path.exists():
            print(f"  [OK] '{text}' -> {clean_url} exists.")
        else:
            print(f"  [FAILED] '{text}' -> {clean_url} NOT FOUND at {target_path}")
            all_ok = False

if all_ok:
    print("\n[SUCCESS] RESULT: ALL RELATIVE LINKS ARE 100% VALID AND EXIST ON DISK!")
else:
    print("\n[FAILED] RESULT: SOME LINKS FAILED VERIFICATION.")
