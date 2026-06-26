"""Package stats-cli for Feishu Aily distribution.

Uses SKILL_AILY.md (shortened description <700 chars) instead of SKILL.md.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ZIP = PROJECT_ROOT / "stats-cli-py-aily.zip"

# Files and directories to include
INCLUDE = [
    "stats_engine/",
    "utils/",
    "main.py",
    "__main__.py",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "VALIDATION.md",
    "pyproject.toml",
    "requirements.txt",
    "examples/",
    "docs/",
]

# Patterns to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    ".claude",
    "tests",
    "test-output",
    "excel",
    "CLAUDE.md",
    "TODO.md",
    ".env",
    "*.egg-info",
    "dist/",
    "build/",
    ".github/",
    "scripts/",
    "ARCHIVE.md",
]


def should_exclude(path: str) -> bool:
    """Check if path should be excluded."""
    return any(pattern in path for pattern in EXCLUDE_PATTERNS)


def main():
    """Create Aily-compatible ZIP package."""
    print(f"Packaging Aily version from: {PROJECT_ROOT}")
    print(f"Output: {OUTPUT_ZIP}")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add SKILL_AILY.md as SKILL.md
        skill_aily = PROJECT_ROOT / "SKILL_AILY.md"
        zf.write(skill_aily, "SKILL.md")
        print("  + SKILL.md (from SKILL_AILY.md)")

        for include_item in INCLUDE:
            item_path = PROJECT_ROOT / include_item

            if item_path.is_file():
                if not should_exclude(include_item):
                    zf.write(item_path, include_item)
                    print(f"  + {include_item}")

            elif item_path.is_dir():
                for root, dirs, files in os.walk(item_path):
                    dirs[:] = [d for d in dirs if not should_exclude(d)]

                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(PROJECT_ROOT)
                        if not should_exclude(str(arcname)):
                            zf.write(file_path, arcname)
                            print(f"  + {arcname}")

    # Verify
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
        names = zf.namelist()
        with zf.open("SKILL.md") as f:
            content = f.read().decode("utf-8")
            # Extract description
            for line in content.split("\n"):
                if line.startswith("description:"):
                    desc = line.split('"')[1] if '"' in line else line.split(": ", 1)[1]
                    print(f"\nDescription length: {len(desc)} chars")
                    if len(desc) <= 700:
                        print("✓ Within Aily 700-char limit")
                    else:
                        print("✗ EXCEEDS 700-char limit!")
                    break

        sensitive = [n for n in names if any(p in n for p in ["tests/", "CLAUDE.md", ".claude/", "TODO.md"])]
        if sensitive:
            print(f"✗ Sensitive files: {sensitive}")
        else:
            print("✓ No sensitive files")

    file_size = OUTPUT_ZIP.stat().st_size
    print(f"\nPackage size: {file_size / 1024:.1f} KB")
    print(f"Files: {len(names)}")


if __name__ == "__main__":
    main()
