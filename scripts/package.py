"""Package stats-cli for distribution as ZIP.

Excludes sensitive/test files, includes only production code and docs.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ZIP = PROJECT_ROOT / "stats-cli-py.zip"

# Files and directories to include
INCLUDE = [
    "stats_engine/",
    "utils/",
    "main.py",
    "__main__.py",
    "SKILL.md",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "VALIDATION.md",
    "pyproject.toml",
    "requirements.txt",
    "examples/",
    "docs/",
]

# Patterns to exclude even if in included dirs
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
    """Create ZIP package."""
    print(f"Packaging from: {PROJECT_ROOT}")
    print(f"Output: {OUTPUT_ZIP}")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for include_item in INCLUDE:
            item_path = PROJECT_ROOT / include_item

            if item_path.is_file():
                if not should_exclude(include_item):
                    zf.write(item_path, include_item)
                    print(f"  + {include_item}")

            elif item_path.is_dir():
                for root, dirs, files in os.walk(item_path):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not should_exclude(d)]

                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(PROJECT_ROOT)
                        if not should_exclude(str(arcname)):
                            zf.write(file_path, arcname)
                            print(f"  + {arcname}")

    # Verify SKILL.md exists in ZIP
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
        names = zf.namelist()
        if "SKILL.md" in names:
            print("\n✓ SKILL.md included")
        else:
            print("\n✗ SKILL.md MISSING!")

        # Check no sensitive files
        sensitive = [n for n in names if any(p in n for p in ["CLAUDE.md", "tests/", ".claude/", "TODO.md", ".env"])]
        if sensitive:
            print(f"✗ Sensitive files found: {sensitive}")
        else:
            print("✓ No sensitive files")

    file_size = OUTPUT_ZIP.stat().st_size
    print(f"\nPackage size: {file_size / 1024:.1f} KB")
    print(f"Files: {len(names)}")


if __name__ == "__main__":
    main()
