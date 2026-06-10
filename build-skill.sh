#!/bin/bash
# Build distributable skill package for stats-cli-py
# Includes: SKILL.md + all runnable code (no tests, no test data)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="$SCRIPT_DIR/stats-cli-py-skill.zip"

echo "Building stats-cli-py skill package..."

# Remove old zip
rm -f "$OUTPUT"

cd "$SCRIPT_DIR"

# Create zip with runnable code, excluding tests/data/cache
python -c "
import zipfile, os

EXCLUDE_DIRS = {'__pycache__', '.pytest_cache', '.ruff_cache', '.git', 'test-output', '.claude', 'excel', 'tests'}
EXCLUDE_FILES = {'test_real_data.py', 'stats-cli-py-skill.zip'}
EXCLUDE_EXTS = {'.pyc'}

with zipfile.ZipFile('stats-cli-py-skill.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f in EXCLUDE_FILES or any(f.endswith(e) for e in EXCLUDE_EXTS):
                continue
            filepath = os.path.join(root, f)
            arcname = os.path.relpath(filepath, '.').replace(os.sep, '/')
            zf.write(filepath, arcname)

    count = len(zf.infolist())
    total = sum(i.file_size for i in zf.infolist())
    print(f'  {count} files, {total:,} bytes')
"

echo ""
echo "Package created: $OUTPUT"
echo ""
echo "Contents: SKILL.md + main.py + stats_engine/ + utils/ + config files"
echo "Excludes: tests/, excel/, __pycache__/, .git/"
echo ""
echo "To install:"
echo "  1. Unzip to any directory"
echo "  2. pip install -r requirements.txt"
echo "  3. echo '{\"command\":\"descriptive\",\"params\":{\"values\":[1,2,3]}}' | python main.py"
echo ""
echo "Done!"
