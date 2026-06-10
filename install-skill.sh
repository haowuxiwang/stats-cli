#!/bin/bash
set -e
SKILL_DIR="$HOME/.claude/skills/data-analysis/stats-cli-py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing stats-cli-py SKILL.md..."
mkdir -p "$SKILL_DIR"
cp "$SCRIPT_DIR/SKILL.md" "$SKILL_DIR/"
echo "SKILL.md installed to: $SKILL_DIR"
echo "Done!"
