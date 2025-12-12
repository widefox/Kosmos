#!/bin/bash
# Install kosmos-xray skill and kosmos_architect agent to target repository
#
# Usage:
#   ./install.sh /path/to/target/repo
#   ./install.sh .  # Install to current directory

set -e

TARGET_DIR="${1:-.}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory '$TARGET_DIR' does not exist"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing kosmos-xray to: $TARGET_DIR"

# Create .claude directories if needed
mkdir -p "$TARGET_DIR/.claude/skills"
mkdir -p "$TARGET_DIR/.claude/agents"

# Copy skill
echo "  Copying kosmos-xray skill..."
cp -r "$SCRIPT_DIR/kosmos-xray" "$TARGET_DIR/.claude/skills/"

# Copy agent
echo "  Copying kosmos_architect agent..."
cp "$SCRIPT_DIR/agents/kosmos_architect.md" "$TARGET_DIR/.claude/agents/"

# Copy example WARM_START.md (optional)
if [ ! -f "$TARGET_DIR/WARM_START.md" ]; then
    echo "  Copying example WARM_START.md..."
    cp "$SCRIPT_DIR/WARM_START.md" "$TARGET_DIR/WARM_START.example.md"
fi

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .claude/skills/kosmos-xray/configs/ignore_patterns.json"
echo "  2. Edit .claude/skills/kosmos-xray/configs/priority_modules.json"
echo "  3. Test: python .claude/skills/kosmos-xray/scripts/mapper.py --summary"
echo "  4. Generate docs: @kosmos_architect generate"
