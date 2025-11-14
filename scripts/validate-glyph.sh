#!/bin/bash
# GΛLYPH validation script
set -e
echo "🔍 Validating GΛLYPH files..."
GLYPH_FILES=$(find . -name "*.glyph" -type f)
if [ -z "$GLYPH_FILES" ]; then
    echo "ℹ️  No .glyph files found"
    exit 0
fi
for glyph_file in $GLYPH_FILES; do
    echo "📝 Validating $glyph_file"
    if ! cargo run --bin glyph_parser --validate "$glyph_file"; then
        echo "❌ GΛLYPH validation failed for $glyph_file"
        exit 1
    fi
    echo "✅ $glyph_file validated successfully"
done
echo "🎉 All GΛLYPH files validated successfully!"
