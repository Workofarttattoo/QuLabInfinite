#!/bin/bash
# Sync all labs to iCloud Drive

ICLOUD_DIR="$HOME/Library/Mobile Documents/com~apple~CloudDocs/QuLabInfinite_Labs"

# Create iCloud directory if it doesn't exist
mkdir -p "$ICLOUD_DIR"

# Copy all lab files
rsync -av --include='*_lab.py' --include='lab_build*.json' --exclude='*' /Users/noone/QuLabInfinite/ "$ICLOUD_DIR/"

echo "✅ Synced labs to iCloud Drive: $ICLOUD_DIR"
ls -lh "$ICLOUD_DIR" | wc -l
