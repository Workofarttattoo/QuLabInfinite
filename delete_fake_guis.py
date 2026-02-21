#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

DELETE ALL FAKE GUI FILES
This script removes all deceptive GUI HTML files and replaces them with honest MCP tool documentation
"""

import os
import re
from pathlib import Path
from typing import List, Dict

def find_fake_guis(base_path: str = "/Users/noone/aios/QuLabInfinite") -> List[Path]:
    """Find all fake GUI HTML files"""
    gui_files = []
    for html_file in Path(base_path).glob("*_gui.html"):
        # Check if file contains the fake sine wave pattern
        with open(html_file, 'r') as f:
            content = f.read()
            if "Math.sin(i * 0.1 + Date.now()" in content:
                gui_files.append(html_file)
    return gui_files

def delete_fake_guis(dry_run: bool = True) -> int:
    """Delete all fake GUI files"""
    fake_guis = find_fake_guis()

    print(f"\nFound {len(fake_guis)} fake GUI files to delete:")
    print("-" * 60)

    for gui_file in fake_guis:
        print(f"  - {gui_file.name}")

    if dry_run:
        print(f"\nDRY RUN MODE - No files deleted")
        print(f"Run with --delete flag to actually remove files")
    else:
        print(f"\nDELETING {len(fake_guis)} fake GUI files...")
        for gui_file in fake_guis:
            gui_file.unlink()
            print(f"  ✓ Deleted {gui_file.name}")
        print(f"\nSuccessfully deleted {len(fake_guis)} fake GUI files")

    return len(fake_guis)

def main():
    """Main execution"""
    import sys

    dry_run = "--delete" not in sys.argv

    print("=" * 60)
    print("FAKE GUI DELETION TOOL")
    print("=" * 60)

    count = delete_fake_guis(dry_run=dry_run)

    if count > 0 and not dry_run:
        print("\nAll fake GUIs have been removed.")
        print("Next step: Run create_mcp_docs.py to generate honest documentation")

if __name__ == "__main__":
    main()