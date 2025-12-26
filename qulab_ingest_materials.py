#!/usr/bin/env python3
"""
QuLabInfinite Unified Materials Ingester
==========================================

Downloads and indexes materials from all available sources:
- NIST Chemistry WebBook (10K+ substances)
- Materials Project (150K+ structures)
- OQMD (850K+ structures)
- AFLOW (3.5M computed - when available)

Creates indexed SQLite database for instant access.

Usage:
  python qulab_ingest_materials.py --quick      # Fast test (1K materials)
  python qulab_ingest_materials.py --standard   # Normal (150K materials)
  python qulab_ingest_materials.py --full       # Everything available (1M+ materials)
  python qulab_ingest_materials.py --mp-only    # Just Materials Project
  python qulab_ingest_materials.py --oqmd-only  # Just OQMD
  python qulab_ingest_materials.py --resume     # Resume interrupted download
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Ensure we can import the builder
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_api_key():
    """Check if Materials Project API key is set"""
    api_key = os.environ.get("MP_API_KEY")

    logger.info("")
    logger.info("=" * 80)
    logger.info("CHECKING PREREQUISITES")
    logger.info("=" * 80)

    if api_key:
        logger.info("✓ MP_API_KEY found")
        logger.info(f"  Key: {api_key[:10]}...{api_key[-10:]}")
        return True
    else:
        logger.warning("⚠ MP_API_KEY not found")
        logger.warning("")
        logger.warning("Get your free API key from: https://materialsproject.org/api")
        logger.warning("")
        logger.warning("Then set it:")
        logger.warning("  export MP_API_KEY='your_key_here'")
        logger.warning("")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    required = ['requests', 'sqlite3']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info(f"Install with: pip install {' '.join(missing)}")
        return False

    logger.info("✓ All dependencies available")
    return True


def ingest_materials(args):
    """Main ingestion function"""

    # Import the builder
    try:
        from ingest.sources.comprehensive_materials_builder import ComprehensiveBuilder
    except ImportError:
        logger.error("Could not import builder. Make sure you're in QuLabInfinite root.")
        return False

    logger.info("")
    logger.info("=" * 80)
    logger.info("MATERIALS INGESTION CONFIGURATION")
    logger.info("=" * 80)

    # Determine limits based on preset
    presets = {
        'quick': {
            'mp': 100,
            'oqmd': 100,
            'description': 'Quick test (200 materials)',
            'time': '~2 minutes'
        },
        'standard': {
            'mp': 1000,
            'oqmd': 1000,
            'description': 'Standard (2K materials)',
            'time': '~15 minutes'
        },
        'full': {
            'mp': 150000,
            'oqmd': 100000,
            'description': 'Full download (250K materials)',
            'time': '~60 minutes'
        }
    }

    preset = args.preset or 'standard'
    config = presets.get(preset, presets['standard'])

    # Override with individual limits
    if args.mp_only:
        config['oqmd'] = 0
    if args.oqmd_only:
        config['mp'] = 0

    if args.mp_limit:
        config['mp'] = args.mp_limit
    if args.oqmd_limit:
        config['oqmd'] = args.oqmd_limit

    logger.info(f"Preset: {preset.upper()}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Expected time: {config['time']}")
    logger.info("")
    logger.info(f"Settings:")
    logger.info(f"  Materials Project: {config['mp']:,} structures")
    logger.info(f"  OQMD: {config['oqmd']:,} structures")
    logger.info(f"  Output: data/materials_comprehensive.db")
    logger.info("")

    if args.resume:
        logger.warning("⚠ Resume mode not yet fully implemented")
        logger.warning("  Will create new database instead")

    # Confirm
    if not args.yes:
        response = input("Proceed with ingestion? [y/N]: ").strip().lower()
        if response != 'y':
            logger.info("Cancelled.")
            return False

    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING MATERIALS INGESTION")
    logger.info("=" * 80)
    logger.info("")

    # Create builder
    builder = ComprehensiveBuilder()

    # Download from all enabled sources
    try:
        if config['mp'] > 0:
            builder.download_from_mp(limit=config['mp'])

        if config['oqmd'] > 0:
            builder.download_from_oqmd(limit=config['oqmd'])

        # Finalize
        builder.finalize()

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Start the API: python materials_api.py")
        logger.info("  2. Or use launcher: python qulab_launcher.py")
        logger.info("  3. Access materials at: http://localhost:8000/search")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_status():
    """Show current database status"""
    db_path = Path("data/materials_comprehensive.db")

    logger.info("")
    logger.info("=" * 80)
    logger.info("MATERIALS DATABASE STATUS")
    logger.info("=" * 80)
    logger.info("")

    if not db_path.exists():
        logger.warning("⚠ No materials database found")
        logger.info("Run: python qulab_ingest_materials.py --quick")
        return

    import sqlite3

    size_mb = db_path.stat().st_size / (1024**2)
    logger.info(f"✓ Database exists")
    logger.info(f"  Location: {db_path}")
    logger.info(f"  Size: {size_mb:.1f} MB")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM materials")
        count = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(density) FROM materials WHERE density > 0")
        avg_density = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM materials WHERE category != ''")
        categorized = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT category FROM materials WHERE category != ''")
        categories = [row[0] for row in cursor.fetchall()]

        conn.close()

        logger.info(f"  Total materials: {count:,}")
        logger.info(f"  Categorized: {categorized:,}")
        logger.info(f"  Categories: {', '.join(sorted(categories))}")
        logger.info(f"  Avg density: {avg_density:.2f} g/cm³" if avg_density else "  Avg density: N/A")

    except Exception as e:
        logger.warning(f"Could not read database: {e}")

    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="QuLabInfinite Unified Materials Ingester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (200 materials, 2 min)
  python qulab_ingest_materials.py --quick

  # Standard (2K materials, 15 min)
  python qulab_ingest_materials.py --standard

  # Full download (250K+ materials, 60 min)
  python qulab_ingest_materials.py --full

  # Just Materials Project
  python qulab_ingest_materials.py --mp-only

  # Custom limits
  python qulab_ingest_materials.py --mp 50000 --oqmd 50000

  # Check status
  python qulab_ingest_materials.py --status

  # No prompts
  python qulab_ingest_materials.py --quick --yes
        """
    )

    parser.add_argument(
        "--quick",
        action="store_const",
        const="quick",
        dest="preset",
        help="Quick test (200 materials, ~2 minutes)"
    )
    parser.add_argument(
        "--standard",
        action="store_const",
        const="standard",
        dest="preset",
        help="Standard (2K materials, ~15 minutes)"
    )
    parser.add_argument(
        "--full",
        action="store_const",
        const="full",
        dest="preset",
        help="Full download (250K+ materials, ~60 minutes)"
    )

    parser.add_argument(
        "--mp-only",
        action="store_true",
        help="Download only from Materials Project"
    )
    parser.add_argument(
        "--oqmd-only",
        action="store_true",
        help="Download only from OQMD"
    )

    parser.add_argument(
        "--mp",
        type=int,
        dest="mp_limit",
        help="Custom Materials Project limit"
    )
    parser.add_argument(
        "--oqmd",
        type=int,
        dest="oqmd_limit",
        help="Custom OQMD limit"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted download (experimental)"
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current database status"
    )

    args = parser.parse_args()

    # Show status if requested
    if args.status:
        show_status()
        return

    # Check if we're in the right directory
    if not Path("qulab_master_api.py").exists():
        logger.error("Must run from QuLabInfinite root directory")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 80)
    logger.info("QULAB INFINITE - MATERIALS INGESTER")
    logger.info("=" * 80)

    # Check prerequisites
    if not check_dependencies():
        sys.exit(1)

    if not check_api_key():
        logger.warning("Continuing without Materials Project (will skip MP download)")

    # Run ingestion
    success = ingest_materials(args)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
