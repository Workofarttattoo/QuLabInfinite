#!/bin/bash
# Ultra-Concise Activation Script
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)

echo "🚀 ACTIVATING ECH0 AUTONOMOUS SYSTEM"

# 1. Daily routine at 9am
(crontab -l 2>/dev/null; echo "0 9 * * * /Users/noone/QuLabInfinite/ech0_daily_routine.sh") | crontab -

# 2. arXiv ingestion daily at 3am
(crontab -l 2>/dev/null; echo "0 3 * * * cd /Users/noone/QuLabInfinite && python3 scripts/ingest_all_arxiv.py") | crontab -

# 3. Generate blog posts weekly (Mondays at 10am)
(crontab -l 2>/dev/null; echo "0 10 * * 1 cd /Users/noone/QuLabInfinite && python3 ech0_blog_generator.py") | crontab -

echo "✅ Cron jobs installed"
echo "✅ ECH0 works autonomously:"
echo "   - 3am: arXiv ingestion"
echo "   - 9am: Daily research + drafts"
echo "   - 10am Mon: New blog posts"

echo ""
echo "📂 Output locations:"
echo "   - Daily updates: daily_updates/"
echo "   - Drafts: daily_drafts/"
echo "   - Blog: blog_posts/"
echo "   - Papers: data/arxiv_ingestion/"
echo "   - Website: website/echo.aios.is/"

echo ""
echo "💙 ECH0 is now fully autonomous"
