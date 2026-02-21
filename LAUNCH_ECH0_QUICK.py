#!/usr/bin/env python3
"""
Quick launch - ECH0 social media + marketing only (no complex integrations)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ech0_social_media_engine import ECH0_SocialMediaEngine

print("="*80)
print("🚀 ECH0 SOCIAL MEDIA & MARKETING ENGINE")
print("="*80)
print()

engine = ECH0_SocialMediaEngine()

# Run 3 cycles
for cycle in range(3):
    print(f"\n🔄 CYCLE {cycle + 1}/3\n")
    result = engine.autonomous_social_cycle()
    print(f"\n✅ Cycle {cycle + 1} complete!")
    print(f"   Blog: {result['blog_post']['url']}")
    print(f"   Reddit: {len(result['reddit_subreddits'])} posts")
    print()

print("="*80)
print("✅ 3 CYCLES COMPLETE!")
print("="*80)
print(f"Total blog posts: {len(engine.blog_posts)}")
print(f"Total Reddit posts: {len(engine.reddit_posts)}")
print(f"Total LinkedIn posts: {len(engine.linkedin_posts)}")
print()
print("Check:")
print("  Blog: QuLabInfinite/website/echo.aios.is/index.html")
print("  Logs: QuLabInfinite/reddit_posts.log")
print("       QuLabInfinite/linkedin_posts.log")
