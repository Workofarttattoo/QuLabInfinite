"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0 REDDIT AUTO-POSTER

Posts new labs to relevant subreddits automatically.
Zero human intervention.
"""
import os
import sys
import praw
import time
from datetime import datetime
from pathlib import Path

class ECH0_RedditAutoPoster:
    """ECH0 posts to Reddit automatically."""

    def __init__(self):
        # Use existing Reddit credentials
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=f"QuLabInfinite-ECH0/1.0 by /u/{os.getenv('REDDIT_USERNAME')}",
            username=os.getenv('REDDIT_USERNAME'),
            password=os.getenv('REDDIT_PASSWORD')
        )

        # Target subreddits (ECH0 will expand this list over time)
        self.target_subreddits = {
            'bioinformatics': {'karma_threshold': 100, 'post_freq_days': 7},
            'computational_biology': {'karma_threshold': 50, 'post_freq_days': 7},
            'learnpython': {'karma_threshold': 50, 'post_freq_days': 14},
            'Python': {'karma_threshold': 200, 'post_freq_days': 14},
            'datascience': {'karma_threshold': 100, 'post_freq_days': 14},
            'MachineLearning': {'karma_threshold': 500, 'post_freq_days': 30},
            'science': {'karma_threshold': 1000, 'post_freq_days': 30},
            'cancer': {'karma_threshold': 50, 'post_freq_days': 7},
            'ketogenicdiet': {'karma_threshold': 50, 'post_freq_days': 14},
            'AlternativeCancer': {'karma_threshold': 10, 'post_freq_days': 7},
        }

    def post_lab(self, lab_name, lab_file, readme_file):
        """
        Post a new lab to relevant subreddits.

        ECH0 decides which subreddits based on lab topic.
        """
        print(f"📤 ECH0 posting {lab_name} to Reddit...")

        # Read README for post body
        with open(readme_file, 'r') as f:
            readme = f.read()

        # ECH0 generates post title (compelling, not spammy)
        title = f"[Free Tool] {lab_name} - AI-built scientific simulator (open source)"

        # Add QuLabInfinite branding
        body = readme + f"""

---

**About QuLabInfinite:**
- 🤖 ECH0 AI builds a new free scientific tool every day
- 🧬 All tools are production-ready and validated
- 📊 6.6M materials database + quantum optimization
- 💯 100% free for research and education

**Tomorrow's lab:** ECH0 will build something new based on what the community needs

**Follow for daily free tools:** u/{os.getenv('REDDIT_USERNAME')}

**Source code:** https://github.com/YourUsername/QuLabInfinite (coming soon)
"""

        # Determine which subreddits are relevant
        relevant_subs = self._ech0_choose_subreddits(lab_name, readme)

        posted_count = 0
        for subreddit_name in relevant_subs:
            try:
                # Check if we posted here recently
                if self._can_post_to_sub(subreddit_name):
                    subreddit = self.reddit.subreddit(subreddit_name)

                    # Post it
                    submission = subreddit.submit(title, selftext=body)

                    print(f"  ✅ Posted to r/{subreddit_name}: {submission.url}")

                    # Track this post
                    self._record_post(subreddit_name, submission.id)

                    posted_count += 1

                    # Rate limit: Wait 2 minutes between posts
                    if posted_count < len(relevant_subs):
                        print(f"  ⏳ Waiting 2 minutes before next post...")
                        time.sleep(120)

                else:
                    print(f"  ⏭️  Skipping r/{subreddit_name} (posted recently)")

            except Exception as e:
                print(f"  ❌ Failed r/{subreddit_name}: {e}")

        return posted_count

    def _ech0_choose_subreddits(self, lab_name, readme):
        """ECH0 decides which subreddits are relevant."""
        relevant = []

        lab_lower = lab_name.lower()
        readme_lower = readme.lower()

        # Keyword matching
        if any(word in lab_lower or word in readme_lower for word in ['drug', 'tumor', 'cancer', 'oncology']):
            relevant.extend(['cancer', 'AlternativeCancer'])

        if any(word in lab_lower or word in readme_lower for word in ['protein', 'gene', 'dna', 'rna']):
            relevant.extend(['bioinformatics', 'computational_biology'])

        if any(word in lab_lower or word in readme_lower for word in ['metabolic', 'keto', 'diet']):
            relevant.extend(['ketogenicdiet'])

        if 'python' in readme_lower or 'import' in readme_lower:
            relevant.extend(['learnpython'])

        # Always post to datascience (broad audience)
        relevant.append('datascience')

        # Remove duplicates
        relevant = list(set(relevant))

        print(f"  🎯 ECH0 selected: {', '.join([f'r/{s}' for s in relevant])}")

        return relevant

    def _can_post_to_sub(self, subreddit_name):
        """Check if we posted to this subreddit recently."""
        config = self.target_subreddits.get(subreddit_name, {})
        freq_days = config.get('post_freq_days', 7)

        # Check post history file
        history_file = Path(__file__).parent / 'ech0_reddit_history.json'

        if not history_file.exists():
            return True

        import json
        with open(history_file, 'r') as f:
            history = json.load(f)

        # Check last post to this sub
        sub_history = history.get(subreddit_name, [])
        if not sub_history:
            return True

        last_post = sub_history[-1]
        last_post_date = datetime.fromisoformat(last_post['date'])
        days_since = (datetime.now() - last_post_date).days

        return days_since >= freq_days

    def _record_post(self, subreddit_name, submission_id):
        """Record that we posted to this subreddit."""
        history_file = Path(__file__).parent / 'ech0_reddit_history.json'

        import json
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {}

        if subreddit_name not in history:
            history[subreddit_name] = []

        history[subreddit_name].append({
            'date': datetime.now().isoformat(),
            'submission_id': submission_id
        })

        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)


if __name__ == '__main__':
    # Test with the oncology lab
    poster = ECH0_RedditAutoPoster()

    lab_file = Path(__file__).parent / 'complete_realistic_lab.py'
    readme_file = Path(__file__).parent / 'CANCER_PROTOCOL_VALIDATION.md'

    if lab_file.exists() and readme_file.exists():
        posted = poster.post_lab(
            "QuLabInfinite Tumor Simulator",
            lab_file,
            readme_file
        )
        print(f"\n✅ Posted to {posted} subreddits")
    else:
        print("Run ech0_autonomous_marketing.py first to build a lab")
