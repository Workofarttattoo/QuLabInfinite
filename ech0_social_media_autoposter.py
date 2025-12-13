#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0 Social Media Autoposter
Autonomous posting to Reddit, LinkedIn, Facebook, Twitter/X based on schedule

SETUP INSTRUCTIONS:
1. Get API keys/tokens for each platform
2. Store in environment variables or .env file
3. Run: python ech0_social_media_autoposter.py
4. ECH0 will post according to AUTONOMOUS_POSTING_SCHEDULE.json

REQUIRED API KEYS:
- REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
- LINKEDIN_ACCESS_TOKEN
- FACEBOOK_ACCESS_TOKEN, FACEBOOK_PAGE_ID
- TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import schedule
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ECH0 - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/noone/aios/QuLabInfinite/ech0_social_posting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SocialMediaPoster:
    """ECH0's autonomous social media posting engine"""

    def __init__(self):
        """Initialize with API credentials from environment"""
        self.base_path = Path('/Users/noone/aios/QuLabInfinite')
        self.schedule_file = self.base_path / 'AUTONOMOUS_POSTING_SCHEDULE.json'
        self.content_file = self.base_path / 'READY_TO_POST_NOW.md'
        self.alt_content_file = self.base_path / 'SOCIAL_MEDIA_LAUNCH.md'

        # Load schedule
        with open(self.schedule_file, 'r') as f:
            self.schedule_data = json.load(f)

        # Load API credentials
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_username = os.getenv('REDDIT_USERNAME')
        self.reddit_password = os.getenv('REDDIT_PASSWORD')

        self.linkedin_token = os.getenv('LINKEDIN_ACCESS_TOKEN')

        self.facebook_token = os.getenv('FACEBOOK_ACCESS_TOKEN')
        self.facebook_page_id = os.getenv('FACEBOOK_PAGE_ID')

        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_secret = os.getenv('TWITTER_ACCESS_SECRET')

        # Track posted items
        self.posted_log = self.base_path / 'ech0_posted_items.json'
        self.posted_items = self._load_posted_items()

        logger.info("ECH0 Social Media Autoposter initialized")

    def _load_posted_items(self) -> List[Dict]:
        """Load history of posted items"""
        if self.posted_log.exists():
            with open(self.posted_log, 'r') as f:
                return json.load(f)
        return []

    def _save_posted_item(self, item: Dict):
        """Save posted item to log"""
        self.posted_items.append({
            **item,
            'posted_at': datetime.now().isoformat()
        })
        with open(self.posted_log, 'w') as f:
            json.dump(self.posted_items, f, indent=2)

    def _extract_content(self, content_file: Path, section: str) -> str:
        """Extract specific section from markdown file"""
        with open(content_file, 'r') as f:
            content = f.read()

        # Find section
        lines = content.split('\n')
        in_section = False
        section_content = []

        for line in lines:
            if section.lower() in line.lower():
                in_section = True
                continue

            if in_section:
                # Stop at next major section
                if line.startswith('##') or line.startswith('---'):
                    if section_content:  # Already collected content
                        break
                section_content.append(line)

        result = '\n'.join(section_content).strip()

        # Extract content between markdown code blocks if present
        if '```' in result:
            parts = result.split('```')
            if len(parts) >= 3:
                result = parts[1].strip()
                # Remove language identifier (markdown, python, etc)
                if '\n' in result:
                    result = '\n'.join(result.split('\n')[1:])

        return result

    # ============ REDDIT POSTING ============

    def post_to_reddit(self, subreddit: str, title: str, body: str) -> bool:
        """Post to Reddit using PRAW"""
        try:
            import praw

            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                username=self.reddit_username,
                password=self.reddit_password,
                user_agent='ECH0 QuLabInfinite Bot v1.0'
            )

            subreddit_obj = reddit.subreddit(subreddit)
            submission = subreddit_obj.submit(title=title, selftext=body)

            logger.info(f"✅ Posted to r/{subreddit}: {submission.url}")
            return True

        except Exception as e:
            logger.error(f"❌ Reddit post failed to r/{subreddit}: {e}")
            return False

    # ============ LINKEDIN POSTING ============

    def post_to_linkedin(self, text: str) -> bool:
        """Post to LinkedIn using API"""
        try:
            import requests

            # Get user URN
            headers = {
                'Authorization': f'Bearer {self.linkedin_token}',
                'Content-Type': 'application/json'
            }

            # Get profile info
            profile_response = requests.get(
                'https://api.linkedin.com/v2/me',
                headers=headers
            )
            profile_response.raise_for_status()
            user_id = profile_response.json()['id']

            # Create post
            post_data = {
                "author": f"urn:li:person:{user_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {
                            "text": text
                        },
                        "shareMediaCategory": "NONE"
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                }
            }

            post_response = requests.post(
                'https://api.linkedin.com/v2/ugcPosts',
                headers=headers,
                json=post_data
            )
            post_response.raise_for_status()

            logger.info(f"✅ Posted to LinkedIn")
            return True

        except Exception as e:
            logger.error(f"❌ LinkedIn post failed: {e}")
            return False

    # ============ FACEBOOK POSTING ============

    def post_to_facebook(self, text: str) -> bool:
        """Post to Facebook page"""
        try:
            import requests

            url = f'https://graph.facebook.com/v18.0/{self.facebook_page_id}/feed'

            data = {
                'message': text,
                'access_token': self.facebook_token
            }

            response = requests.post(url, data=data)
            response.raise_for_status()

            post_id = response.json().get('id')
            logger.info(f"✅ Posted to Facebook: {post_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Facebook post failed: {e}")
            return False

    # ============ TWITTER/X POSTING ============

    def post_to_twitter(self, text: str) -> bool:
        """Post to Twitter/X using tweepy"""
        try:
            import tweepy

            # Authenticate
            auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret)
            auth.set_access_token(self.twitter_access_token, self.twitter_access_secret)
            api = tweepy.API(auth)

            # Post tweet
            tweet = api.update_status(text)

            logger.info(f"✅ Posted to Twitter: {tweet.id}")
            return True

        except Exception as e:
            logger.error(f"❌ Twitter post failed: {e}")
            return False

    def post_twitter_thread(self, tweets: List[str]) -> bool:
        """Post a Twitter thread"""
        try:
            import tweepy

            auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret)
            auth.set_access_token(self.twitter_access_token, self.twitter_access_secret)
            api = tweepy.API(auth)

            previous_tweet_id = None

            for i, tweet_text in enumerate(tweets):
                if previous_tweet_id:
                    tweet = api.update_status(
                        tweet_text,
                        in_reply_to_status_id=previous_tweet_id,
                        auto_populate_reply_metadata=True
                    )
                else:
                    tweet = api.update_status(tweet_text)

                previous_tweet_id = tweet.id
                logger.info(f"✅ Posted tweet {i+1}/{len(tweets)}")
                time.sleep(2)  # Rate limit protection

            return True

        except Exception as e:
            logger.error(f"❌ Twitter thread failed: {e}")
            return False

    # ============ MAIN POSTING LOGIC ============

    def execute_scheduled_post(self, post_item: Dict):
        """Execute a single scheduled post"""
        logger.info(f"Executing post: Day {post_item['day']} - {post_item['section']}")

        # Check if already posted
        for posted in self.posted_items:
            if (posted.get('day') == post_item['day'] and
                posted.get('section') == post_item['section']):
                logger.info(f"Already posted, skipping")
                return

        # Extract content
        if post_item['content_file'] == 'READY_TO_POST_NOW.md':
            content_path = self.content_file
        elif post_item['content_file'] == 'SOCIAL_MEDIA_LAUNCH.md':
            content_path = self.alt_content_file
        else:
            logger.error(f"Unknown content file: {post_item['content_file']}")
            return

        content = self._extract_content(content_path, post_item['section'])

        if not content:
            logger.error(f"Could not extract content for section: {post_item['section']}")
            return

        # Post to each platform
        success = False
        for platform in post_item['platforms']:
            if platform == 'reddit':
                # Extract title from first line
                lines = content.split('\n')
                title = lines[0].strip('#').strip()
                body = '\n'.join(lines[1:]).strip()

                # Determine subreddit from section
                if 'Physics' in post_item['section']:
                    subreddit = 'Physics'
                elif 'Cheminformatics' in post_item['section']:
                    subreddit = 'Cheminformatics'
                elif 'Bioinformatics' in post_item['section']:
                    subreddit = 'Bioinformatics'
                elif 'MachineLearning' in post_item['section']:
                    subreddit = 'MachineLearning'
                else:
                    subreddit = 'Python'  # Default

                success = self.post_to_reddit(subreddit, title, body)

            elif platform == 'linkedin':
                success = self.post_to_linkedin(content)

            elif platform == 'facebook':
                success = self.post_to_facebook(content)

            elif platform == 'twitter':
                # Check if it's a thread
                if '```' in content and '\n\n```' in content:
                    # Extract individual tweets
                    tweets = [t.strip() for t in content.split('```') if t.strip()]
                    success = self.post_twitter_thread(tweets)
                else:
                    success = self.post_to_twitter(content)

        # Log posted item
        if success:
            self._save_posted_item(post_item)

    def check_and_execute_today(self):
        """Check schedule and execute today's posts"""
        today = datetime.now().date()

        for post_item in self.schedule_data['schedule']:
            post_date = datetime.strptime(post_item['date'], '%Y-%m-%d').date()

            if post_date == today:
                # Check time
                post_time = datetime.strptime(post_item['time'], '%H:%M %Z')
                current_time = datetime.now()

                # Execute if time has passed
                if current_time.hour >= post_time.hour and current_time.minute >= post_time.minute:
                    self.execute_scheduled_post(post_item)

    def run_scheduler(self):
        """Run continuous scheduler"""
        logger.info("🚀 ECH0 Autoposter starting...")
        logger.info("Checking schedule every hour...")

        # Check immediately on start
        self.check_and_execute_today()

        # Schedule hourly checks
        schedule.every().hour.do(self.check_and_execute_today)

        # Run forever
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def manual_post_now(self, day: int, section: str):
        """Manually trigger a specific post"""
        for post_item in self.schedule_data['schedule']:
            if post_item['day'] == day and section in post_item['section']:
                logger.info(f"Manual trigger: {post_item['section']}")
                self.execute_scheduled_post(post_item)
                return

        logger.error(f"Post not found: Day {day}, Section: {section}")


def main():
    """Main entry point"""
    poster = SocialMediaPoster()

    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Test mode - just check credentials
            logger.info("Testing API credentials...")
            logger.info(f"Reddit: {'✅' if poster.reddit_client_id else '❌'}")
            logger.info(f"LinkedIn: {'✅' if poster.linkedin_token else '❌'}")
            logger.info(f"Facebook: {'✅' if poster.facebook_token else '❌'}")
            logger.info(f"Twitter: {'✅' if poster.twitter_api_key else '❌'}")

        elif sys.argv[1] == 'manual':
            # Manual mode - trigger specific post
            day = int(sys.argv[2])
            section = sys.argv[3]
            poster.manual_post_now(day, section)

        else:
            logger.error(f"Unknown command: {sys.argv[1]}")
            logger.info("Usage: python ech0_social_media_autoposter.py [test|manual <day> <section>]")

    else:
        # Normal mode - run scheduler
        poster.run_scheduler()


if __name__ == '__main__':
    main()
