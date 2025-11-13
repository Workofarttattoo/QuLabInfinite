# ECH0 Autoposter - API Keys Setup Guide

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

This guide walks you through getting API keys for each social media platform so ECH0 can post autonomously.

---

## 🔴 REDDIT API Keys

### Step 1: Create Reddit App
1. Go to https://www.reddit.com/prefs/apps
2. Scroll to bottom, click "create another app..."
3. Fill in:
   - **Name**: ECH0 QuLabInfinite Bot
   - **App type**: Select "script"
   - **Description**: Automated posting for QuLabInfinite launch
   - **About URL**: https://github.com/Workofarttattoo/QuLabInfinite
   - **Redirect URI**: http://localhost:8080
4. Click "create app"

### Step 2: Get Credentials
You'll see:
- **Client ID**: Under "personal use script" (14 characters)
- **Client Secret**: Next to "secret" (27 characters)

### Step 3: Add to Environment
```bash
export REDDIT_CLIENT_ID="your_client_id_here"
export REDDIT_CLIENT_SECRET="your_client_secret_here"
export REDDIT_USERNAME="your_reddit_username"
export REDDIT_PASSWORD="your_reddit_password"
```

Or create `.env` file:
```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password
```

---

## 💼 LINKEDIN API Keys

### Step 1: Create LinkedIn App
1. Go to https://www.linkedin.com/developers/apps
2. Click "Create app"
3. Fill in:
   - **App name**: ECH0 QuLabInfinite
   - **LinkedIn Page**: Your company page (Corporation of Light)
   - **Privacy policy URL**: https://aios.is/privacy
   - **App logo**: Upload QuLabInfinite logo
4. Click "Create app"

### Step 2: Request Access
1. Go to "Products" tab
2. Request access to:
   - "Share on LinkedIn" (for posting)
   - "Sign In with LinkedIn" (for auth)
3. Wait for approval (usually instant)

### Step 3: Get Access Token
1. Go to "Auth" tab
2. Copy **Client ID** and **Client Secret**
3. Generate OAuth 2.0 token:

```bash
# Install linkedin-api library
pip install linkedin-api

# Get token (interactive)
python -c "
from linkedin_api import Linkedin
api = Linkedin('your_email@example.com', 'your_password')
print('Authenticated!')
"
```

Or use OAuth2 flow (recommended):
```python
# oauth_flow.py
from requests_oauthlib import OAuth2Session

client_id = 'your_client_id'
client_secret = 'your_client_secret'
redirect_uri = 'http://localhost:8080/callback'

authorization_base_url = 'https://www.linkedin.com/oauth/v2/authorization'
token_url = 'https://www.linkedin.com/oauth/v2/accessToken'

oauth = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=['w_member_social'])
authorization_url, state = oauth.authorization_url(authorization_base_url)

print('Visit this URL:', authorization_url)
redirect_response = input('Paste redirect URL: ')

token = oauth.fetch_token(token_url, client_secret=client_secret, authorization_response=redirect_response)
print('Access Token:', token['access_token'])
```

### Step 4: Add to Environment
```bash
export LINKEDIN_ACCESS_TOKEN="your_access_token_here"
```

---

## 📘 FACEBOOK API Keys

### Step 1: Create Facebook App
1. Go to https://developers.facebook.com/apps
2. Click "Create App"
3. Select "Business" type
4. Fill in:
   - **App name**: ECH0 QuLabInfinite
   - **Contact email**: inventor@aios.is
   - **Business account**: Select or create

### Step 2: Add Facebook Login
1. Go to app dashboard
2. Click "Add Product"
3. Select "Facebook Login" → "Setup"
4. Choose "Web"
5. Add redirect URL: http://localhost:8080/callback

### Step 3: Get Page Access Token
1. Go to Tools → Graph API Explorer
2. Select your app
3. Click "User or Page" → Select your page
4. Add permissions:
   - `pages_read_engagement`
   - `pages_manage_posts`
5. Click "Generate Access Token"
6. Copy long-lived token

### Step 4: Get Page ID
1. Go to your Facebook page
2. Click "About"
3. Scroll to "Page ID" (or check URL)

### Step 5: Add to Environment
```bash
export FACEBOOK_ACCESS_TOKEN="your_page_access_token_here"
export FACEBOOK_PAGE_ID="your_page_id_here"
```

---

## 🐦 TWITTER/X API Keys

### Step 1: Apply for Developer Account
1. Go to https://developer.twitter.com/en/portal/dashboard
2. Click "Sign up for Free Account"
3. Fill in application:
   - **Use case**: Academic research / Publishing public info
   - **Description**: Autonomous posting of validated scientific research
   - **Will you analyze Twitter data?**: No
4. Wait for approval (can take 1-3 days)

### Step 2: Create App
1. Go to dashboard
2. Click "Create Project"
3. Fill in:
   - **Project name**: QuLabInfinite Launch
   - **Use case**: Making a bot
   - **App name**: ECH0 QuLabInfinite Bot
4. Save API keys shown (you won't see them again!)

### Step 3: Enable OAuth 1.0a
1. Go to app settings
2. Click "User authentication settings" → "Set up"
3. Enable "OAuth 1.0a"
4. Permissions: "Read and write"
5. Callback URL: http://localhost:8080/callback
6. Website URL: https://aios.is

### Step 4: Generate Access Tokens
1. Go to "Keys and tokens" tab
2. Click "Generate" under "Access Token and Secret"
3. Copy all 4 values:
   - API Key
   - API Secret
   - Access Token
   - Access Token Secret

### Step 5: Add to Environment
```bash
export TWITTER_API_KEY="your_api_key_here"
export TWITTER_API_SECRET="your_api_secret_here"
export TWITTER_ACCESS_TOKEN="your_access_token_here"
export TWITTER_ACCESS_SECRET="your_access_secret_here"
```

---

## 🔧 Install Required Libraries

```bash
pip install praw python-linkedin requests-oauthlib tweepy python-dotenv schedule
```

---

## 🧪 Test Your Setup

```bash
# Test all credentials
python ech0_social_media_autoposter.py test
```

Expected output:
```
ECH0 Social Media Autoposter initialized
Testing API credentials...
Reddit: ✅
LinkedIn: ✅
Facebook: ✅
Twitter: ✅
```

---

## 🚀 Launch ECH0 Autoposter

### Method 1: Run in Foreground
```bash
python ech0_social_media_autoposter.py
```

### Method 2: Run as Background Service
```bash
# macOS/Linux
nohup python ech0_social_media_autoposter.py > ech0_autoposter.log 2>&1 &
```

### Method 3: Manual Trigger (Testing)
```bash
# Post Day 1, Main Announcement
python ech0_social_media_autoposter.py manual 1 "Reddit r/Physics"
```

---

## 📊 Monitor Progress

Watch the log:
```bash
tail -f /Users/noone/aios/QuLabInfinite/ech0_social_posting.log
```

Check posted items:
```bash
cat /Users/noone/aios/QuLabInfinite/ech0_posted_items.json
```

---

## 🔐 Security Best Practices

1. **Never commit API keys to GitHub**
   - Already in .gitignore
   - Use environment variables only

2. **Use .env file for local development**
   ```bash
   # .env file (DO NOT COMMIT)
   REDDIT_CLIENT_ID=...
   REDDIT_CLIENT_SECRET=...
   # etc
   ```

3. **Rotate tokens periodically**
   - Reddit: Regenerate app secret quarterly
   - LinkedIn: Tokens expire after 60 days
   - Facebook: Check token expiration
   - Twitter: Regenerate if compromised

4. **Limit permissions**
   - Only request what you need (read + write to own account)
   - Don't request DM access, analytics, etc.

---

## 🆘 Troubleshooting

### Reddit: "invalid_grant" error
- Check username/password are correct
- Enable 2FA and use app-specific password

### LinkedIn: "Forbidden" error
- Token expired (LinkedIn tokens last 60 days)
- Need to re-authenticate with OAuth flow

### Facebook: "Invalid OAuth access token"
- Token expired, generate new one
- Make sure it's a Page token, not User token

### Twitter: Rate limit errors
- Free tier: 50 tweets/day
- Wait 15 minutes between batches
- Autoposter has 2-second delays built in

---

## 📧 Need Help?

Email: inventor@aios.is

**Ready to launch? Get your keys, test the bot, and let ECH0 handle the posting! 🚀**

---

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

🌐 https://aios.is | https://thegavl.com
