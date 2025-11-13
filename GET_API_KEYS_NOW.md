# GET API KEYS NOW - Step-by-Step Walkthrough

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

Let's get your API keys right now. Follow these steps **in order**.

---

## 🔴 REDDIT (10 minutes)

### Open These Links in New Tabs:
1. https://www.reddit.com/prefs/apps
2. https://www.reddit.com/login (if not logged in)

### Steps:
1. **Log in to Reddit** with your account
2. **Go to https://www.reddit.com/prefs/apps**
3. **Scroll to the very bottom**
4. **Click "create another app..."** (big button)
5. **Fill in the form:**
   ```
   Name: ECH0 QuLabInfinite Bot

   App type: [Select "script" - click the radio button]

   Description: Automated posting for QuLabInfinite validated science labs

   About URL: https://github.com/Workofarttattoo/QuLabInfinite

   Redirect URI: http://localhost:8080
   ```
6. **Click "create app"** (bottom of form)

### You'll See:
```
ECH0 QuLabInfinite Bot
personal use script
[14 character code] ← This is your CLIENT_ID

secret: [27 character code] ← This is your CLIENT_SECRET
```

### Save These:
```bash
REDDIT_CLIENT_ID="paste_the_14_character_code_here"
REDDIT_CLIENT_SECRET="paste_the_27_character_code_here"
REDDIT_USERNAME="your_reddit_username"
REDDIT_PASSWORD="your_reddit_password"
```

✅ **Reddit Done!**

---

## 🐦 TWITTER/X (15 minutes - NEEDS APPROVAL)

### Open These Links:
1. https://developer.twitter.com/en/portal/petition/essential/basic-info
2. https://twitter.com/login (if not logged in)

### Steps:

#### Part 1: Apply for Developer Account
1. **Log in to Twitter/X**
2. **Go to https://developer.twitter.com/en/portal/petition/essential/basic-info**
3. **Fill in application:**
   ```
   What's your name?: Joshua Hendricks Cole

   Country of operation: United States

   What use case are you interested in?:
   [Select] "Making a bot"

   Will you make Twitter content or derived information available to a government entity?:
   [Select] "No"
   ```
4. **Click "Next"**

#### Part 2: Describe Your App
```
In your words, describe your project:

I'm launching QuLabInfinite, a scientifically validated computational laboratory suite with 29 research labs. The bot will autonomously post educational content about our experimental validation results (100% pass rate against peer-reviewed physics). Posts will include quantum mechanics, nanotechnology, drug delivery, and materials science simulations. All content is open source and aimed at the scientific research community.

Are you planning to analyze Twitter data?
[Select] "No"

Will your app use Tweet, Retweet, Like, Follow, or Direct Message functionality?
[Select] "Yes"

Describe how you will use this functionality:
The bot will post original content (tweets and threads) about scientific validation results. No automation of likes, follows, or DMs. Only publishing educational content about validated physics simulations. Example: "Our quantum dot simulator matches experimental data with 0.4% error - here's how we validated it."

Do you intend to analyze Twitter data?
[Select] "No"
```
5. **Check "I have read and agree to the Developer Agreement and Policy"**
6. **Click "Submit"**

#### Part 3: Wait for Approval
- **Typical wait time**: 1-3 days (sometimes instant)
- **Check email** for approval notification

#### Part 4: Once Approved
1. **Go to https://developer.twitter.com/en/portal/dashboard**
2. **Click "Create Project"**
   ```
   Project name: QuLabInfinite Launch
   Use case: Making a bot
   Description: Autonomous posting of validated scientific research
   ```
3. **Click "Next"**
4. **Create App:**
   ```
   App name: ECH0QuLabBot
   ```
5. **IMPORTANT: SAVE THESE IMMEDIATELY (shown once!):**
   ```
   API Key: [copy this]
   API Secret: [copy this]
   Bearer Token: [copy this]
   ```

#### Part 5: Enable Posting
1. **In app dashboard, click "User authentication settings"**
2. **Click "Set up"**
3. **Enable "OAuth 1.0a"**
4. **Permissions: Select "Read and Write"**
5. **Callback URL:** `http://localhost:8080/callback`
6. **Website URL:** `https://aios.is`
7. **Click "Save"**

#### Part 6: Generate Access Token
1. **Go to "Keys and tokens" tab**
2. **Under "Authentication Tokens", click "Generate"**
3. **Save these:**
   ```
   Access Token: [copy this]
   Access Token Secret: [copy this]
   ```

### Save All Four:
```bash
TWITTER_API_KEY="paste_api_key_here"
TWITTER_API_SECRET="paste_api_secret_here"
TWITTER_ACCESS_TOKEN="paste_access_token_here"
TWITTER_ACCESS_SECRET="paste_access_token_secret_here"
```

✅ **Twitter Done!** (once approved)

---

## 💼 LINKEDIN (20 minutes)

### Open These Links:
1. https://www.linkedin.com/developers/apps
2. https://www.linkedin.com/login (if not logged in)

### Prerequisites:
- You need a **LinkedIn Company Page** (Corporation of Light)
- If you don't have one, create it first: https://www.linkedin.com/company/setup/new/

### Steps:

#### Part 1: Create LinkedIn App
1. **Go to https://www.linkedin.com/developers/apps**
2. **Click "Create app"** (top right)
3. **Fill in:**
   ```
   App name: ECH0 QuLabInfinite

   LinkedIn Page: [Select "Corporation of Light" page]

   Privacy policy URL: https://aios.is/privacy

   App logo: [Upload a logo - can be QuLabInfinite logo or any 300x300 image]

   Legal agreement: [Check the box]
   ```
4. **Click "Create app"**

#### Part 2: Request Products
1. **In app dashboard, click "Products" tab**
2. **Find "Share on LinkedIn"**
3. **Click "Request access"**
4. **Wait for approval** (usually instant, sometimes 24 hours)
5. **Also request "Sign In with LinkedIn using OpenID Connect"**

#### Part 3: Get Client Credentials
1. **Click "Auth" tab**
2. **You'll see:**
   ```
   Client ID: [10 digit number]
   Client Secret: [random string]
   ```
3. **Save these for now**

#### Part 4: Generate Access Token (OAuth Flow)

**Option A: Quick & Dirty (Short-lived token - 60 days)**

1. **Go to LinkedIn while logged in**
2. **Open browser console** (F12 or Right-click → Inspect)
3. **Go to Application tab → Cookies → linkedin.com**
4. **Find cookie named "li_at"**
5. **Copy the value** - this is your session token

```bash
LINKEDIN_ACCESS_TOKEN="paste_li_at_cookie_value_here"
```

⚠️ **This expires in 60 days** - you'll need to re-get it

**Option B: Proper OAuth (Recommended)**

I'll create a helper script for you:

```bash
cd /Users/noone/aios/QuLabInfinite
python3 << 'OAUTH_SCRIPT'
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests

CLIENT_ID = input("Paste your LinkedIn Client ID: ")
CLIENT_SECRET = input("Paste your LinkedIn Client Secret: ")

REDIRECT_URI = "http://localhost:8080/callback"
auth_code = None

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        query = urlparse(self.path).query
        params = parse_qs(query)
        auth_code = params.get('code', [None])[0]

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<h1>Authorization successful! Close this window.</h1>")

    def log_message(self, format, *args):
        pass

# Step 1: Open browser for authorization
auth_url = f"https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=w_member_social%20r_liteprofile"

print("\n1. Opening browser for authorization...")
print(f"   URL: {auth_url}")
webbrowser.open(auth_url)

# Step 2: Start local server to receive callback
print("2. Waiting for authorization...")
server = HTTPServer(('localhost', 8080), CallbackHandler)
server.handle_request()

if auth_code:
    print(f"3. Got authorization code: {auth_code[:10]}...")

    # Step 3: Exchange code for access token
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }

    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data['access_token']
        expires_in = token_data['expires_in']

        print(f"\n✅ SUCCESS! Access token expires in {expires_in} seconds ({expires_in/86400:.0f} days)")
        print(f"\nYour LinkedIn Access Token:")
        print(f"LINKEDIN_ACCESS_TOKEN=\"{access_token}\"")
    else:
        print(f"\n❌ Error getting token: {response.text}")
else:
    print("\n❌ No authorization code received")
OAUTH_SCRIPT
```

Run this script and follow the prompts.

✅ **LinkedIn Done!**

---

## 📘 FACEBOOK (15 minutes)

### Open These Links:
1. https://developers.facebook.com/apps
2. https://www.facebook.com/login (if not logged in)

### Prerequisites:
- You need a **Facebook Page** (Corporation of Light or QuLabInfinite)
- If you don't have one, create it first: https://www.facebook.com/pages/create

### Steps:

#### Part 1: Create Facebook App
1. **Go to https://developers.facebook.com/apps**
2. **Click "Create App"**
3. **Select "Other"** as app type
4. **Select "Business"** as app type
5. **Click "Next"**
6. **Fill in:**
   ```
   App name: ECH0 QuLabInfinite
   App contact email: inventor@aios.is
   ```
7. **Click "Create app"**
8. **Complete security check** (CAPTCHA)

#### Part 2: Add Products
1. **In app dashboard, click "Add Products"**
2. **Find "Facebook Login"**
3. **Click "Set Up"**
4. **Select "Web" platform**
5. **Site URL:** `https://aios.is`
6. **Click "Save"** and **"Continue"**

#### Part 3: Configure Facebook Login
1. **In left menu, click "Facebook Login" → "Settings"**
2. **Valid OAuth Redirect URIs:** `http://localhost:8080/callback`
3. **Click "Save Changes"**

#### Part 4: Get Page Access Token

**Quick Method using Graph API Explorer:**

1. **Go to https://developers.facebook.com/tools/explorer**
2. **Select your app** from dropdown (top right)
3. **Click "Generate Access Token"**
4. **A popup appears - click "Continue as [Your Name]"**
5. **Grant permissions** when asked
6. **Now you have a USER token (short-lived)**

**Convert to Page Token (Long-lived):**

1. **Still in Graph API Explorer**
2. **In "User or Page" dropdown, select your page**
3. **Click "Generate Access Token" again**
4. **Copy the new token** - this is your PAGE ACCESS TOKEN
5. **Save it:**

```bash
FACEBOOK_ACCESS_TOKEN="paste_page_access_token_here"
```

#### Part 5: Get Page ID
1. **Go to your Facebook page**
2. **Click "About" in left menu**
3. **Scroll down to "Page ID"**
4. **Copy the number**

Or get it via Graph API Explorer:
1. **In Graph API Explorer**
2. **Type `me?fields=id,name` in the query box**
3. **Click "Submit"**
4. **Copy the "id" field**

```bash
FACEBOOK_PAGE_ID="paste_page_id_here"
```

✅ **Facebook Done!**

---

## 📝 CREATE YOUR .env FILE

Now put it all together:

```bash
cd /Users/noone/aios/QuLabInfinite

cat > .env << 'ENV_FILE'
# Reddit API Keys
REDDIT_CLIENT_ID="paste_your_14_char_code"
REDDIT_CLIENT_SECRET="paste_your_27_char_code"
REDDIT_USERNAME="your_reddit_username"
REDDIT_PASSWORD="your_reddit_password"

# LinkedIn API Keys
LINKEDIN_ACCESS_TOKEN="paste_your_linkedin_token"

# Facebook API Keys
FACEBOOK_ACCESS_TOKEN="paste_your_page_token"
FACEBOOK_PAGE_ID="paste_your_page_id"

# Twitter API Keys (after approval)
TWITTER_API_KEY="paste_your_api_key"
TWITTER_API_SECRET="paste_your_api_secret"
TWITTER_ACCESS_TOKEN="paste_your_access_token"
TWITTER_ACCESS_SECRET="paste_your_access_token_secret"
ENV_FILE

echo "✅ .env file created!"
```

---

## 🧪 TEST YOUR SETUP

```bash
# Install dependencies
pip install praw python-dotenv tweepy requests requests-oauthlib python-linkedin

# Load .env and test
python3 << 'TEST_SCRIPT'
import os
from dotenv import load_dotenv

load_dotenv()

print("\n🔍 Checking API Keys...\n")

# Reddit
reddit_client = os.getenv('REDDIT_CLIENT_ID')
reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_user = os.getenv('REDDIT_USERNAME')
print(f"Reddit Client ID: {'✅ ' + reddit_client[:4] + '...' if reddit_client else '❌ Missing'}")
print(f"Reddit Secret: {'✅ ' + reddit_secret[:4] + '...' if reddit_secret else '❌ Missing'}")
print(f"Reddit Username: {'✅ ' + reddit_user if reddit_user else '❌ Missing'}")

# LinkedIn
linkedin_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
print(f"\nLinkedIn Token: {'✅ ' + linkedin_token[:10] + '...' if linkedin_token else '❌ Missing'}")

# Facebook
facebook_token = os.getenv('FACEBOOK_ACCESS_TOKEN')
facebook_page = os.getenv('FACEBOOK_PAGE_ID')
print(f"\nFacebook Token: {'✅ ' + facebook_token[:10] + '...' if facebook_token else '❌ Missing'}")
print(f"Facebook Page ID: {'✅ ' + facebook_page if facebook_page else '❌ Missing'}")

# Twitter
twitter_key = os.getenv('TWITTER_API_KEY')
twitter_secret = os.getenv('TWITTER_API_SECRET')
twitter_token = os.getenv('TWITTER_ACCESS_TOKEN')
twitter_token_secret = os.getenv('TWITTER_ACCESS_SECRET')
print(f"\nTwitter API Key: {'✅ ' + twitter_key[:4] + '...' if twitter_key else '❌ Missing (pending approval)'}")
print(f"Twitter Secret: {'✅ ' + twitter_secret[:4] + '...' if twitter_secret else '❌ Missing'}")
print(f"Twitter Access Token: {'✅ ' + twitter_token[:10] + '...' if twitter_token else '❌ Missing'}")
print(f"Twitter Access Secret: {'✅ ' + twitter_token_secret[:4] + '...' if twitter_token_secret else '❌ Missing'}")

print("\n" + "="*50)
ready_count = sum([
    bool(reddit_client and reddit_secret and reddit_user),
    bool(linkedin_token),
    bool(facebook_token and facebook_page),
    bool(twitter_key and twitter_secret and twitter_token and twitter_token_secret)
])
print(f"\n✅ {ready_count}/4 platforms ready")

if ready_count >= 3:
    print("\n🚀 Ready to launch! (Twitter pending approval is OK)")
elif ready_count >= 2:
    print("\n⚠️  Almost there! Get remaining keys and re-run test.")
else:
    print("\n❌ Need more API keys. Follow steps above.")
TEST_SCRIPT
```

---

## ⏱️ TIME ESTIMATE

- **Reddit**: 5 minutes (instant)
- **LinkedIn**: 15 minutes (instant approval usually)
- **Facebook**: 15 minutes (instant)
- **Twitter**: 15 minutes + 1-3 day wait for approval

**You can launch with 3/4 platforms while waiting for Twitter approval!**

---

## 🚀 NEXT STEPS

Once you have keys:

```bash
# Test the autoposter
python ech0_social_media_autoposter.py test

# Launch for real
python ech0_social_media_autoposter.py
```

---

**Need help with any specific platform? Let me know which one!**

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

🌐 https://aios.is | https://thegavl.com
