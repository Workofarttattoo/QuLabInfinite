#!/usr/bin/env python3
"""
LinkedIn OAuth Token Generator
Gets long-lived access token for ECH0 autoposter
"""

import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests

CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID", "SET_ME")
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET", "SET_ME")
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
        html = """
        <html>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1 style="color: green;">✅ Authorization Successful!</h1>
            <p>You can close this window and return to the terminal.</p>
        </body>
        </html>
        """
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        pass  # Suppress server logs

print("\n" + "="*60)
print("LinkedIn OAuth Token Generator")
print("="*60 + "\n")

# Step 1: Authorization URL
auth_url = (
    f"https://www.linkedin.com/oauth/v2/authorization"
    f"?response_type=code"
    f"&client_id={CLIENT_ID}"
    f"&redirect_uri={REDIRECT_URI}"
    f"&scope=w_member_social%20r_liteprofile%20r_emailaddress"
)

print("Step 1: Opening LinkedIn authorization page in your browser...")
print(f"\nIf browser doesn't open, go to:\n{auth_url}\n")

webbrowser.open(auth_url)

# Step 2: Start local server
print("Step 2: Waiting for you to authorize the app...")
print("(Click 'Allow' on the LinkedIn page)\n")

server = HTTPServer(('localhost', 8080), CallbackHandler)
server.handle_request()

if auth_code:
    print(f"Step 3: Received authorization code: {auth_code[:10]}...\n")

    # Step 3: Exchange code for access token
    print("Step 4: Exchanging code for access token...")

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
        expires_in = token_data.get('expires_in', 5184000)  # Default 60 days

        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print("="*60)
        print(f"\nYour LinkedIn Access Token (expires in {expires_in//86400} days):\n")
        print(f"LINKEDIN_ACCESS_TOKEN=\"{access_token}\"")
        print("\n" + "="*60)
        print("\nAdd this to your .env file:")
        print("="*60)
        print(f'\nLINKEDIN_ACCESS_TOKEN="{access_token}"')
        print("\n")

    else:
        print(f"\n❌ Error getting token:")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
else:
    print("\n❌ No authorization code received")
    print("Make sure you clicked 'Allow' on LinkedIn")
