🎯 **What:**
Removed hardcoded credentials (`admin`, `admin123`, `admin@qulab.ai`) used to create a default admin user during application startup in `api/secure_production_api.py`. It now reads from environment variables (`QU_LAB_ADMIN_USERNAME`, `QU_LAB_ADMIN_PASSWORD`, `QU_LAB_ADMIN_EMAIL`).

⚠️ **Risk:**
If left unfixed, anyone could have accessed the API as an admin using the hardcoded credentials, giving them unauthorized access to secure functionality and potentially exposing sensitive data. This is a severe security vulnerability.

🛡️ **Solution:**
The application now requires environment variables to define the initial admin credentials. If these environment variables are missing, the system gracefully logs a message and skips default user creation, preventing a backdoor from being established. Additionally, automated tests have been added to verify this behavior.
