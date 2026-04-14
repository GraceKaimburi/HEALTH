# Firebase + Streamlit on Vercel Deployment Guide

## ✅ Local Setup

1. **Firebase Service Account**
   - File: `firebase-service-account.json` (already downloaded)
   - Keep it **local only** - never commit to GitHub
   - It's already in `.gitignore`

2. **Run Locally**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
   Visit: http://localhost:8501

---

## 🚀 Deploy to Vercel

### Prerequisites
- GitHub account with your repo
- Vercel account (https://vercel.com)

### Step 1: Store Firebase Credentials in Vercel

Since `firebase-service-account.json` cannot be committed to Git, you must add it as an environment variable:

**Option A: Convert JSON to Base64 String** (Recommended for Vercel)

1. Run this command locally:
   ```bash
   cat firebase-service-account.json | base64
   ```

2. Copy the entire base64 string

3. In Vercel Dashboard:
   - Go to your project settings
   - Navigate to **Environment Variables**
   - Add new variable:
     - **Name**: `FIREBASE_SERVICE_ACCOUNT_BASE64`
     - **Value**: Paste the base64 string
     - Apply to: Production, Preview, Development

### Step 2: Update `app.py` to Load from Environment

The app will first check for the file, then fall back to environment variable:

```python
import os
import base64

firebase_service_account_path = Path('firebase-service-account.json')
db = None
firebase_initialized = False

# Try to load from environment variable (for Vercel)
if not firebase_service_account_path.exists():
    firebase_cred_base64 = os.getenv('FIREBASE_SERVICE_ACCOUNT_BASE64')
    if firebase_cred_base64:
        firebase_cred_json = base64.b64decode(firebase_cred_base64).decode('utf-8')
        firebase_service_account_path.write_text(firebase_cred_json)

if firebase_admin and firestore and firebase_service_account_path.exists():
    try:
        cred = credentials.Certificate(str(firebase_service_account_path))
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_initialized = True
    except Exception as e:
        db = None
```

### Step 3: Deploy

1. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Add Firebase integration"
   git push origin main
   ```

2. In Vercel Dashboard:
   - Click **"New Project"**
   - Import your GitHub repository
   - Configure environment variables (from Step 1)
   - Deploy!

### Step 4: Enable Vercel + GitHub Auto-Deploy

Once deployed, Vercel will auto-deploy whenever you push to `main` branch.

---

## 🔒 Security Notes

✅ **DO**: 
- Keep `firebase-service-account.json` in `.gitignore`
- Use Vercel environment variables for credentials
- Never hardcode API keys

❌ **DON'T**:
- Commit `firebase-service-account.json` to GitHub
- Share credentials in code comments or issues
- Use `DEFAULT_PATIENTS` for production data

---

## 🧪 Testing Firebase Connection

Run this locally to verify Firebase works:

```bash
python -c "
import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path

cred = credentials.Certificate('firebase-service-account.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
print('✅ Firebase connected!')
print(f'Project: {db._client._target.database_string}')
"
```

---

## 📚 Vercel Streamlit Docs

- https://vercel.com/docs/concepts/python/python-runtime
- Configure in `vercel.json` if needed
