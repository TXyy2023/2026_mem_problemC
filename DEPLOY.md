# Streamlit App Deployment Guide

This guide explains how to deploy the `csv_display.py` application so others can access it remotely.

## 1. Prerequisites
Ensure you have the `requirements.txt` file in your project directory (I have just created this for you).
It contains:
```text
streamlit
pandas
plotly
scipy
numpy
```

## 2. Deployment Options

### Option A: Streamlit Community Cloud (Recommended & Free)
This is the easiest way to deploy Streamlit apps.
1. Push your code (`csv_display.py`, `requirements.txt`, and the CSV data files) to a **GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in.
3. Click **"New app"**.
4. Select your GitHub repository, branch, and the file `csv_display.py`.
5. Click **"Deploy"**.
6. **Important**: Since your app reads local CSV files (`/Users/a1234/...`), you MUST update the paths in `csv_display.py` to be relative paths (e.g., just `2026_MCM_Problem_C_Data.csv`) before committing to GitHub.

### Option B: Deploy on a Linux Server (VPS)
If you have a server (Ubuntu/CentOS):
1. Copy all files to the server.
2. Install Python and pip.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app in the background (using `nohup` or `tmux`):
   ```bash
   nohup streamlit run csv_display.py --server.port 8501 &
   ```
5. Access via `http://YOUR_SERVER_IP:8501`.

### Option C: Quick Sharing (ngrok)
If you just want to share it temporarily from your currently running Mac without a server:
1. Don't stop your running Streamlit app.
2. Install `ngrok` (if not installed): `brew install ngrok`.
3. Run:
   ```bash
   ngrok http 8501
   ```
4. Send the generated `https://....ngrok-free.app` link to your team.

## 3. Configuration (Optional)
You can add a `.streamlit/config.toml` file to customize the server:
```toml
[server]
headless = true
port = 8501
enableCORS = false
```

## Critical Code Change Required
**Before deploying**, you must change the absolute file paths in `csv_display.py` to relative paths, otherwise the cloud server won't find your files.

**Example Change:**
```python
# Change this:
RESULTS_FILE = '/Users/a1234/Desktop/美赛/MCM_Problem_C_Results.csv'
# To this:
RESULTS_FILE = 'MCM_Problem_C_Results.csv'
```
