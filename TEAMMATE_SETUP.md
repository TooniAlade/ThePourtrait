# 🚀 Quick Setup for Teammates

## Problem You're Seeing:
```
The system cannot find the path specified.
```

## ✅ Solution:

### Step 1: One-Time Setup
```bash
# Navigate to the project root (ThePourtrait folder)
cd ThePourtrait

# Run the setup script (this installs everything)
.\setup_for_teammates.bat
```

### Step 2: Start the Gallery
```bash
# Navigate to BP-work folder
cd BP-work

# Start the server
.\start_with_email.bat
```

### Step 3: Open Browser
Go to: **http://localhost:5000**

---

## What This Fixes:

The original batch file had hardcoded paths like:
- `C:\Users\itunu\OneDrive\Documents\...` ❌

The new version uses relative paths that work on any computer:
- `%~dp0` (current directory) ✅
- Automatically finds your Python virtual environment ✅

## If You Still Have Issues:

1. **Make sure Python is installed**: `python --version`
2. **Re-run the setup**: `.\setup_for_teammates.bat`
3. **Check you're in the right folder**: You should see `gallery_server.py` in the BP-work folder

---

🎉 **Once working, you'll have the full marble art gallery with email functionality!**