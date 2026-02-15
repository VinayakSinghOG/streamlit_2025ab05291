# 🚀 Quick Start Guide

Get your Streamlit app running in 5 minutes!

---

## ⚡ Local Development

### 1. Install Dependencies

```bash
cd streamlit-app
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

Your app will open automatically at `http://localhost:8501`

---

## 🌐 Deploy to Streamlit Cloud (3 Steps)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repo, branch, and `app.py`
5. Click "Deploy!"

### Step 3: Wait 2-5 Minutes

Your app will be live at: `https://your-app-name.streamlit.app`

---

## 📋 Quick Checklist

Before deploying, make sure you have:

- [x] `app.py` in repository root
- [x] `requirements.txt` with all dependencies
- [x] `README.md` with documentation
- [x] GitHub repository is **public**
- [x] All files committed and pushed

---

## 🐛 Common Issues

### "Module not found"
**Fix**: Add missing package to `requirements.txt`

### "App not found"
**Fix**: Check file is named exactly `app.py`

### "Deployment failed"
**Fix**: Check Streamlit Cloud logs for specific error

---

## 📚 Full Guides

- **Detailed Deployment**: See `DEPLOYMENT_GUIDE.md`
- **Project Documentation**: See `README.md`

---

**That's it! Your app should be live. Happy deploying! 🎉**
