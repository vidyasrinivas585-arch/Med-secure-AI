# MedSecure AI — Complete VS Code Setup Guide

## ── STEP 1: Prerequisites ──────────────────────────────────────────────────

Install the following before starting:

| Tool | Download | Notes |
|------|----------|-------|
| Python 3.10+ | https://python.org | Add to PATH during install |
| VS Code | https://code.visualstudio.com | Recommended IDE |
| Git | https://git-scm.com | For version control |
| MongoDB Atlas | https://www.mongodb.com/atlas | Free M0 cluster |

**Recommended VS Code Extensions:**
- Python (Microsoft)
- Pylance
- MongoDB for VS Code
- Thunder Client (for API testing)

---

## ── STEP 2: Project Setup ──────────────────────────────────────────────────

```bash
# 1. Navigate to project folder
cd medsecure_ai

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install all dependencies
pip install -r requirements.txt

# NOTE: First install takes 5–10 minutes (TensorFlow + EasyOCR are large)
```

---

## ── STEP 3: MongoDB Atlas Configuration ────────────────────────────────────

1. Go to https://cloud.mongodb.com
2. Create a **free M0 cluster** (no credit card needed)
3. Go to **Security → Database Access** → Add new database user
   - Username: `medsecure_user`
   - Password: (generate a strong password)
   - Role: `Atlas Admin`
4. Go to **Security → Network Access** → Add IP Address → `0.0.0.0/0` (allow all, for development)
5. Go to **Deployment → Database** → Connect → Drivers → Python
6. Copy the connection string (looks like):
   ```
   mongodb+srv://medsecure_user:<password>@cluster0.xxxxx.mongodb.net/
   ```

---

## ── STEP 4: Environment Configuration ─────────────────────────────────────

```bash
# Copy the example env file
cp .env.example .env

# Open .env and fill in:
```

Edit `.env`:
```
MONGODB_URI=mongodb+srv://medsecure_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/medsecure_ai?retryWrites=true&w=majority
SECRET_KEY=any_random_long_string_here_change_this
FLASK_ENV=development
FLASK_DEBUG=True
```

---

## ── STEP 5: Dataset Preparation ────────────────────────────────────────────

**Option A — Use your own images (recommended for production):**
```bash
# Add genuine medicine photos to:
dataset/genuine/    (jpg, png, webp)

# Add counterfeit / fake medicine photos to:
dataset/counterfeit/    (jpg, png, webp)

# Recommended: 200+ images per class
```

**Option B — Generate synthetic placeholder dataset (for quick testing):**
```bash
python model/prepare_dataset.py
# When asked: type 'y' to create synthetic images
```

---

## ── STEP 6: Train the Model ─────────────────────────────────────────────────

```bash
# Verify dataset and augment if needed
python model/prepare_dataset.py

# Train MobileNetV2 (takes 15–60 min depending on dataset size and hardware)
python model/train.py

# You will see:
# Phase 1: Training top layers...
# Phase 2: Fine-tuning...
# ✅ Model saved to: trained_model/medicine_classifier.h5

# Evaluate model performance
python model/evaluate.py
# Generates: trained_model/evaluation/ (confusion matrix, ROC curve, metrics.json)
```

**GPU Acceleration (optional):**
If you have an NVIDIA GPU, install CUDA + cuDNN. TensorFlow will auto-detect it and training will be 5–10× faster.

---

## ── STEP 7: Run the Application ────────────────────────────────────────────

```bash
# Make sure venv is activated and .env is configured

python app.py

# Expected output:
# ✅ Connected to MongoDB Atlas successfully.
# * Running on http://0.0.0.0:5000
# * Debug mode: on
```

Open your browser: **http://localhost:5000**

---

## ── STEP 8: Using the Application ──────────────────────────────────────────

1. Click **"Analyze a Medicine"** on the home page
2. Select language: **English** or **ಕನ್ನಡ (Kannada)**
3. Drag-and-drop or click to upload a medicine package image
4. Click **"Analyze Medicine"**
5. Wait 3–8 seconds for the analysis to complete
6. View the result: prediction, score, risk level, OCR fields
7. Click **"Download PDF Report"** to save the report
8. Visit **History** to see all past analyses

---

## ── Troubleshooting ─────────────────────────────────────────────────────────

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv activated |
| MongoDB connection failed | Check `.env` URI, whitelist your IP in Atlas |
| EasyOCR slow on first run | It downloads language models (~200MB) once |
| `Model not found` error | Run `python model/train.py` first |
| Image upload fails | Check file is jpg/png/webp and < 10MB |
| Port 5000 already in use | Change port in app.py: `app.run(port=5001)` |

---

## ── VS Code Debug Configuration ────────────────────────────────────────────

Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "MedSecure AI",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/app.py",
      "env": {
        "FLASK_ENV": "development",
        "FLASK_DEBUG": "1"
      },
      "jinja": true,
      "justMyCode": false
    }
  ]
}
```

Press **F5** in VS Code to launch with debugging enabled.

