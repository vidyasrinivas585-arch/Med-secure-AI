# 🛡️ MedSecure AI
### AI-Based Detection of Counterfeit Medicines

> *Addressing the WHO 1-in-5 Drug Crisis using Computer Vision, OCR, and Deep Learning*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey?logo=flask)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)](https://tensorflow.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)](https://mongodb.com/atlas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [MongoDB Atlas Setup](#mongodb-atlas-setup)
- [Training the Model](#training-the-model)
- [Screenshots](#screenshots)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🔍 Overview

MedSecure AI is a production-ready web application that detects counterfeit medicines using:

- **MobileNetV2** transfer learning for visual authenticity classification
- **EasyOCR** for bilingual text extraction (English + Kannada)
- **OpenCV** for packaging quality analysis
- **Decision Engine** combining multiple signals into an authenticity score (0–100)
- **MongoDB Atlas** for cloud-based report storage
- **ReportLab** for downloadable PDF reports

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 AI Detection | MobileNetV2 classifies medicine as Genuine / Counterfeit |
| 📝 OCR Extraction | Extracts name, manufacturer, batch number, and expiry date |
| 📅 Expiry Check | Compares expiry date against today; calculates days remaining |
| 💯 Authenticity Score | Weighted 0–100 composite score from AI + OCR + packaging |
| ⚠️ Risk Assessment | Low / Medium / High risk with actionable recommendation |
| 🌐 Bilingual Support | Full English and Kannada output |
| 📊 History Dashboard | All analyses stored in MongoDB Atlas with search |
| 📄 PDF Reports | Downloadable report with image, scores, and recommendation |

---

## 🛠️ Tech Stack

```
Frontend  → HTML5 · CSS3 · Bootstrap 5 · JavaScript
Backend   → Python 3.10+ · Flask 2.3
AI/ML     → TensorFlow 2.13 · Keras · MobileNetV2
Vision    → OpenCV 4.8
OCR       → EasyOCR (English + Kannada)
Database  → MongoDB Atlas · PyMongo
Reports   → ReportLab
Trans.    → Deep Translator (Google)
```

---

## 📁 Project Structure

```
medsecure_ai/
├── app.py                    ← Flask application entry point
├── requirements.txt
├── .env.example              ← Copy to .env and configure
│
├── dataset/
│   ├── genuine/              ← Class 0: genuine medicine images
│   └── counterfeit/          ← Class 1: counterfeit medicine images
│
├── model/
│   ├── train.py              ← MobileNetV2 training script
│   ├── evaluate.py           ← Metrics & evaluation plots
│   ├── decision_engine.py    ← Final prediction + scoring
│   ├── translator.py         ← English ↔ Kannada translation
│   └── prepare_dataset.py    ← Dataset verification & augmentation
│
├── ocr/
│   ├── extractor.py          ← EasyOCR field extraction
│   └── preprocessor.py       ← OpenCV image preprocessing
│
├── database/
│   └── mongodb.py            ← MongoDB Atlas CRUD operations
│
├── reports/
│   └── pdf_generator.py      ← ReportLab PDF generation
│
├── trained_model/
│   ├── medicine_classifier.h5
│   ├── class_indices.json
│   └── evaluation/           ← Plots and metrics after evaluate.py
│
├── static/
│   ├── css/main.css
│   ├── js/main.js
│   ├── uploads/              ← Uploaded images
│   └── reports/              ← Generated PDF reports
│
└── templates/
    ├── base.html
    ├── index.html
    ├── upload.html
    ├── result.html
    └── history.html
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- VS Code (recommended)
- MongoDB Atlas account (free tier works)
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/medsecure-ai.git
cd medsecure-ai

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your MongoDB Atlas connection string + secret key
```

### 3. Prepare Dataset & Train Model

```bash
# Verify / create placeholder dataset
python model/prepare_dataset.py

# Train MobileNetV2 (requires dataset/genuine and dataset/counterfeit)
python model/train.py

# Evaluate model performance
python model/evaluate.py
```

### 4. Run Application

```bash
python app.py
# → Open http://localhost:5000
```

---

## 🌿 MongoDB Atlas Setup

1. Create a free account at [mongodb.com/atlas](https://mongodb.com/atlas)
2. Create a new **free** M0 cluster
3. Add your IP address to the **Network Access** whitelist
4. Create a database user (username + password)
5. Click **Connect → Drivers → Python** and copy the connection string
6. Paste it into `.env` as `MONGODB_URI`

The database `medsecure_ai` and collection `medicine_reports` are created automatically on first run.

---

## 🧠 Training the Model

```bash
# Step 1 — Add images
# Place genuine medicine images in:   dataset/genuine/
# Place counterfeit images in:        dataset/counterfeit/
# Recommended: 200+ images per class

# Step 2 — Augment dataset
python model/prepare_dataset.py

# Step 3 — Train (Phase 1 + Phase 2 fine-tuning)
python model/train.py
# Output: trained_model/medicine_classifier.h5

# Step 4 — Evaluate
python model/evaluate.py
# Output: trained_model/evaluation/
```

---

## 📸 Screenshots

| Page | Description |
|------|-------------|
| Home | Landing page with live stats |
| Upload | Drag-and-drop image upload with language selector |
| Result | Full analysis with scores, risk level, and recommendation |
| History | Searchable log of all analyses |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/upload` | Upload form |
| POST | `/analyze` | Run analysis pipeline |
| GET | `/result/<id>` | Show result |
| GET | `/history` | All reports |
| GET | `/search?q=<term>` | Search reports (JSON) |
| GET | `/download/<id>` | Download PDF report |
| GET | `/api/stats` | JSON statistics |

---

## 🚢 Deployment

### Render
```bash
# Add a render.yaml or connect GitHub repo
# Set environment variables in Render dashboard
# Build command: pip install -r requirements.txt
# Start command: gunicorn app:app
```

### Railway
```bash
railway login
railway init
railway up
# Set MONGODB_URI in Railway dashboard
```

### PythonAnywhere
```bash
# Upload project files
# Create virtualenv and install requirements.txt
# Set WSGI to point to app:app
# Add MONGODB_URI to environment variables
```

---

## 🔮 Future Enhancements

- [ ] Mobile app (Flutter / React Native)
- [ ] Barcode / QR code scanning
- [ ] Drug interaction checker
- [ ] Government drug database integration (CDSCO)
- [ ] Real-time batch verification for pharmacies
- [ ] SMS / email alert system for counterfeit detections
- [ ] Admin dashboard with analytics

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Built as an AIML Major Project demonstrating production-grade AI integration.

---

> ⚠️ **Disclaimer:** MedSecure AI is a research prototype. Results are for screening purposes only. Always consult a licensed pharmacist or healthcare professional before making any medical decisions.
