# MedSecure AI — 50 Viva Questions & Answers

## Section 1: Deep Learning & Neural Networks

**Q1. What is deep learning and how does it differ from traditional machine learning?**
A: Deep learning is a subset of ML using artificial neural networks with multiple hidden layers. Traditional ML requires manual feature engineering; deep learning learns features automatically from raw data. Example: a CNN learns edge → shape → object detectors without human-coded rules.

**Q2. What is a Convolutional Neural Network (CNN) and why is it suited for image classification?**
A: A CNN uses convolutional layers to extract spatial features (edges, textures, shapes) from images using learnable filters. Pooling layers reduce spatial dimensions, and fully-connected layers make final predictions. CNNs exploit the spatial locality and translation invariance of visual data, making them ideal for image tasks.

**Q3. What is the vanishing gradient problem and how does MobileNetV2 address it?**
A: In deep networks, gradients shrink exponentially during backpropagation through many layers, causing early layers to learn slowly. MobileNetV2 uses residual connections (identity shortcuts) that add the input directly to the output of a block, providing a gradient highway that bypasses non-linear transformations.

**Q4. What is transfer learning?**
A: Transfer learning reuses weights trained on a large dataset (ImageNet) as initialization for a new task. Lower layers capture universal features (edges, textures); only the top layers are retrained for the new task. This drastically reduces training data requirements and training time.

**Q5. What is fine-tuning and how did you implement it?**
A: Fine-tuning unfreezes some layers of the pre-trained base and retrains them with a lower learning rate (LR/10). In train.py, Phase 1 freezes the MobileNetV2 base entirely; Phase 2 unfreezes layers from index 100 onward, allowing adaptation to medicine-specific visual patterns.

---

## Section 2: MobileNetV2

**Q6. What is MobileNetV2 and why was it chosen?**
A: MobileNetV2 is a lightweight CNN designed for mobile/edge deployment using depthwise separable convolutions and inverted residual bottlenecks. It was chosen for: (1) high accuracy on ImageNet, (2) low parameter count (~3.4M), (3) fast inference — critical for real-time analysis.

**Q7. Explain depthwise separable convolutions.**
A: A standard 3×3 conv on C channels requires 3×3×C×K parameters (K output channels). Depthwise separable splits this into: (1) depthwise conv — one 3×3 filter per input channel, and (2) pointwise conv — 1×1 conv to combine channels. This reduces computation by ~8–9×.

**Q8. What is an inverted residual bottleneck in MobileNetV2?**
A: Unlike standard residuals that compress → transform → expand, MobileNetV2 expands (by 6×) → depthwise conv → projects back. The residual connection is applied at the narrow bottleneck. ReLU6 is used for numerical stability in low-precision hardware.

**Q9. What is the input size for MobileNetV2 and why?**
A: MobileNetV2 expects 224×224×3 RGB images — this is the ImageNet standard. Images are resized to this during preprocessing. Using this fixed size ensures compatibility with pre-trained weights.

**Q10. What activation function does MobileNetV2 use and why ReLU6?**
A: ReLU6 = min(max(0, x), 6). The clipping at 6 prevents very large activations that could cause overflow in 8-bit fixed-point hardware (mobile devices). It also acts as mild regularization.

---

## Section 3: OCR with EasyOCR

**Q11. What is EasyOCR and what languages does this project support?**
A: EasyOCR is an open-source OCR library using deep learning (CRAFT for detection + CRNN for recognition). This project initializes it with `['en', 'kn']` — English and Kannada — enabling bilingual text extraction from medicine packages.

**Q12. How does EasyOCR handle text detection vs. recognition?**
A: CRAFT (Character Region Awareness for Text detection) detects character regions and links them into text boxes. The CRNN (Convolutional Recurrent Neural Network) with CTC loss then reads each detected text region character by character.

**Q13. What is OCR confidence and how is it used in the authenticity score?**
A: EasyOCR returns a confidence (0–1) for each text detection. We average these over all detections to get a single OCR confidence. Low confidence suggests blurry, misaligned, or damaged packaging text — a potential counterfeit indicator. It contributes 25% to the authenticity score.

**Q14. What regex patterns are used for field extraction?**
A: Date patterns capture DD/MM/YYYY, MM/YYYY, MMM YYYY formats. Batch number patterns look for keywords like "Batch No.", "Lot No." followed by alphanumeric codes. Manufacturer patterns capture text following "Mfg. by", "Manufactured by". Medicine name uses heuristic: first capitalised noun phrase with optional dosage suffix.

**Q15. How does the system handle missed OCR fields?**
A: Fields default to "Not Detected" if no regex pattern matches. The OCR confidence still contributes to the score — low confidence with missing fields lowers the authenticity score and may raise risk level. Users are shown exactly what was and wasn't extracted.

---

## Section 4: OpenCV & Image Preprocessing

**Q16. Why is preprocessing needed before model inference?**
A: Raw uploaded images vary in size, lighting, noise, and contrast. Preprocessing standardises these: resize to 224×224, noise removal, normalisation to [0,1]. Without it, the model would receive out-of-distribution inputs and produce unreliable predictions.

**Q17. What is CLAHE and why is it used for OCR preprocessing?**
A: Contrast Limited Adaptive Histogram Equalization enhances local contrast in small image tiles (8×8) while limiting over-amplification of noise (clip limit = 2.0). Unlike global histogram equalization, CLAHE improves readability of text in poorly lit or low-contrast regions of medicine packaging.

**Q18. What is adaptive thresholding and how does it help OCR?**
A: Adaptive thresholding converts a grayscale image to binary (black/white) by computing a threshold for each pixel from its local neighbourhood (Gaussian-weighted). Unlike global Otsu thresholding, it handles uneven illumination — common in photos of curved medicine boxes.

**Q19. How is packaging quality measured using OpenCV?**
A: Three metrics are computed: (1) Sharpness via Laplacian variance — blurry images score low, (2) Colour consistency via HSV saturation standard deviation — inconsistent colours suggest fake branding, (3) Text density via contour area ratio — genuine packages have dense labelling. Weighted average = packaging score.

**Q20. What is the Laplacian variance and what does it measure?**
A: The Laplacian operator is a second-order derivative filter that highlights rapid intensity changes (edges). Its variance measures overall edge strength. A high variance → sharp image; low variance → blurry image. Genuine medicine images are typically photographed clearly; counterfeits may be photographed to obscure details.

---

## Section 5: Model Evaluation Metrics

**Q21. Define accuracy, precision, recall, and F1 score.**
A:
- **Accuracy** = (TP + TN) / total — overall correct predictions
- **Precision** = TP / (TP + FP) — of predicted positives, how many are truly positive
- **Recall** = TP / (TP + FN) — of actual positives, how many were found
- **F1** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean, balanced metric

**Q22. Why is F1 score more important than accuracy for this project?**
A: Medicine datasets may be imbalanced (more genuine samples than counterfeit). High accuracy can be achieved by predicting the majority class always. F1 balances precision and recall, penalising both false positives (unnecessary alarms) and false negatives (missed counterfeits), the latter being the critical failure mode.

**Q23. What is a confusion matrix?**
A: A 2×2 table: TP (genuine predicted genuine), TN (counterfeit predicted counterfeit), FP (counterfeit predicted genuine — dangerous!), FN (genuine predicted counterfeit). It shows where the model makes errors.

**Q24. What is an ROC curve and what does AUC represent?**
A: ROC (Receiver Operating Characteristic) plots True Positive Rate vs False Positive Rate at every decision threshold. AUC (Area Under Curve) is a threshold-independent measure of discriminability — 1.0 is perfect, 0.5 is random. High AUC means the model ranks genuine higher than counterfeit reliably.

**Q25. What is class weight balancing and why was it used in training?**
A: If one class has more examples, the model biases toward it. `compute_class_weight('balanced', ...)` from scikit-learn assigns higher loss weight to the minority class, making the model pay more attention to rare but critical counterfeit examples.

---

## Section 6: Flask Web Framework

**Q26. What is Flask and why was it chosen over Django?**
A: Flask is a lightweight WSGI micro-framework for Python. Django is full-stack (ORM, admin, etc.). Flask was chosen because this project has its own database layer (PyMongo) and ML pipeline — the extra Django scaffolding would add unnecessary complexity. Flask's simplicity also aligns with AIML project requirements.

**Q27. How does Flask routing work?**
A: The `@app.route('/path', methods=['GET','POST'])` decorator binds a URL pattern and HTTP method to a Python function. Flask's router matches incoming requests and calls the appropriate view function, passing URL path parameters as arguments.

**Q28. What is Jinja2 and how is it used in this project?**
A: Jinja2 is Flask's default template engine. Templates use `{{ variable }}` for expressions, `{% for %}` / `{% if %}` for control flow, and `{% extends %}` / `{% block %}` for template inheritance. All HTML pages extend `base.html`, keeping the nav and footer DRY.

**Q29. How does the file upload work securely in Flask?**
A: `request.files['image']` retrieves the uploaded file. `secure_filename()` from Werkzeug sanitises the filename to prevent path traversal attacks. A UUID prefix ensures no two uploads collide. File extension is validated against an allowlist (jpg, png, webp). File size is limited to 10 MB via `MAX_CONTENT_LENGTH`.

**Q30. What are Flask flash messages and how are they used?**
A: `flash(message, category)` stores a message in the session. `get_flashed_messages(with_categories=True)` retrieves them in the next rendered template. They display error/success notifications without requiring AJAX, and automatically disappear after being read once.

---

## Section 7: MongoDB Atlas

**Q31. What is MongoDB Atlas and why was it chosen over SQLite?**
A: MongoDB Atlas is a cloud-hosted NoSQL document database. It was chosen because: (1) medicine reports are JSON-like documents with varying fields, (2) Atlas provides free cloud hosting — no local DB server needed, (3) PyMongo provides a Pythonic interface, (4) Atlas scales horizontally for production.

**Q32. How does a MongoDB document differ from a SQL row?**
A: A SQL row has a fixed schema defined at table creation. A MongoDB document is a JSON-like BSON object — any document in a collection can have different fields. This flexibility suits medicine reports that may have varying OCR extraction results.

**Q33. What indexes were created and why?**
A: Three indexes: (1) `timestamp` (DESCENDING) — for fetching most recent records fast, (2) `medicine_name` — for search queries, (3) `prediction` — for filtering by Genuine/Counterfeit in analytics. Indexes dramatically speed up queries on large collections.

**Q34. How does the project handle MongoDB being unavailable?**
A: The `MedSecureDB._connect()` wraps the connection attempt in try/except. If connection fails, `self.client = None`. All DB methods check `is_connected()` and return graceful defaults (None, [], 0). The Flask app falls back to Flask session storage for the last result so the user still sees output.

**Q35. What is PyMongo and how does it connect to Atlas?**
A: PyMongo is MongoDB's official Python driver. It uses a connection URI (mongodb+srv://...) that encodes cluster hostname, credentials, and TLS/SRV settings. The MongoClient object manages a connection pool, automatically reconnecting on transient failures.

---

## Section 8: Transfer Learning & Decision Engine

**Q36. What is ImageNet and why are weights pre-trained on it useful?**
A: ImageNet is a 1.2M image, 1000-class benchmark dataset. MobileNetV2 trained on ImageNet has learned rich visual features — edges, colours, textures, shapes — transferable to other vision tasks. Even though medicine packages are different from ImageNet classes, the low-level features are universal.

**Q37. What is the difference between Phase 1 and Phase 2 training in train.py?**
A: Phase 1 freezes the entire MobileNetV2 base. Only the new Dense head layers learn from scratch with LR=1e-4. Phase 2 unfreezes layers from index 100 onward and retrains with LR=1e-5 (10× lower), allowing the high-level base layers to adapt to medicine packaging while preserving the low-level features.

**Q38. How is the authenticity score computed?**
A: `score = AI_score × 0.50 + OCR_confidence × 0.25 + packaging_score × 0.25`. If the prediction is "Counterfeit", AI_score = 100 - AI_confidence (high confidence in counterfeit = low authenticity). Scores are clamped to [0, 100].

**Q39. How is the risk level determined?**
A: Rules in priority order: (1) Expired medicines → High risk. (2) Counterfeit prediction + score < 80 → High risk. (3) Counterfeit + score ≥ 80 → Medium (edge case). (4) Score ≥ 75 → Low risk. (5) Score 45–74 → Medium risk. (6) Score < 45 → High risk.

**Q40. What are the three components contributing to the authenticity score and their weights?**
A: (1) AI confidence (50%) — the model's certainty about its Genuine/Counterfeit prediction. (2) OCR confidence (25%) — mean text recognition confidence; low = blurry/damaged packaging. (3) Packaging score (25%) — composite from sharpness, colour consistency, and text density analysis.

---

## Section 9: General AI/ML Questions

**Q41. What is data augmentation and which techniques are used?**
A: Data augmentation artificially increases training data by applying random transformations: rotation (±20°), zoom (15%), width/height shift (15%), horizontal flip, brightness adjustment (0.7–1.3×), and Gaussian noise. This reduces overfitting and simulates real-world image capture variations.

**Q42. What is overfitting and how is it prevented in this project?**
A: Overfitting is when a model memorises training data and fails to generalise. Prevention methods used: (1) Data augmentation — training on varied images, (2) Dropout layers (0.3) — randomly zero neurons during training, (3) EarlyStopping — stops training when val_loss stops improving, (4) BatchNormalization — normalises layer inputs.

**Q43. What is EarlyStopping and ReduceLROnPlateau?**
A: EarlyStopping monitors `val_loss` and stops training if it doesn't improve for `patience=5` epochs, restoring the best weights. ReduceLROnPlateau halves the learning rate when `val_loss` stagnates for 3 epochs, allowing finer gradient steps to escape local minima.

**Q44. What is binary cross-entropy loss and when is it used?**
A: Binary cross-entropy = -[y × log(p) + (1-y) × log(1-p)]. It measures the difference between a predicted probability p and a binary label y. It's used when the output is a single sigmoid neuron (two-class problem: Genuine vs Counterfeit).

**Q45. What is the Adam optimizer and why is it used?**
A: Adam (Adaptive Moment Estimation) combines momentum (running average of gradients) and RMSProp (running average of squared gradients). It adapts the learning rate per parameter, converges faster than SGD on most tasks, and is robust to noisy gradients — ideal for fine-tuning vision models.

---

## Section 10: Project-Specific Questions

**Q46. How does the system support Kannada language?**
A: EasyOCR is initialized with `['en', 'kn']` to read Kannada script. The `model/translator.py` module first checks a static dictionary of pre-translated medicine terms, then calls Google Translate via deep-translator for dynamic text. The language choice (en/kn) is stored with each MongoDB report.

**Q47. How is expiry detection implemented?**
A: Regex patterns extract all date-like strings from OCR text. The system identifies the expiry date by looking for keywords ("Exp", "Expiry", "Use Before") or assumes the second date found is the expiry. The date is parsed using Python's datetime module and compared with `datetime.utcnow()` to compute days remaining.

**Q48. How are PDF reports generated and what do they contain?**
A: ReportLab's `SimpleDocTemplate` builds a PDF with: branded header, medicine image, prediction banner (colour-coded), authenticity score, risk level, medicine information table, score breakdown table, recommendation text, and disclaimer. PDFs are saved to `static/reports/` and served via `send_file()`.

**Q49. What security measures are implemented in the Flask application?**
A: (1) `secure_filename()` prevents path traversal attacks, (2) File extension whitelist (jpg/png/webp), (3) Max upload size (10 MB), (4) Secret key for session signing, (5) UUID-based filenames prevent enumeration, (6) `.env` stores credentials outside version control, (7) MongoDB credentials are never hardcoded.

**Q50. How would you improve this system for production deployment at a pharmacy?**
A: (1) Train on a curated dataset of verified genuine vs counterfeit medicines (collaborate with CDSCO/WHO), (2) Add barcode/QR code scanning for DSCSA compliance, (3) Integrate with national drug databases for cross-reference, (4) Implement user authentication and pharmacy-level audit logs, (5) Add a mobile app for pharmacists, (6) Deploy model on edge devices for offline use in rural areas, (7) Set up real-time alerting when counterfeits are detected, (8) Add multi-drug batch verification for wholesalers.

---

*Prepared for final year AIML Major Project viva examination.*
