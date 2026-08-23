"""
model/recommendation_engine.py
MedSecure AI - Recommendation Engine
FIXED: Alternative medicine always shown for Counterfeit/Expired/Suspicious
"""

import urllib.parse


# ─────────────────────────────────────────────────────────────────────────────
# MEDICINE DATABASE — expanded with categories and alternatives
# ─────────────────────────────────────────────────────────────────────────────

MEDICINE_DB = {

    # ── Pain Relief / Fever ──────────────────────────────────────────────────
    "paracetamol":    {"alt": "Crocin 500mg or Dolo 650mg",      "category": "Pain Relief"},
    "dolo":           {"alt": "Calpol 500mg or Paracetamol IP",   "category": "Pain Relief"},
    "dolo 650":       {"alt": "Calpol 500mg or Crocin 650mg",     "category": "Pain Relief"},
    "crocin":         {"alt": "Dolo 650mg or Paracetamol IP",     "category": "Pain Relief"},
    "calpol":         {"alt": "Dolo 650mg or Crocin 500mg",       "category": "Pain Relief"},
    "ibuprofen":      {"alt": "Brufen 400mg or Combiflam",        "category": "Pain Relief"},
    "combiflam":      {"alt": "Ibuprofen 400mg + Paracetamol",    "category": "Pain Relief"},
    "aspirin":        {"alt": "Ecosprin 75mg or Disprin",         "category": "Pain Relief"},
    "diclofenac":     {"alt": "Voveran 50mg or Voltaren",         "category": "Pain Relief"},

    # ── Antibiotics ──────────────────────────────────────────────────────────
    "azithromycin":   {"alt": "Azee 500mg or Zithromax 500mg",    "category": "Antibiotic"},
    "amoxicillin":    {"alt": "Mox 500mg or Novamox 500mg",       "category": "Antibiotic"},
    "amoxyclav":      {"alt": "Augmentin 625mg or Clavam 625mg",  "category": "Antibiotic"},
    "ciprofloxacin":  {"alt": "Ciplox 500mg or Cifran 500mg",     "category": "Antibiotic"},
    "metronidazole":  {"alt": "Flagyl 400mg or Metrogyl 400mg",   "category": "Antibiotic"},
    "doxycycline":    {"alt": "Doxylin 100mg or Microdox 100mg",  "category": "Antibiotic"},
    "cephalexin":     {"alt": "Sporidex 500mg or Phexin 500mg",   "category": "Antibiotic"},

    # ── Diabetes ─────────────────────────────────────────────────────────────
    "metformin":      {"alt": "Glycomet 500mg or Glucophage",     "category": "Diabetes"},
    "glimepiride":    {"alt": "Amaryl 2mg or Glimisave 2mg",      "category": "Diabetes"},
    "insulin":        {"alt": "Consult endocrinologist immediately","category": "Diabetes"},
    "januvia":        {"alt": "Sitagliptin 100mg — consult doctor","category": "Diabetes"},

    # ── Blood Pressure ───────────────────────────────────────────────────────
    "amlodipine":     {"alt": "Amlovas 5mg or Amcard 5mg",        "category": "Blood Pressure"},
    "telmisartan":    {"alt": "Telma 40mg or Telmikind 40mg",     "category": "Blood Pressure"},
    "losartan":       {"alt": "Losar 50mg or Cozaar 50mg",        "category": "Blood Pressure"},
    "atenolol":       {"alt": "Tenolol 50mg or Aten 50mg",        "category": "Blood Pressure"},
    "ramipril":       {"alt": "Cardace 5mg or Hopace 5mg",        "category": "Blood Pressure"},

    # ── Heart ────────────────────────────────────────────────────────────────
    "atorvastatin":   {"alt": "Atorva 10mg or Lipitor 10mg",      "category": "Heart"},
    "rosuvastatin":   {"alt": "Rosuvas 10mg or Crestor 10mg",     "category": "Heart"},
    "clopidogrel":    {"alt": "Clopivas 75mg or Plavix 75mg",     "category": "Heart"},
    "digoxin":        {"alt": "Lanoxin 0.25mg — consult cardiologist","category": "Heart"},

    # ── Vitamins ─────────────────────────────────────────────────────────────
    "vitamin c":      {"alt": "Celin 500mg or Limcee 500mg",      "category": "Vitamins"},
    "vitamin d":      {"alt": "Calcirol 60000IU or D-Rise 60000IU","category": "Vitamins"},
    "vitamin b12":    {"alt": "Methylcobalamin 500mcg or Cobadex", "category": "Vitamins"},
    "calcium":        {"alt": "Shelcal 500mg or Calcichew",        "category": "Vitamins"},
    "iron":           {"alt": "Feronia XT or Orofer XT",           "category": "Vitamins"},
    "folic acid":     {"alt": "Folvite 5mg or Folic Acid IP",      "category": "Vitamins"},

    # ── Cold & Flu ───────────────────────────────────────────────────────────
    "cetirizine":     {"alt": "Cetzine 10mg or Alerid 10mg",      "category": "Cold & Flu"},
    "loratadine":     {"alt": "Lorfast 10mg or Claritin 10mg",    "category": "Cold & Flu"},
    "montelukast":    {"alt": "Montair 10mg or Singulair 10mg",   "category": "Cold & Flu"},

    # ── Gastrointestinal ─────────────────────────────────────────────────────
    "omeprazole":     {"alt": "Omez 20mg or Prilosec 20mg",       "category": "GI"},
    "pantoprazole":   {"alt": "Pan 40mg or Pantocid 40mg",        "category": "GI"},
    "ranitidine":     {"alt": "Rantac 150mg or Aciloc 150mg",     "category": "GI"},
    "domperidone":    {"alt": "Domstal 10mg or Motilium 10mg",    "category": "GI"},
    "ondansetron":    {"alt": "Emeset 4mg or Zofran 4mg",         "category": "GI"},

    # ── Neurology ────────────────────────────────────────────────────────────
    "gabapentin":     {"alt": "Gabapin 300mg or Neurontin 300mg", "category": "Neurology"},
    "pregabalin":     {"alt": "Lyrica 75mg or Pregalin 75mg",     "category": "Neurology"},
    "levetiracetam":  {"alt": "Levera 500mg or Keppra 500mg",     "category": "Neurology"},

    # ── Asthma ───────────────────────────────────────────────────────────────
    "salbutamol":     {"alt": "Asthalin inhaler or Ventolin inhaler","category": "Asthma"},
    "budesonide":     {"alt": "Budecort inhaler or Pulmicort",    "category": "Asthma"},

    # ── Thyroid ──────────────────────────────────────────────────────────────
    "levothyroxine":  {"alt": "Thyronorm 50mcg or Eltroxin 50mcg","category": "Thyroid"},

    # ── Skin ─────────────────────────────────────────────────────────────────
    "betamethasone":  {"alt": "Betnovate cream or Diprovate cream","category": "Skin"},
    "clotrimazole":   {"alt": "Canesten cream or Candid cream",   "category": "Skin"},

    # ── Eye ──────────────────────────────────────────────────────────────────
    "moxifloxacin":   {"alt": "Moxicip eye drops or Vigamox",     "category": "Eye"},
    "ciprofloxacin eye": {"alt": "Ciplox-D eye drops",            "category": "Eye"},
}

# ── Default alternative by category ──────────────────────────────────────────
CATEGORY_DEFAULTS = {
    "Pain Relief":    "Paracetamol IP 500mg (available at any pharmacy)",
    "Antibiotic":     "Consult a doctor for an alternative antibiotic prescription",
    "Diabetes":       "Consult your endocrinologist for a verified alternative",
    "Blood Pressure": "Consult your cardiologist for a verified alternative",
    "Heart":          "Consult your cardiologist immediately",
    "Vitamins":       "Standard multivitamin supplement from a licensed pharmacy",
    "Cold & Flu":     "Cetirizine 10mg or consult a pharmacist",
    "GI":             "Omeprazole 20mg or consult a gastroenterologist",
    "Neurology":      "Consult your neurologist for a verified alternative",
    "Asthma":         "Consult your pulmonologist for a verified inhaler",
    "Thyroid":        "Consult your endocrinologist — never switch thyroid medicine without guidance",
    "Skin":           "Consult a dermatologist for a verified alternative",
    "Eye":            "Consult an ophthalmologist for a verified eye drop",
}

# ── Universal fallback ────────────────────────────────────────────────────────
UNIVERSAL_FALLBACK = (
    "Purchase a verified medicine from a licensed pharmacy. "
    "Ask your pharmacist for the same active ingredient from a trusted brand."
)


# ─────────────────────────────────────────────────────────────────────────────
# LOOKUP FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    return str(name or "").strip().lower()


def get_alternative_medicine(medicine_name: str) -> str:
    """
    Always returns an alternative medicine string.
    Never returns None.

    Strategy:
    1. Exact match in MEDICINE_DB
    2. Partial match (name contains key or key contains name)
    3. Category default
    4. Universal fallback
    """
    name = _normalize(medicine_name)

    if not name or name in ("not detected", "medicine", "n/a", ""):
        return UNIVERSAL_FALLBACK

    # ── 1. Exact match ────────────────────────────────────────────────────
    if name in MEDICINE_DB:
        return MEDICINE_DB[name]["alt"]

    # ── 2. Partial match ──────────────────────────────────────────────────
    for key, data in MEDICINE_DB.items():
        if key in name or name in key:
            return data["alt"]

    # ── 3. Word-level partial match ───────────────────────────────────────
    name_words = set(name.split())
    for key, data in MEDICINE_DB.items():
        key_words = set(key.split())
        if name_words & key_words:  # any common word
            return data["alt"]

    # ── 4. Universal fallback ─────────────────────────────────────────────
    return UNIVERSAL_FALLBACK


def get_category(medicine_name: str) -> str:
    """Get medicine category from name."""
    name = _normalize(medicine_name)
    for key, data in MEDICINE_DB.items():
        if key in name or name in key:
            return data.get("category", "General")
    return "General"


def generate_pharmacy_url(medicine_name: str = "medicine") -> str:
    """Generate 1mg.com search URL."""
    medicine = str(medicine_name or "medicine").strip()
    if medicine.lower() in ("not detected", "medicine", "n/a", ""):
        medicine = "medicine"
    query = urllib.parse.quote_plus(medicine)
    return f"https://www.1mg.com/search/all?name={query}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendation(
    prediction,
    ai_confidence=0,
    authenticity_score=0,
    expiry_status="Unknown",
    ocr_confidence=0,
    packaging_score=0,
    medicine_name="Medicine"
) -> dict:
    """
    Returns a rich recommendation dict.
    alternative_medicine is ALWAYS populated for unsafe predictions.
    """

    medicine_name = str(medicine_name or "Medicine").strip()
    display_name  = medicine_name if medicine_name.lower() not in (
        "medicine", "not detected", "n/a", "") else "This medicine"

    pharmacy_url = generate_pharmacy_url(medicine_name)

    # ── EXPIRED ───────────────────────────────────────────────────────────
    if prediction == "Expired" or expiry_status == "Expired":

        alt = get_alternative_medicine(medicine_name)

        return {
            "type":    "danger",
            "title":   "⏰ Do Not Use — Medicine Has Expired",
            "message": (
                f"{display_name} is EXPIRED. "
                "Expired medicines lose effectiveness and can be harmful. "
                "Do not consume under any circumstances."
            ),
            "actions": [
                "Do not consume this expired medicine.",
                "Check the expiry date printed on the package.",
                "Dispose of the medicine safely in a bin — do not flush.",
                "Purchase a fresh supply from a licensed pharmacy.",
                "Consult your doctor or pharmacist for guidance.",
            ],
            "alternative_medicine": alt,
            "show_pharmacy_button": True,
            "pharmacy_url":        pharmacy_url,
            "pharmacy_button_text": "Find Fresh Medicine on 1mg",
        }

    # ── COUNTERFEIT ───────────────────────────────────────────────────────
    if prediction == "Counterfeit":

        alt = get_alternative_medicine(medicine_name)

        return {
            "type":    "danger",
            "title":   "🚨 Possible Counterfeit Medicine Detected",
            "message": (
                f"{display_name} shows strong counterfeit indicators "
                f"(AI confidence: {ai_confidence:.0f}%, "
                f"Authenticity score: {authenticity_score:.0f}%). "
                "This medicine may be unsafe to consume."
            ),
            "actions": [
                "Do not consume this medicine.",
                "Keep the medicine and original packaging as evidence.",
                "Note the batch number and manufacturer details.",
                "Report to your nearest pharmacist or health authority.",
                "Purchase a verified replacement from a licensed pharmacy.",
            ],
            "alternative_medicine": alt,
            "show_pharmacy_button": True,
            "pharmacy_url":        pharmacy_url,
            "pharmacy_button_text": "Find Verified Medicine on 1mg",
        }

    # ── SUSPICIOUS ────────────────────────────────────────────────────────
    if prediction == "Suspicious":

        alt = get_alternative_medicine(medicine_name)

        return {
            "type":    "warning",
            "title":   "⚠️ Medicine Requires Verification",
            "message": (
                f"The authenticity of {display_name} could not be "
                f"fully confirmed (Authenticity score: {authenticity_score:.0f}%). "
                "Please verify before consuming."
            ),
            "actions": [
                "Do not consume until verified by a pharmacist.",
                "Upload a clearer, well-lit image for better analysis.",
                "Check the batch number and expiry date manually.",
                "Purchase from a licensed pharmacy if in doubt.",
                "Consult a licensed pharmacist or doctor.",
            ],
            "alternative_medicine": alt,
            "show_pharmacy_button": True,
            "pharmacy_url":        pharmacy_url,
            "pharmacy_button_text": "Find Verified Medicine on 1mg",
        }

    # ── GENUINE ───────────────────────────────────────────────────────────
    if prediction == "Genuine":

        return {
            "type":    "safe",
            "title":   "✅ Medicine Appears Genuine",
            "message": (
                f"{display_name} appears authentic "
                f"(AI confidence: {ai_confidence:.0f}%, "
                f"Authenticity score: {authenticity_score:.0f}%). "
                "Safe to use as prescribed."
            ),
            "actions": [
                "Verify the expiry date before consuming.",
                "Check the batch number matches the outer packaging.",
                "Always purchase medicines from licensed pharmacies.",
                "Store as directed on the label.",
                "Use only as prescribed by your physician.",
            ],
            "alternative_medicine": None,
            "show_pharmacy_button": False,
            "pharmacy_url":        None,
            "pharmacy_button_text": None,
        }

    # ── DEFAULT / UNKNOWN ─────────────────────────────────────────────────
    alt = get_alternative_medicine(medicine_name)

    return {
        "type":    "warning",
        "title":   "⚠️ Verification Recommended",
        "message": (
            f"{display_name} could not be confidently classified. "
            "Please verify manually before use."
        ),
        "actions": [
            "Upload a clearer image with better lighting.",
            "Ensure the medicine label is fully visible.",
            "Check batch number and expiry date manually.",
            "Consult a licensed pharmacist.",
        ],
        "alternative_medicine": alt,
        "show_pharmacy_button": True,
        "pharmacy_url":        pharmacy_url,
        "pharmacy_button_text": "Find Medicine on 1mg",
    }