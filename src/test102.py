# -------------------------------------------
# ✅ Hybrid OCR + NLP + Semantic Matcher
# -------------------------------------------

import cv2
import pytesseract
import easyocr
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
from PIL import Image

# -------------------------------------------
# ✅ LOAD SEMANTIC MODEL (for meaning matching)
# -------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")   # small + fast + accurate

# -------------------------------------------
# ✅ Medicine list (you’ll query MySQL later)
# -------------------------------------------
medicine_list = [
    "Amoxicillin",
    "Paracetamol",
    "Ibuprofen",
    "Azithromycin",
    "Panadol Extra",
    "Augmentin",
    "Brufen",
    "Risek",
    "Softin",
    "Ascard",
    "Rigix",
    "Betnesol",
]

# -------------------------------------------
# ✅ Image Preprocessing (improve OCR accuracy)
# -------------------------------------------
def preprocess(img_path):
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoising + sharpening improves OCR massively
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=0)

    # Adaptive thresholding (clean background)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )

    return thresh


# -------------------------------------------
# ✅ OCR (EasyOCR + Tesseract COMBINED)
# -------------------------------------------
def extract_text(img_path):
    processed = preprocess(img_path)

    # ★ EasyOCR
    reader = easyocr.Reader(['en'])
    easy_text = reader.readtext(processed, detail=0)

    # ★ Tesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    tess_text = pytesseract.image_to_string(processed)

    # merge and remove duplicates
    tokens = set(" ".join(easy_text).lower().split() + tess_text.lower().split())
    return tokens


# -------------------------------------------
# ✅ Hybrid similarity matcher (Fuzzy + Semantic + Length score)
# -------------------------------------------
def smart_match(ocr_token, medicine_list):
    best_match = None
    best_score = -9999

    for med in medicine_list:
        med_clean = med.lower()

        # Fuzzy / spelling based similarity (edit distance)
        fuzzy_score = fuzz.ratio(ocr_token, med_clean)

        # Token + partial matches (helps with "panadol extra", "amox 500", etc.)
        partial = fuzz.partial_ratio(ocr_token, med_clean)
        token_sort = fuzz.token_sort_ratio(ocr_token, med_clean)

        # Semantic meaning score (like ChatGPT reasoning)
        emb_score = util.cos_sim(
            model.encode(ocr_token),
            model.encode(med_clean)
        ).item() * 100  # convert to percentage

        # Penalize for big length difference (fixes Softin vs Augmentin issue)
        length_penalty = abs(len(med_clean) - len(ocr_token)) * 2

        # final weighted score
        final_score = (0.30 * fuzzy_score) + (0.20 * partial) + (0.20 * token_sort) + (0.30 * emb_score)
        final_score -= length_penalty

        if final_score > best_score:
            best_score = final_score
            best_match = med

    return best_match, best_score


# -------------------------------------------
# ✅ MAIN
# -------------------------------------------
image_path = "WhatsApp Image 2025-11-11 at 6.51.05 PM.jpeg"  # change filename here

tokens = extract_text(image_path)

print("\n📌 OCR Tokens Detected:", tokens)

print("\n✅ Identified Medicines:")
for token in tokens:
    match, score = smart_match(token, medicine_list)

    if score > 50:  # threshold
        print(f"OCR Token: {token} ➝ Match: {match} ({round(score,1)}%)")
