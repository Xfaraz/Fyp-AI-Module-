# ...existing code...
import cv2
import numpy as np
import easyocr
from rapidfuzz import fuzz

medicine_list = [
    "Amoxicillin 500mg",
    "Paracetamol 500mg",
    "Ibuprofen 200mg",
    "Azithromycin 250mg",
    "Panadol Extra",
    "Augmentin 625mg",
    "Brufen",
    "risek",
    "softin",
    "ascard",
    "rigix",
    "Betnesol",
]

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # upscale a bit to help OCR
    img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE to boost local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # denoise + slight blur
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.medianBlur(gray, 3)
    # adaptive threshold to binarize (helps handwriting sometimes)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 12)
    # optional morphological opening to remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

def ngrams(tokens, n_max=3):
    out = []
    L = len(tokens)
    for n in range(1, n_max+1):
        for i in range(L - n + 1):
            out.append(" ".join(tokens[i:i+n]))
    return out

def smart_match(ocr_token, medicine_list):
    o = ocr_token.strip().lower()
    best_match = None
    best_score = -9999
    for med in medicine_list:
        m = med.lower()
        # multiple fuzzy measures
        score_edit = fuzz.ratio(o, m)
        score_partial = fuzz.partial_ratio(o, m)
        score_token = fuzz.token_sort_ratio(o, m)
        score_w = fuzz.WRatio(o, m)
        # softer length penalty
        length_penalty = abs(len(m) - len(o)) * 0.6
        final_score = (0.35*score_edit + 0.25*score_partial + 0.2*score_token + 0.2*score_w) - length_penalty
        if final_score > best_score:
            best_score = final_score
            best_match = med
    return best_match, best_score

# STEP 1: preprocess and run EasyOCR
img_proc = preprocess("WhatsApp Image 2025-11-11 at 6.51.05 PM.jpeg")
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if available
# tune thresholds if needed
results = reader.readtext(img_proc, detail=0, contrast_ths=0.1, adjust_contrast=1.5, text_threshold=0.4)
raw_text = " ".join(results)
print("\n📌 OCR OUTPUT:", raw_text)

# STEP 2: tokenization -> ngrams
tokens = [t.strip(".,;:()[]") for t in raw_text.lower().split()]
cand_ngrams = ngrams(tokens, n_max=3)

print("\n✅ Identified Medicines:")
seen = set()
for tok in cand_ngrams:
    match, score = smart_match(tok, medicine_list)
    if score > 55 and match not in seen:  # tuned threshold
        print(f"OCR token: '{tok}' -> Match: {match} ({round(score,1)}%)")
        seen.add(match)
# ...existing code...