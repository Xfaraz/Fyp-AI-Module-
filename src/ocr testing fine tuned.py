# pip install easyocr rapidfuzz

import easyocr
from rapidfuzz import fuzz

# Example medicine database (fetch from MySQL later)
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

# ✅ Hybrid similarity matcher (fixes the Softin vs Augmentin issue)
def smart_match(ocr_token, medicine_list):
    best_match = None
    best_score = -9999  # very low initial score

    for med in medicine_list:

        # multiple similarity measurements
        score_edit = fuzz.ratio(ocr_token, med.lower())
        score_partial = fuzz.partial_ratio(ocr_token, med.lower())
        score_token = fuzz.token_sort_ratio(ocr_token, med.lower())
        score_position = fuzz.WRatio(ocr_token, med.lower())

        # penalize long words if OCR word is short (fixes Augmentin issue)
        length_penalty = abs(len(med) - len(ocr_token)) * 2

        # hybrid score
        final_score = (0.35 * score_edit) + (0.25 * score_partial) + (0.25 * score_position) + (0.15 * score_token)
        final_score -= length_penalty

        if final_score > best_score:
            best_score = final_score
            best_match = med

    return best_match, best_score


# STEP 1: Run EasyOCR
reader = easyocr.Reader(['en'])
results = reader.readtext("Screenshot 2025-10-19 163433.png", detail=0)
raw_text = " ".join(results)  # join OCR words into a single string

print("\n📌 OCR OUTPUT:", raw_text)

# STEP 2: NLP - normalize text
tokens = raw_text.lower().split()

print("\n✅ Identified Medicines:")
for token in tokens:
    match, score = smart_match(token, medicine_list)

    if score > 50:  # threshold can be adjusted
        print(f"OCR Token: {token} -> Match: {match} ({round(score,1)}%)")
