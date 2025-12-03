# trocr_column_clustering.py
"""
Column + vertical block clustering for prescription lines (medicine names only).
- EasyOCR: detect word boxes
- Cluster boxes into columns by X-center
- Within each column split into vertical blocks by large Y gaps
- For each block crop: try rotated variants and run TrOCR (handwritten)
- Choose the best TrOCR result per block and print them
- Saves debug image trocr_column_blocks_debug.png
"""

import os
import re
import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ------------- CONFIG -------------
IMAGE_PATH = "test shot.png"  # change to your file
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # keep for fallback if you want
pytesseract.pytesseract.tesseract_cmd = TESS_PATH

TROCR_MODEL = "microsoft/trocr-base-handwritten"
GEN_KWARGS = dict(max_length=64, num_beams=4, early_stopping=True)

# thresholds (tweak if needed)
X_CLUSTER_THRESH_RATIO = 0.18   # column grouping threshold relative to image width (0.12..0.25)
MIN_BLOCK_GAP_FACTOR = 1.6      # split vertical blocks when gap > median_gap * factor
PADDING_RATIO = 0.26            # crop padding ratio (how much extra around block bbox)
UPSCALE_IF_SMALL = True
UPSCALE_MIN_H = 70
UPSCALE_SCALE = 3

USE_URDU = False  # set True if text contains Urdu script

# ------------- device & models -------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

languages = ['en']
if USE_URDU:
    languages = ['en', 'ur']
reader = easyocr.Reader(languages, gpu=(device == "cuda"))

print("Loading TrOCR model (this will download on first run)...")
processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
trocr_model.to(device)

# ------------- helpers -------------
def clean_text_basic(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace("|", "i").replace("—", "-").replace("\n", " ")
    s = s.translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))
    s = re.sub(r"[^A-Za-z0-9\u0600-\u06FF\+\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def trocr_recognize(crop_bgr):
    """Return trOCR cleaned text for a BGR crop (tries to run inference)."""
    try:
        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        # convert to PIL RGB
        if len(crop_bgr.shape) == 2:
            pil = Image.fromarray(crop_bgr).convert("RGB")
        else:
            pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil, return_tensors="pt").pixel_values.to(device)
        generated_ids = trocr_model.generate(pixel_values, **GEN_KWARGS)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return clean_text_basic(decoded)
    except Exception as e:
        # optional: print("TrOCR error:", e)
        return ""

def auto_rotate_variants(crop):
    """Return list of variants [orig, rot90cw, rot90ccw]."""
    variants = [crop]
    try:
        rot_cw = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        rot_ccw = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        variants.extend([rot_cw, rot_ccw])
    except Exception:
        pass
    return variants

def choose_best_candidate(candidates):
    """Heuristic pick: prefer alphabetic-rich longer strings."""
    best = ""
    best_score = -999
    for s in candidates:
        if not s:
            continue
        alpha = sum(ch.isalpha() for ch in s)
        digits = sum(ch.isdigit() for ch in s)
        score = alpha * 3 + digits * 1 + min(10, len(s))
        if score > best_score:
            best_score = score
            best = s
    return best

# ------------- clustering logic -------------
def cluster_columns_by_x(raw_boxes, img_w):
    """
    Simple incremental clustering by x_centers:
    raw_boxes: list of (bbox, text, prob)
    returns dict: column_id -> list of (x_center, y_center, bbox, text)
    """
    x_thresh = max(20, int(img_w * X_CLUSTER_THRESH_RATIO))
    columns = []  # each column: {'mean_x':..., 'items':[...]}
    for (bbox, text, prob) in raw_boxes:
        xs = [pt[0] for pt in bbox]; ys = [pt[1] for pt in bbox]
        x_center = sum(xs) / 4.0
        y_center = sum(ys) / 4.0
        placed = False
        for col in columns:
            if abs(x_center - col['mean_x']) <= x_thresh:
                col['items'].append((x_center, y_center, bbox, text))
                col['mean_x'] = (col['mean_x'] * (len(col['items']) - 1) + x_center) / len(col['items'])
                placed = True
                break
        if not placed:
            columns.append({'mean_x': x_center, 'items': [(x_center, y_center, bbox, text)]})
    # convert to simpler mapping: list of columns with items
    columns_sorted = sorted(columns, key=lambda c: c['mean_x'])  # left to right
    return [c['items'] for c in columns_sorted]

def split_column_into_blocks(items, img_h):
    """
    items: list of (x_center, y_center, bbox, text) for a column, unsorted
    returns list of blocks (each block is a list of items)
    Splits by large vertical gaps between sorted y_centers.
    """
    # handle trivial cases
    if not items:
        return []
    if len(items) == 1:
        return [items[:] ]

    # sort by y_center (top -> bottom)
    items_sorted = sorted(items, key=lambda x: x[1])
    y_centers = [it[1] for it in items_sorted]

    # compute consecutive gaps (length = n-1)
    gaps = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]

    # median gap; fall back to img_h if gaps empty
    median_gap = float(np.median(gaps)) if gaps else float(img_h)
    gap_thresh = max(25, median_gap * MIN_BLOCK_GAP_FACTOR)

    blocks = []
    current_block = [items_sorted[0]]

    # iterate gaps and decide whether to break
    for idx, gap in enumerate(gaps):
        # next item index is idx+1
        next_item = items_sorted[idx+1]
        if gap > gap_thresh:
            # gap large -> finish current block
            blocks.append(current_block)
            current_block = [ next_item ]
        else:
            # same block -> append next item
            current_block.append(next_item)

    # append the last block (whatever remains)
    if current_block:
        blocks.append(current_block)

    return blocks


def block_bbox_from_items(block_items, img_w, img_h):
    """Return expanded bbox (x1,y1,x2,y2) covering items in block, with padding."""
    xs = []
    ys = []
    for (_, _, bbox, _) in block_items:
        for (x, y) in bbox:
            xs.append(int(x)); ys.append(int(y))
    if not xs:
        return (0,0,0,0)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    # add padding
    pad_w = int((x2 - x1) * PADDING_RATIO) + 8
    pad_h = int((y2 - y1) * PADDING_RATIO) + 6
    xa = max(0, x1 - pad_w)
    ya = max(0, y1 - pad_h)
    xb = min(img_w - 1, x2 + pad_w)
    yb = min(img_h - 1, y2 + pad_h)
    return (xa, ya, xb, yb)

# ------------- main extraction using column splits -------------
def extract_medicines_by_columns(image_path, show_debug=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    img_color = cv2.imread(image_path)
    img_h, img_w = img_color.shape[:2]

    # mild global preprocess for detection
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # detect word boxes using EasyOCR (detail=1)
    raw = reader.readtext(gray, detail=1, paragraph=False)

    # cluster into columns (left to right)
    columns = cluster_columns_by_x(raw, img_w)

    all_blocks = []  # list of dicts {col_idx, block_idx, bbox, items}
    for col_idx, col_items in enumerate(columns):
        blocks = split_column_into_blocks(col_items, img_h)
        for block_idx, block_items in enumerate(blocks):
            bbox = block_bbox_from_items(block_items, img_w, img_h)
            all_blocks.append({
                "col": col_idx,
                "block_idx": block_idx,
                "items": block_items,
                "bbox": bbox
            })

    # For debug visualization
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # OCR each block using rotated TrOCR variants & pick best candidate
    results = []
    for bi, block in enumerate(all_blocks):
        xa, ya, xb, yb = block['bbox']
        if xb - xa < 10 or yb - ya < 10:
            continue
        crop = img_color[ya:yb, xa:xb]

        # upscale small crops for better recognition
        h, w = crop.shape[:2]
        if UPSCALE_IF_SMALL and (h < UPSCALE_MIN_H or w < UPSCALE_MIN_H*2):
            crop = cv2.resize(crop, (w * UPSCALE_SCALE, h * UPSCALE_SCALE), interpolation=cv2.INTER_CUBIC)

        # variants (original + rotations)
        variants = auto_rotate_variants(crop)

        cand_texts = []
        for var in variants:
            txt = trocr_recognize(var)
            if txt:
                cand_texts.append(txt)

        # fallback to tesseract on variants if nothing from trocr
        if not cand_texts:
            for var in variants:
                gray_var = cv2.cvtColor(var, cv2.COLOR_BGR2GRAY) if len(var.shape) == 3 else var
                try:
                    pil = Image.fromarray(cv2.cvtColor(var, cv2.COLOR_BGR2RGB))
                    cfg = "--oem 1 --psm 7"
                    ttxt = pytesseract.image_to_string(pil, config=cfg)
                    ttxt = clean_text_basic(ttxt)
                    if ttxt:
                        cand_texts.append(ttxt)
                except Exception:
                    pass

        best = choose_best_candidate(cand_texts) if cand_texts else clean_text_basic(" ".join([it[3] for it in block['items']]))

        results.append({
            "col": block['col'],
            "block_idx": block['block_idx'],
            "bbox": block['bbox'],
            "ocr_candidates": cand_texts,
            "final": best
        })

        # debug overlay
        if show_debug:
            label = best if best else "?"
            cv2.rectangle(vis, (xa, ya), (xb, yb), (0,255,0), 1)
            cv2.putText(vis, label[:36], (xa, max(ya-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    if show_debug:
        debug_out = "trocr_column_blocks_debug.png"
        cv2.imwrite(debug_out, vis)
        print("[debug] wrote", debug_out)

    # sort results top->bottom by bbox y center (so printed order is natural)
    results_sorted = sorted(results, key=lambda r: (r['bbox'][1] + r['bbox'][3]) / 2.0)
    return results_sorted

# ------------- run -------------
if __name__ == "__main__":
    res = extract_medicines_by_columns(IMAGE_PATH, show_debug=True)
    print("\nDetected medicine blocks (top->bottom):")
    for i, r in enumerate(res, 1):
        print(f"{i}: col={r['col']} block={r['block_idx']} final='{r['final']}'  candidates={r['ocr_candidates']}")
    print("\nDone.")
