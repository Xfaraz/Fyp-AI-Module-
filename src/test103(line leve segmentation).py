# improved_line_ocr.py
import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
import re
import os

# ----- config -----
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESS_PATH

# EasyOCR reader (create once globally)
reader = easyocr.Reader(['en'], gpu=False)     # add 'ur' if you expect Urdu text

# ----- helpers -----
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def deskew(img_gray):
    # compute angle via moments of edges
    coords = np.column_stack(np.where(img_gray < 255))
    if coords.size == 0:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def apply_clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(img_gray)

def sharpen(img_gray):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img_gray, -1, kernel)

def upscale(img, scale=2):
    h,w = img.shape[:2]
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def clean_text_basic(s):
    # lowercase
    s = s.lower()
    # map similar characters commonly misrecognized
    subs = {
        '0': 'o',   # sometimes digits mistaken for letters; do selective mapping later
        '1': 'i', 
        'l': 'i',
        '—': '-',
        '|': 'i',
        '‚': ',',
        '“': '"',
        '”': '"',
        '_': '',
        '~': '',
    }
    for a,b in subs.items():
        s = s.replace(a,b)
    # remove weird repeating punctuation
    s = re.sub(r'[^a-z0-9\+\-\s\u0600-\u06FF]', ' ', s)  # allow Urdu block if needed
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def urdu_digits_to_latin(s):
    # map eastern arabic numerals to latin
    mapping = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
    return s.translate(mapping)

def tesseract_on_image(img_gray, psm=6, oem=1, lang='eng'):
    # receive grayscale or PIL; returns text
    if isinstance(img_gray, np.ndarray):
        pil = Image.fromarray(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB))
    else:
        pil = img_gray
    config = f'--oem {oem} --psm {psm}'
    try:
        txt = pytesseract.image_to_string(pil, lang=lang, config=config)
    except Exception as e:
        txt = ""
    return txt

# ----- main improved extraction -----
def preprocess_full(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = to_gray(img)

    # deskew + denoise
    gray = deskew(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # contrast + CLAHE
    gray = apply_clahe(gray)
    gray = sharpen(gray)

    # small morphological close to connect strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return gray, img

def cluster_boxes_to_lines(boxes, img_h, y_thresh_px=None):
    if y_thresh_px is None:
        y_thresh_px = max(10, int(img_h * 0.015))
    items = []
    for (bbox, text, prob) in boxes:
        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        y_center = sum(ys)/4.0
        x_center = sum(xs)/4.0
        items.append((y_center, x_center, text, bbox))
    items.sort(key=lambda x: x[0])
    lines = []
    current = []
    current_y = None
    for y_center, x_center, text, bbox in items:
        if not current:
            current = [(x_center, text, bbox, y_center)]
            current_y = y_center
            continue
        if abs(y_center - current_y) <= y_thresh_px:
            current.append((x_center, text, bbox, y_center))
            current_y = (current_y * (len(current)-1) + y_center) / len(current)
        else:
            lines.append(current)
            current = [(x_center, text, bbox, y_center)]
            current_y = y_center
    if current:
        lines.append(current)
    # build joined lines and ROI bbox
    joined_lines = []
    for line in lines:
        line.sort(key=lambda x: x[0])
        words = [w for (_, w, _, _) in line]
        joined = " ".join(words)
        # compute bounding roi for the line
        xs = []
        ys = []
        for (_,_,bbox,_) in line:
            for (x,y) in bbox:
                xs.append(int(x)); ys.append(int(y))
        x1,x2 = min(xs), max(xs)
        y1,y2 = min(ys), max(ys)
        joined_lines.append((joined, (x1, y1, x2, y2)))
    return joined_lines

def extract_improved_lines(img_path, show_debug=False):
    gray, orig_color = preprocess_full(img_path)

    # run EasyOCR on the preprocessed grayscale (it accepts numpy arrays)
    raw = reader.readtext(gray, detail=1, paragraph=False)

    img_h = gray.shape[0]
    lines_with_bbox = cluster_boxes_to_lines(raw, img_h)

    final_lines = []
    for text, (x1,y1,x2,y2) in lines_with_bbox:
        # expand bbox a bit
        pad = 6
        h,w = gray.shape
        xa,ya = max(0,x1-pad), max(0,y1-pad)
        xb,yb = min(w-1,x2+pad), min(h-1,y2+pad)
        crop = gray[ya:yb, xa:xb]

        # upscale crop to help OCR
        crop_up = upscale(crop, scale=3)

        # run both OCRs on the crop_up
        e_txt = ""
        try:
            e_res = reader.readtext(crop_up, detail=0, paragraph=False)
            e_txt = " ".join(e_res)
        except Exception:
            e_txt = ""

        t_txt = tesseract_on_image(crop_up, psm=6, oem=1, lang='eng')

        # merge preferring easyocr but fall back to tesseract words
        merged_raw = (e_txt + " " + t_txt).strip()

        # normalize: urdu digits + char fixes + remove junk
        merged_raw = urdu_digits_to_latin(merged_raw)
        merged_raw = clean_text_basic(merged_raw)

        final_lines.append({
            "raw": merged_raw,
            "bbox": (xa,ya,xb,yb),
            "ocr_easy": e_txt,
            "ocr_tess": t_txt
        })

    # debug visualization
    if show_debug:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for item in final_lines:
            xa,ya,xb,yb = item['bbox']
            cv2.rectangle(vis, (xa,ya), (xb,yb), (0,255,0), 1)
            txt = item['raw'][:40]
            cv2.putText(vis, txt, (xa, max(ya-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.imwrite("improved_ocr_debug.png", vis)

    return final_lines

# ---------- quick test ----------
if __name__ == "__main__":
    image_path = "WhatsApp Image 2025-11-11 at 6.51.05 PM.jpeg"
    if not os.path.exists(image_path):
        raise SystemExit("image not found")
    lines = extract_improved_lines(image_path, show_debug=True)
    print("Detected lines (cleaned):")
    for i, L in enumerate(lines,1):
        print(i, "=>", L['raw'])
