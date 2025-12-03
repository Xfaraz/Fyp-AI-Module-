import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re

IMAGE_PATH = "test shot.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)


def clean_text(t):
    t = t.lower()
    t = t.replace("|", "i")
    t = re.sub(r"[^a-z0-9\+\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def extract_handwriting_lines(image_path, debug=True):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove printed text by heavy threshold
    bin_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 11)

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    clean = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(clean)

    line_boxes = []
    H, W = gray.shape

    for i in range(1, num):
        x, y, w, h, area = stats[i]

        # Filter out printed text regions:
        if area < 120: continue
        if h < 18: continue
        if y < H*0.12: continue         # ignore header
        if y > H*0.90: continue         # ignore footer

        line_boxes.append((x, y, x+w, y+h))

    if debug:
        dbg = img.copy()
        for (x1, y1, x2, y2) in line_boxes:
            cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imwrite("debug_lines.png", dbg)
        print("Saved debug_lines.png")

    return sorted(line_boxes, key=lambda b: b[1])


def run_trocr_on_crop(crop):
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    pix = processor(images=pil, return_tensors="pt").pixel_values.to(device)
    out = model.generate(pix, max_length=64)
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return clean_text(text)


def main():
    boxes = extract_handwriting_lines(IMAGE_PATH, debug=True)
    img = cv2.imread(IMAGE_PATH)

    print("\nDetected handwritten lines:")
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = img[y1:y2, x1:x2]

        # upscale for accuracy
        crop = cv2.resize(crop, None, fx=2.8, fy=2.8, interpolation=cv2.INTER_CUBIC)

        text = run_trocr_on_crop(crop)
        print(f"{i}: {text}")

    print("\nDone.")


if __name__ == "__main__":
    main()
