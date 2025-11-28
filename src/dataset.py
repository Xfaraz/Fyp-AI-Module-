import cv2
import pytesseract
import os

# ✅ Set correct Tesseract path (update if your install path is different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\faraz\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# ✅ Use current directory (root) for input/output
root_dir = "."

for fname in os.listdir(root_dir):
    if not fname.endswith(".png"):
        continue

    img_path = os.path.join(root_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Threshold for line detection
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Step 3: Find contours of lines
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines_y = [cv2.boundingRect(c)[1] for c in contours]
    lines_y.sort()

    if len(lines_y) < 2:
        print(f"⚠ Could not detect lines in {fname}, skipping...")
        continue

    # Step 4: Crop regions
    top_line = lines_y[0]
    bottom_line = lines_y[-1]

    # Printed text region
    printed_crop = img[:top_line, :]

    # Handwritten region
    handwritten_crop = img[bottom_line+5:, :]

    # Step 5: OCR on printed region (to get labels)
    label_text = pytesseract.image_to_string(printed_crop, config="--psm 7").strip()

    # Step 6: Save results in root directory
    base_name = os.path.splitext(fname)[0]

    # Save handwriting crop as new image
    hand_image_path = os.path.join(root_dir, f"{base_name}_hand.png")
    cv2.imwrite(hand_image_path, handwritten_crop)

    # Save label text
    label_path = os.path.join(root_dir, f"{base_name}.txt")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(label_text)

    print(f"[✔] {fname} → {base_name}_hand.png + {base_name}.txt saved")
