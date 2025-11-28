# Install EasyOCR if not already installed
# pip install easyocr opencv-python

import easyocr
import cv2
import matplotlib.pyplot as plt

# Load EasyOCR Reader (English for now, can add 'ur' for Urdu if needed)
reader = easyocr.Reader(['en'])  

# Load an image (replace with your prescription image path)
image_path = "Screenshot 2025-10-19 163433.png"

# Read text from image
results = reader.readtext(image_path)

# Print detected text
print("Detected Handwritten Text:")
for (bbox, text, prob) in results:
    print(f"{text} (Confidence: {prob:.2f})")

# Display the image with bounding boxes
image = cv2.imread(image_path)

for (bbox, text, prob) in results:
    # Draw bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Show the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
