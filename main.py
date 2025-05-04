from paddleocr import PaddleOCR, draw_ocr
import cv2
import os

# === CONFIGURATION ===
img_path = 'images/12.jpeg'  # Put your image inside an 'images' folder
output_dir = 'output'
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # <-- Make sure this exists

# === Initialize OCR model ===
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Supports 'ch', 'en', 'french', etc.

# === Load image ===
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# === Run OCR ===
results = ocr.ocr(img_path, cls=True)

# === Print detected text ===
print("\n🔍 OCR Results:")
for line in results:
    for box, (text, confidence) in line:
        print(f'Text: {text} | Confidence: {confidence:.2f}')

# === Prepare boxes and text for drawing ===
boxes = [elements[0] for line in results for elements in line]
txts = [elements[1][0] for line in results for elements in line]
scores = [elements[1][1] for line in results for elements in line]

# === Draw OCR results ===
img_with_boxes = draw_ocr(img, boxes, txts, scores, font_path=font_path)

# === Save output ===
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, os.path.basename(img_path))
cv2.imwrite(output_path, img_with_boxes)

print(f'\n📸 Output image saved to: {output_path}')
