from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
import os


input_folder = 'images'
output_folder = 'output'
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
confidence_threshold = 0.7
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_score_mode='fast', layout=True)


def compute_iou(box1, box2):
    box1 = np.array(box1).reshape(-1, 2)
    box2 = np.array(box2).reshape(-1, 2)
    x_min1, y_min1 = np.min(box1, axis=0)
    x_max1, y_max1 = np.max(box1, axis=0)
    x_min2, y_min2 = np.min(box2, axis=0)
    x_max2, y_max2 = np.max(box2, axis=0)
    inter_xmin = max(x_min1, x_min2)
    inter_ymin = max(y_min1, y_min2)
    inter_xmax = min(x_max1, x_max2)
    inter_ymax = min(y_max1, y_max2)
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def is_similar(text1, text2, threshold=85):
    return fuzz.ratio(text1.strip().lower(), text2.strip().lower()) >= threshold

def preprocess_adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

def preprocess_morph_gamma(img):
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(morph, table)

def update_results(new_results, existing_results):
    if not new_results:
        return existing_results
    for item in new_results:
        box, (text, score) = item
        if score < confidence_threshold or not text.strip():
            continue
        matched = False
        for i, (ex_box, (ex_text, ex_score)) in enumerate(existing_results):
            if is_similar(text, ex_text) and compute_iou(box, ex_box) > 0.5:
                matched = True
                if score > ex_score:
                    existing_results[i] = (box, (text, score))
                break
        if not matched:
            existing_results.append((box, (text, score)))
    return existing_results

# main
os.makedirs(output_folder, exist_ok=True)

for i in range(1, 13):
    filename = f"{i}.jpeg"
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Skipping missing image: {img_path}")
        continue

    print(f"\nProcessing: {filename}")
    results = []

    def safe_ocr_run(img, description):
        print(f"🔹 OCR Layer: {description}")
        try:
            ocr_result = ocr.ocr(img, cls=True)
            return ocr_result[0] if ocr_result else []
        except Exception as e:
            print(f"OCR failed on {description}: {e}")
            return []

    # l1 Raw
    results = update_results(safe_ocr_run(image, "Raw Image"), results)

    # l2 Adaptive
    pre2 = preprocess_adaptive(image)
    results = update_results(safe_ocr_run(pre2, "Adaptive Threshold"), results)

    # l3 Morph + Gamma
    pre3 = preprocess_morph_gamma(pre2)
    results = update_results(safe_ocr_run(pre3, "Morph + Gamma"), results)

    print("Final deduplicated OCR results:")
    for box, (text, score) in results:
        print(f'Text: {text} | Confidence: {score:.2f}')

    # draw bb n res
    if results:
        boxes = [b for b, _ in results]
        txts = [t[0] for _, t in results]
        scores = [t[1] for _, t in results]
        image_with_boxes = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image_with_boxes)
        print(f"Output saved to: {output_path}")

        # vis
        img_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"OCR Output: {filename}")
        plt.show()
    else:
        print("🫥 No valid OCR results found.")
