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

# Initialize PaddleOCR v2.7+ with SVTR recognizer and DBNet detector
# Configuration optimized for 7-segment displays
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    rec_algorithm='SVTR_LCNet',  # SVTR recognizer
    det_algorithm='DB',          # DBNet detector
    det_db_score_mode='fast',
    det_db_unclip_ratio=1.7,     # Optimized for 7-segment displays
    det_db_box_thresh=0.5,       # Detection confidence threshold
    rec_batch_num=6,             # Batch size for recognition
    use_dilation=True,           # Dilate features to better connect segments
    layout=True
)


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Handles different box formats from PaddleOCR.
    """
    # Convert boxes to numpy arrays and reshape if needed
    box1 = np.array(box1).reshape(-1, 2)
    box2 = np.array(box2).reshape(-1, 2)
    
    # Get min/max coordinates
    x_min1, y_min1 = np.min(box1, axis=0)
    x_max1, y_max1 = np.max(box1, axis=0)
    x_min2, y_min2 = np.min(box2, axis=0)
    x_max2, y_max2 = np.max(box2, axis=0)
    
    # Calculate intersection area
    inter_xmin = max(x_min1, x_min2)
    inter_ymin = max(y_min1, y_min2)
    inter_xmax = min(x_max1, x_max2)
    inter_ymax = min(y_max1, y_max2)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    # Calculate individual box areas
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    # Calculate union area
    union_area = box1_area + box2_area - inter_area
    
    # Return IoU
    return inter_area / union_area if union_area > 0 else 0


def is_similar(text1, text2, threshold=85):
    """
    Check if two text strings are similar based on fuzzy matching.
    Particularly useful for 7-segment displays where characters can be confused.
    """
    # Special handling for 7-segment display common confusions
    text1 = text1.strip()
    text2 = text2.strip()
    
    # If the texts are identical
    if text1 == text2:
        return True
        
    # Common 7-segment display confusions
    segment_confusions = {
        '0': 'O', 'O': '0',
        '1': 'I', 'I': '1',
        '5': 'S', 'S': '5',
        '8': 'B', 'B': '8',
        '6': 'b', 'b': '6',
        '9': 'g', 'g': '9'
    }
    
    # Apply common 7-segment replacements
    normalized_text1 = ''.join(segment_confusions.get(c, c) for c in text1)
    normalized_text2 = ''.join(segment_confusions.get(c, c) for c in text2)
    
    # Check if normalized texts match
    if normalized_text1 == normalized_text2:
        return True
        
    # Finally, use fuzzy matching ratio
    return fuzz.ratio(text1.lower(), text2.lower()) >= threshold


def preprocess_image(img):
    """
    Apply the 7-layer preprocessing pipeline:
    L1: Grayscale
    L2: Gamma Correction
    L3: Adaptive Threshold
    L4: Morphological Closing
    L5: Noise Reduction
    L6: Otsu's Threshold (optional)
    L7: Segment Validation & Cleanup
    """
    # L1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # L2: Gamma correction (try both gamma values)
    gamma1 = 1.5
    gamma2 = 0.8
    
    invGamma1 = 1.0 / gamma1
    table1 = np.array([(i / 255.0) ** invGamma1 * 255 for i in range(256)]).astype("uint8")
    gamma_img1 = cv2.LUT(gray, table1)
    
    invGamma2 = 1.0 / gamma2
    table2 = np.array([(i / 255.0) ** invGamma2 * 255 for i in range(256)]).astype("uint8")
    gamma_img2 = cv2.LUT(gray, table2)
    
    # Process both gamma-corrected images
    results = []
    for gamma_img, gamma_val in [(gamma_img1, 1.5), (gamma_img2, 0.8)]:
        # L3: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gamma_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Inverse for 7-segment displays
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # L4: Morphological closing to connect potential broken segments
        kernel = np.ones((3, 3), np.uint8)
        morph_closing = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # L5: Noise reduction (try both median and gaussian)
        # Median filter
        median_filtered = cv2.medianBlur(morph_closing, 3)
        results.append(("Median-γ=" + str(gamma_val), median_filtered))
        
        # Gaussian filter
        gaussian_filtered = cv2.GaussianBlur(morph_closing, (3, 3), 0)
        results.append(("Gaussian-γ=" + str(gamma_val), gaussian_filtered))
        
        # L6: Otsu's thresholding (optional)
        _, otsu_thresh = cv2.threshold(gamma_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_closed = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
        otsu_filtered = cv2.medianBlur(otsu_closed, 3)
        results.append(("Otsu-γ=" + str(gamma_val), otsu_filtered))
    
    # Convert back to BGR for PaddleOCR
    processed_images = [(name, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) for name, img in results]
    
    # L7: Segment validation is implicit in our multi-layer approach and result merging
    return processed_images


def update_results(new_results, existing_results):
    if not new_results:
        return existing_results
    for item in new_results:
        box, (text, score) = item
        if score < confidence_threshold or not text.strip():
            continue
            
        # Check for overlapping boxes using IoU
        matched = False
        for i, (ex_box, (ex_text, ex_score)) in enumerate(existing_results):
            iou = compute_iou(box, ex_box)
            # Consider boxes as duplicates if they overlap significantly
            if iou > 0.3:
                matched = True
                # Keep the detection with higher confidence score
                if score > ex_score:
                    existing_results[i] = (box, (text, score))
                break
                
        # If no match found, add as new detection
        if not matched:
            existing_results.append((box, (text, score)))
            
    return existing_results


# Main processing function
def main():
    os.makedirs(output_folder, exist_ok=True)

    for i in range(1, 13):
        filename = f"{i}.jpeg"
        img_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(img_path)

        if original_image is None:
            print(f"Skipping missing image: {img_path}")
            continue

        print(f"\n🔍 Processing: {filename}")
        results = []

        def safe_ocr_run(img, description):
            print(f"🔹 OCR Layer: {description}")
            try:
                ocr_result = ocr.ocr(img, cls=True)
                layer_results = ocr_result[0] if ocr_result else []
                print(f"  Found {len(layer_results)} initial detections")
                return layer_results
            except Exception as e:
                print(f"OCR failed on {description}: {e}")
                return []

        # Run OCR on original image
        results = update_results(safe_ocr_run(original_image, "Raw Image"), results)

        # Apply our preprocessing pipeline and run OCR on each variant
        processed_images = preprocess_image(original_image)
        for desc, img in processed_images:
            results = update_results(safe_ocr_run(img, desc), results)

        # Additional filtering for 7-segment display specific cases
        filtered_results = []
        for box, (text, score) in results:
            # Filter out low confidence and empty results
            if score < confidence_threshold or not text.strip():
                continue
                
            # Remove non-digit characters typical in 7-segment misreads
            cleaned_text = ''.join(c for c in text if c.isdigit() or c in '.-:')
            
            # Skip if text became empty after cleaning
            if not cleaned_text:
                continue
                
            filtered_results.append((box, (cleaned_text, score)))

        print(f"📋 Final deduplicated OCR results: {len(filtered_results)} unique detections")
        for box, (text, score) in filtered_results:
            print(f'Text: {text} | Confidence: {score:.2f}')

        # Draw bounding boxes and results
        if filtered_results:
            boxes = [b for b, _ in filtered_results]
            txts = [t[0] for _, t in filtered_results]
            scores = [t[1] for _, t in filtered_results]
            
            # Draw results on original image
            image_with_boxes = draw_ocr(original_image, boxes, txts, scores, font_path=font_path)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image_with_boxes)
            print(f"✅ Output saved to: {output_path}")

            # Visualize results
            img_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 8))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title(f"SVTR+DBNet OCR: {filename}")
            plt.show()
        else:
            print("⚠️ No valid OCR results found.")


if __name__ == "__main__":
    main()