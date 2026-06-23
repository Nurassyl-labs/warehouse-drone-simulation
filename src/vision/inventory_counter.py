import cv2
import numpy as np

def detect_and_count_boxes(img, return_boxes=False):
    """
    Detects cardboard brown boxes in the camera image using OpenCV HSV thresholding.
    Returns:
    - box_count: number of detected boxes (int)
    - occupancy: ratio of occupied shelf space (float)
    - annotated_img: image with bounding box overlays (numpy array)
    - detected_boxes (optional): list of tuples (x, y, w, h, estimated_count)
    """
    if img is None:
        if return_boxes:
            return 0, 0.0, None, []
        return 0, 0.0, None

    annotated_img = img.copy()

    # 1. Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Define range for cardboard brown color in HSV
    lower_brown = np.array([0, 15, 15], dtype=np.uint8)
    upper_brown = np.array([35, 255, 255], dtype=np.uint8)

    # 3. Create mask
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # 4. Perform morphological opening to remove small noise, and closing to merge tape/label details inside a box
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)

    # 5. Find contours
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_count = 0
    detected_boxes = []

    # 6. Filter contours by area and aspect ratio to ensure they are boxes
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # Minimum pixel size of a box
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            # Aspect ratio check
            if 0.3 < aspect_ratio < 4.0:
                # Estimate number of boxes merged in this contour
                estimated_count = max(1, int(np.round(aspect_ratio)))
                box_count += estimated_count
                detected_boxes.append((x, y, w, h, estimated_count))
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (46, 204, 113), 2)
                # Label
                cv2.putText(annotated_img, f"BOX x{estimated_count}", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Total shelf slots is 9 (3 levels x 3 columns)
    occupancy = min(1.0, float(box_count) / 9.0)

    # Draw count HUD on the frame
    cv2.putText(annotated_img, f"Detected Boxes: {box_count} / 9 (Occupancy: {occupancy*100:.1f}%)", 
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 204, 113), 2)

    if return_boxes:
        return box_count, occupancy, annotated_img, detected_boxes
    return box_count, occupancy, annotated_img
