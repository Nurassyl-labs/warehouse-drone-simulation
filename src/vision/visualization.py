import cv2

def draw_bounding_box(img, x, y, w, h, label=None, color=(34, 126, 230), thickness=2):
    """Draws a standard 2D bounding box with label outline on an image."""
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img

def draw_hud_text(img, text, position=(15, 30), color=(46, 204, 113), scale=0.6, thickness=2):
    """Draws HUD text label on a given frame."""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return img
