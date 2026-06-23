import os
import cv2
import numpy as np
import random
from src.utils.config import CONFIG

# Configure path imports
SHELF_Z = CONFIG["shelf"]["z"]
LEVELS = CONFIG["shelf"]["levels"]
COLUMNS = CONFIG["shelf"]["columns"]

STATIC_MARKERS = [
    {"id": m["id"], "x": m["x"], "y": m["y"], "z": SHELF_Z}
    for m in CONFIG["shelf"]["markers"]
]

# Cache dictionary for ArUco marker textures
_aruco_cache = {}

def get_real_aruco(marker_id, size=100):
    """Generates an ArUco marker image using OpenCV's ArUco dictionary, with local caching"""
    if marker_id in _aruco_cache:
        return _aruco_cache[marker_id]

    try:
        # OpenCV 4.7.x+
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    except AttributeError:
        # Fallback for older OpenCV
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        marker = cv2.aruco.drawMarker(dictionary, marker_id, size)

    _aruco_cache[marker_id] = marker
    return marker

def draw_cube_3d(img, rvec, tvec, K, cx, cy, cz, sx, sy, sz, color, border_color=(255, 255, 255), draw_details=False):
    """Projects and draws a shaded 3D box (cube) in perspective projection"""
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    pts_3d = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz]
    ], dtype=np.float32)

    # Project to 2D
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, np.zeros(4))
    pts_2d = np.int32(pts_2d.reshape(-1, 2))

    # Define 6 faces (order of drawing from back to front is important)
    faces = [
        [4, 5, 6, 7], # Back face
        [0, 1, 5, 4], # Top face
        [2, 3, 7, 6], # Bottom face
        [0, 3, 7, 4], # Left face
        [1, 2, 6, 5], # Right face
        [0, 1, 2, 3]  # Front face
    ]

    # Draw shaded faces to simulate simple lighting
    face_shading = [0.7, 0.9, 0.6, 0.8, 0.8, 1.0] # brightness multiplier
    for idx, face in enumerate(faces):
        poly = np.array([pts_2d[i] for i in face], dtype=np.int32)
        shading = face_shading[idx]
        face_color = (int(color[0]*shading), int(color[1]*shading), int(color[2]*shading))
        cv2.fillPoly(img, [poly], face_color)
        if border_color:
            cv2.polylines(img, [poly], True, border_color, 1)

    # Draw realistic shipping tape and label on the front face of cardboard boxes
    if draw_details:
        # Tape coordinates: vertical strip in the center
        tape_pts_3d = np.array([
            [cx - sx * 0.08, cy - hy, cz - hz - 0.001],
            [cx + sx * 0.08, cy - hy, cz - hz - 0.001],
            [cx + sx * 0.08, cy + hy, cz - hz - 0.001],
            [cx - sx * 0.08, cy + hy, cz - hz - 0.001]
        ], dtype=np.float32)
        pts_tape, _ = cv2.projectPoints(tape_pts_3d, rvec, tvec, K, np.zeros(4))
        pts_tape = np.int32(pts_tape.reshape(-1, 2))
        cv2.fillPoly(img, [pts_tape], (35, 50, 75)) # Dark brown tape

        # Shipping label sticker (white patch)
        # Seed consistently based on box center coordinates
        label_seed = int(abs(cx * 1000 + cy * 10)) % 10000
        np.random.seed(label_seed)
        lbl_x = np.random.uniform(-sx * 0.15, sx * 0.15)
        lbl_y = np.random.uniform(-sy * 0.1, sy * 0.1)

        label_pts_3d = np.array([
            [cx + lbl_x - sx * 0.20, cy + lbl_y - sy * 0.12, cz - hz - 0.002],
            [cx + lbl_x + sx * 0.20, cy + lbl_y - sy * 0.12, cz - hz - 0.002],
            [cx + lbl_x + sx * 0.20, cy + lbl_y + sy * 0.18, cz - hz - 0.002],
            [cx + lbl_x - sx * 0.20, cy + lbl_y + sy * 0.18, cz - hz - 0.002]
        ], dtype=np.float32)
        pts_label, _ = cv2.projectPoints(label_pts_3d, rvec, tvec, K, np.zeros(4))
        pts_label = np.int32(pts_label.reshape(-1, 2))
        cv2.fillPoly(img, [pts_label], (240, 240, 240)) # White label
        cv2.polylines(img, [pts_label], True, (180, 180, 180), 1)

def draw_marker_3d(img, rvec, tvec, K, marker_id, cx, cy, cz, size=0.20):
    """Perspective-warps an ArUco marker onto the 3D shelf scene"""
    hs = size / 2.0
    pts_3d = np.array([
        [cx - hs, cy - hs, cz],
        [cx + hs, cy - hs, cz],
        [cx + hs, cy + hs, cz],
        [cx - hs, cy + hs, cz]
    ], dtype=np.float32)

    # Project to 2D
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, np.zeros(4))
    pts_2d = pts_2d.reshape(-1, 2)

    # Check if in front of camera
    R, _ = cv2.Rodrigues(rvec)
    for pt in pts_3d:
        pt_c = R.dot(pt) + tvec.ravel()
        if pt_c[2] < 0.1:
            return

    # Generate marker texture image
    marker_img = get_real_aruco(marker_id, 100)

    # Homography warping
    src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, pts_2d)
    if H is not None:
        warped = cv2.warpPerspective(marker_img, H, (img.shape[1], img.shape[0]))
        mask = cv2.warpPerspective(np.ones_like(marker_img) * 255, H, (img.shape[1], img.shape[0]))
        
        warped_3ch = cv2.merge([warped, warped, warped])
        mask_3ch = cv2.merge([mask, mask, mask])
        img[mask_3ch > 0] = warped_3ch[mask_3ch > 0]

def render_warehouse_view(x_cam, y_cam, z_cam, yaw_cam, boxes_active, image_size=None):
    """Renders the 3D perspective projection of the warehouse shelf from camera pose"""
    if image_size is None:
        image_size = tuple(CONFIG["dataset"]["image_size"])
    w, h = image_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Industrial warehouse dark grayscale background (S=0, prevents HSV brown overlap)
    img[:, :] = (20, 20, 20)

    # Camera intrinsics K
    f = CONFIG["camera"]["focal_length"]
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)

    # Compute view matrix R and T
    yaw_rad = np.radians(yaw_cam)
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ], dtype=np.float32)
    
    R = R_yaw.T
    T = -R.dot(np.array([x_cam, y_cam, z_cam], dtype=np.float32))
    
    rvec, _ = cv2.Rodrigues(R)
    tvec = T.reshape(3, 1)

    # 1. Draw floor lines (stretching to Z+) with dynamic depth fading
    gl_lines = []
    for fx in np.linspace(-4.0, 4.0, 9):
        gl_lines.append([[fx, 1.2, 0.5], [fx, 1.2, 7.0]])
    for fz in np.linspace(0.5, 7.0, 8):
        gl_lines.append([[-4.0, 1.2, fz], [4.0, 1.2, fz]])

    for line in gl_lines:
        pts_3d = np.array(line, dtype=np.float32)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, np.zeros(4))
        pts_2d = np.int32(pts_2d.reshape(-1, 2))
        cv2.line(img, tuple(pts_2d[0]), tuple(pts_2d[1]), (40, 40, 40), 1)

    # 2. Draw Shelf frame pillars (industrial grey vertical legs and blue horizontal beams)
    # Vertical pillars (Industrial Grey, S=0)
    pillar_color = (80, 80, 80) # Grey in BGR
    draw_cube_3d(img, rvec, tvec, K, -1.4, 0.0, SHELF_Z, 0.08, 2.4, 0.5, pillar_color, None)
    draw_cube_3d(img, rvec, tvec, K, 1.4, 0.0, SHELF_Z, 0.08, 2.4, 0.5, pillar_color, None)

    # Horizontal shelf beams (Industrial Blue, H=105 outside brown range)
    beam_color = (220, 130, 20) # Blue in BGR
    for y_lvl in LEVELS:
        draw_cube_3d(img, rvec, tvec, K, 0.0, y_lvl + 0.05, SHELF_Z, 2.8, 0.08, 0.5, beam_color, None)

    # 3. Draw static ArUco markers on the shelf corners
    for marker in STATIC_MARKERS:
        draw_marker_3d(img, rvec, tvec, K, marker["id"], marker["x"], marker["y"], marker["z"], 0.20)

    # 4. Draw boxes on the shelves with organic offsets and detailed decals
    box_base_color = (65, 95, 135) # BGR Cardboard brown
    box_size = 0.48
    for l_idx, y_lvl in enumerate(LEVELS):
        for c_idx, x_col in enumerate(COLUMNS):
            if boxes_active.get((l_idx, c_idx), False) or boxes_active.get(f"{l_idx},{c_idx}", False):
                # Deterministic minor organic offset
                box_seed = int(abs(x_col * 1000 + y_lvl * 100)) % 10000
                np.random.seed(box_seed)
                offset_x = np.random.uniform(-0.03, 0.03)
                offset_y = np.random.uniform(-0.01, 0.0)
                offset_z = np.random.uniform(-0.04, 0.04)

                cz = SHELF_Z - 0.15 + offset_z
                cy = y_lvl - box_size / 2.0 + offset_y  # sits on the beam
                cx = x_col + offset_x

                # Jitter box color slightly to make them look separate
                color_jitter = int(np.random.uniform(-10, 10))
                box_color = (
                    max(0, min(255, box_base_color[0] + color_jitter)),
                    max(0, min(255, box_base_color[1] + color_jitter)),
                    max(0, min(255, box_base_color[2] + color_jitter))
                )

                draw_cube_3d(img, rvec, tvec, K, cx, cy, cz, box_size, box_size, box_size, box_color, (140, 110, 80), draw_details=True)

    # Apply lighting brightness jitter
    brightness = random.uniform(0.85, 1.15)
    img = cv2.convertScaleAbs(img, alpha=brightness, beta=random.randint(-5, 5))

    # Add a vignette/spotlight effect simulating drone camera searchlight (very premium look)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (int(w / 2), int(h / 2)), int(w * 0.72), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (81, 81), 0)
    mask = 0.45 + 0.55 * mask
    img = (img * mask[:, :, np.newaxis]).astype(np.uint8)

    # Add subtle Gaussian camera noise
    noise = np.random.normal(0, 1.5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img
