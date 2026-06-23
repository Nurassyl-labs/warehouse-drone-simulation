import cv2
import numpy as np

def detect_aruco_markers(img):
    """
    Detects ArUco markers in the given image.
    Returns a list of dictionaries, each containing:
    - id: ID of the detected marker (int)
    - cx: Center X coordinate in pixels (float)
    - cy: Center Y coordinate in pixels (float)
    - area: Pixel area of the marker (float)
    """
    if img is None:
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try using the modern ArucoDetector API (OpenCV 4.7.x+) or fallback to legacy
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        # Fallback for older OpenCV versions
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    visible_markers = []
    if ids is not None:
        for i, m_id in enumerate(ids.flatten()):
            c_pts = corners[i].reshape(4, 2)
            # Compute center as mean of corners
            cx = float(np.mean(c_pts[:, 0]))
            cy = float(np.mean(c_pts[:, 1]))
            # Compute area using the Shoelace formula
            area = float(0.5 * np.abs(np.dot(c_pts[:, 0], np.roll(c_pts[:, 1], 1)) - np.dot(c_pts[:, 1], np.roll(c_pts[:, 0], 1))))
            visible_markers.append({
                "id": int(m_id),
                "cx": cx,
                "cy": cy,
                "area": area
            })
    return visible_markers
