import cv2
import numpy as np

def simulate_deuteranopia(frame):
    """Simulates Deuteranopia (green color blindness)"""
    matrix = np.array([
        [0.625, 0.7, -0.025],
        [0.7, 0.3, 0.0],
        [0.0, 0.3, 0.7]
    ], dtype=np.float32)
    # convert to float to avoid uint8 overflow during matrix multiplication
    f = frame.astype(np.float32)
    transformed = np.tensordot(f, matrix.T, axes=([-1], [-1]))
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return transformed


def detect_color(frame):
    """Detects multiple colors across full HSV range."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (7, 7), 0)

    color_ranges = {
        "Red": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
        "Orange": [(11, 100, 100), (25, 255, 255)],
        "Yellow": [(26, 80, 80), (35, 255, 255)],
        "Green": [(36, 40, 40), (85, 255, 255)],
        "Cyan": [(86, 50, 50), (95, 255, 255)],
        "Blue": [(96, 60, 50), (130, 255, 255)],
        "Magenta": [(131, 60, 50), (160, 255, 255)],
        "Pink": [(161, 80, 100), (179, 255, 255)],
        "Brown": [(10, 100, 20), (20, 255, 200)],
        "White": [(0, 0, 200), (180, 25, 255)],
        "Gray": [(0, 0, 60), (180, 25, 200)],
        "Black": [(0, 0, 0), (180, 255, 60)],
    }

    detected = []

    for name, ranges in color_ranges.items():
        if name == "Red":
            lower1, upper1 = np.array(ranges[0]), np.array(ranges[1])
            lower2, upper2 = np.array(ranges[2]), np.array(ranges[3])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.add(mask1, mask2)
        else:
            lower, upper = np.array(ranges[0]), np.array(ranges[1])
            mask = cv2.inRange(hsv, lower, upper)

        # Clean up noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        if cv2.countNonZero(mask) > 800:
            detected.append(name)

    return detected
