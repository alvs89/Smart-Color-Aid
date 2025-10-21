import cv2
import numpy as np

def detect_color(frame):
    """
    Enhanced color detection with adaptive HSV ranges.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (7, 7), 0)  # smooth out lighting noise

    colors = {
        "Red": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
        "Green": [(35, 50, 50), (90, 255, 255)],
        "Blue": [(90, 60, 50), (130, 255, 255)],
        "Yellow": [(15, 80, 100), (40, 255, 255)]
    }

    detected = []

    for name, ranges in colors.items():
        if name == "Red":
            lower1, upper1 = np.array(ranges[0]), np.array(ranges[1])
            lower2, upper2 = np.array(ranges[2]), np.array(ranges[3])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.add(mask1, mask2)
        else:
            lower, upper = np.array(ranges[0]), np.array(ranges[1])
            mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        if cv2.countNonZero(mask) > 800:
            detected.append(name)

    return detected
