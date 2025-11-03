import cv2
import numpy as np

def simulate_deuteranopia(frame):
    """
    Simulates deuteranopia AND enhances confusing colors
    so they are more distinguishable to color-blind users.
    """
    # Convert to HSV for better hue-based manipulation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Mask ranges for confusing color zones
    red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 50, 50), (90, 255, 255))
    brown_mask = cv2.inRange(hsv, (10, 100, 20), (20, 255, 200))
    pink_mask = cv2.inRange(hsv, (161, 80, 100), (179, 255, 255))

    # Apply color remapping (enhancement for differentiation)
    frame[red_mask1 > 0] = [255, 0, 255]   # Red → Magenta
    frame[red_mask2 > 0] = [255, 0, 255]
    frame[green_mask > 0] = [0, 255, 255]  # Green → Cyan
    frame[brown_mask > 0] = [0, 165, 255]  # Brown → Orange
    frame[pink_mask > 0] = [147, 20, 255]  # Pink → Violet

    # Apply mild brightness enhancement to make contrasts pop
    v = cv2.add(v, 30)
    hsv_enhanced = cv2.merge([h, s, v])
    output = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return output
