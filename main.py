import cv2
import numpy as np
from color_adjustment import simulate_deuteranopia, detect_color
from gesture_control import detect_gesture

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found or cannot be opened.")
    exit()

enhanced_mode = False
hand_present = False
gesture_active = False

# Voice timing control
def draw_text_with_background(img, text_lines, position=(20, 40)):
    """Draws readable text box with translucent background."""
    x, y = position
    overlay = img.copy()
    line_height = 35
    padding = 10

    # Background box
    bg_height = line_height * len(text_lines) + padding * 2
    cv2.rectangle(overlay, (x - 10, y - 30),
                  (x + 600, y + bg_height - 20),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # Write each line
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (x, y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2, cv2.LINE_AA)

def format_detected_colors(detected_colors, max_line_len=40):
    """Breaks long color list into multiple readable lines."""
    if not detected_colors:
        return ["Detected: None"]

    text = "Detected: " + ", ".join(detected_colors)
    lines = []
    while len(text) > max_line_len:
        split_idx = text[:max_line_len].rfind(", ")
        if split_idx == -1:
            split_idx = max_line_len
        lines.append(text[:split_idx])
        text = text[split_idx + 2:]
    lines.append(text)
    return lines

# Main camera loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Inside your while True loop
    gesture_type = detect_gesture(frame)  

# Toggle enhanced_mode only once per hand appearance
    if gesture_type and not hand_present:
        if gesture_type == 'index':
            enhanced_mode = not enhanced_mode
            print("Index gesture detected! Enhanced mode is now:", enhanced_mode)
        elif gesture_type == 'thumb':
            enhanced_mode = not enhanced_mode
            print("Thumbs-up detected! Enhanced mode is now:", enhanced_mode)
        elif gesture_type == 'open':
            enhanced_mode = not enhanced_mode
            print("Open hands detected! Enhanced mode is now:", enhanced_mode)
        hand_present = True  # Lock until hand disappears
    elif not gesture_type:
        hand_present = False  # Reset when hand disappears

    # Apply deuteranopia simulation
    if enhanced_mode:
        deuter_frame = simulate_deuteranopia(frame.copy())  # your enhanced mapping
    else:
        deuter_frame = frame.copy()  # normal view

    # Detect colors from the normal frame
    detected_colors = detect_color(frame)
    # Format and draw text
    text_lines = format_detected_colors(detected_colors)
    draw_text_with_background(frame, text_lines)
    draw_text_with_background(deuter_frame, text_lines)

    # Overlay enhanced mode status
    status_text = "Enhanced Mode ON" if enhanced_mode else "Enhanced Mode OFF"
    cv2.putText(deuter_frame, status_text, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Combine windows (Normal left | Deuteranopia right)
    combined = np.hstack((frame, deuter_frame))

    # Display
    cv2.imshow("Smart Color Aid | Normal (Left) vs Deuteranopia (Right)", combined)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
