import cv2
import numpy as np
import pyttsx3
import time
from color_adjustment import simulate_deuteranopia, detect_color

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Camera not found or cannot be opened.")
    exit()

# Initialize voice engine 
engine = pyttsx3.init()
engine.setProperty("rate", 170)   # Speaking speed
engine.setProperty("volume", 1.0) # Volume level

# Voice timing control
last_spoken = ""
last_speak_time = 0
speak_delay = 3  # seconds between announcements

def speak_colors(detected_colors):
    """Speaks detected colors with time delay and change check."""
    global last_spoken, last_speak_time
    current_time = time.time()

    if not detected_colors:
        message = "No color detected"
    else:
        message = "The colors are " + ", ".join(detected_colors)

    # Speak only if colors changed AND enough time passed
    if (message != last_spoken) and (current_time - last_speak_time > speak_delay):
        last_spoken = message
        last_speak_time = current_time
        engine.say(message)
        engine.runAndWait()

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
        print("❌ Error: Unable to capture frame.")
        break

    # Right window: simulate deuteranopia
    deuter_frame = simulate_deuteranopia(frame.copy())

    # Detect colors from the normal frame
    detected_colors = detect_color(frame)

    # Format and draw text
    text_lines = format_detected_colors(detected_colors)
    draw_text_with_background(frame, text_lines)
    draw_text_with_background(deuter_frame, text_lines)

    # Speak colors periodically
    speak_colors(detected_colors)

    # Combine windows (Normal left | Deuteranopia right)
    combined = np.hstack((frame, deuter_frame))

    # Display
    cv2.imshow("Smart Color Aid | Normal (Left) vs Deuteranopia (Right)", combined)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
