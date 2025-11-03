import cv2
import numpy as np
import os
import tensorflow as tf
from gesture_control import detect_gesture

# ---------------------------
# Color vision matrices
# ---------------------------
rgb2lms = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709]
])
lms2rgb = np.linalg.inv(rgb2lms)

# ---------------------------
# Simulation functions
# ---------------------------
def simulate_deuteranopia(image):
    img = image.astype(np.float32) / 255.0
    deuteranopia = np.array([
        [1.0, 0.0, 0.0],
        [0.494207, 0.0, 1.24827],
        [0.0, 0.0, 1.0]
    ])
    LMS = np.dot(img.reshape(-1, 3), rgb2lms.T)
    sim_LMS = np.dot(LMS, deuteranopia.T)
    sim_RGB = np.dot(sim_LMS, lms2rgb.T)
    sim_RGB = np.clip(sim_RGB, 0, 1)
    simulated = (sim_RGB.reshape(img.shape) * 255).astype(np.uint8)
    return simulated

def apply_model_to_image(image, model, input_size=(128, 128)):
    orig_h, orig_w = image.shape[:2]
    patch = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
    patch = patch.astype("float32") / 255.0
    patch = np.expand_dims(patch, axis=0)
    pred = model.predict(patch, verbose=0)[0]
    pred = np.clip(pred, 0.0, 1.0)
    pred = (pred * 255.0).astype(np.uint8)
    return cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

# ---------------------------
# Color detection helper
# ---------------------------
def detect_color(frame):
    return ["Red", "Green", "Blue"]  # placeholder

def format_detected_colors(detected_colors, max_line_len=40):
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

def draw_text_with_background(img, text_lines, position=(20, 40)):
    x, y = position
    overlay = img.copy()
    line_height = 35
    padding = 10
    bg_height = line_height * len(text_lines) + padding * 2
    cv2.rectangle(overlay, (x - 10, y - 30), (x + 600, y + bg_height - 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (x, y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

# ---------------------------
# Load TensorFlow model
# ---------------------------
model_path = r"E:\Programs\3rdyear\Smart-Color-Aid\models\daltonization_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found: " + model_path)
model = tf.keras.models.load_model(model_path, compile=False)

# ---------------------------
# Camera initialization
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

enhanced_mode = False
use_model = False
hand_present = False
intensity = 100  # initial slider value

# ---------------------------
# Trackbar callback
# ---------------------------
def update_intensity(val):
    global intensity
    intensity = val

cv2.namedWindow("Smart Color Aid")
cv2.createTrackbar("Intensity", "Smart Color Aid", intensity, 200, update_intensity)

# ---------------------------
# Main loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Gesture detection
    gesture_type = detect_gesture(frame)
    if gesture_type and not hand_present:
        if gesture_type in ['index', 'thumb', 'open']:
            enhanced_mode = not enhanced_mode
            use_model = enhanced_mode
            print(f"{gesture_type.capitalize()} gesture detected! Enhanced mode: {enhanced_mode}")
        hand_present = True
    elif not gesture_type:
        hand_present = False

    # Deuteranopia simulation
    deuter_frame = simulate_deuteranopia(frame.copy()) if enhanced_mode else frame.copy()

    # Apply model with intensity blending
    if use_model:
        model_frame = apply_model_to_image(deuter_frame, model)
        alpha = intensity / 100.0
        alpha = np.clip(alpha, 0, 2.0)
        # Convert to LAB for blending
        lab_orig = cv2.cvtColor(deuter_frame, cv2.COLOR_BGR2LAB)
        lab_model = cv2.cvtColor(model_frame, cv2.COLOR_BGR2LAB)
        blended_lab = lab_orig.copy()
        blended_lab[..., 1] = cv2.addWeighted(lab_orig[..., 1], 1 - alpha, lab_model[..., 1], alpha, 0)
        blended_lab[..., 2] = cv2.addWeighted(lab_orig[..., 2], 1 - alpha, lab_model[..., 2], alpha, 0)
        model_frame = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)
    else:
        model_frame = deuter_frame.copy()

    # Color detection
    detected_colors = detect_color(frame)
    text_lines = format_detected_colors(detected_colors)
    draw_text_with_background(frame, text_lines)
    draw_text_with_background(deuter_frame, text_lines)
    draw_text_with_background(model_frame, text_lines)

    # Overlay status
    status_text = f"Enhanced Mode {'ON' if enhanced_mode else 'OFF'} | Intensity: {intensity}%"
    cv2.putText(model_frame, status_text, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Combine and show
    combined = np.hstack((frame, deuter_frame, model_frame))
    cv2.imshow("Smart Color Aid", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
