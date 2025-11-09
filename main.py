import cv2
import numpy as np
import os
import tensorflow as tf

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

# Detect screen size (safe fallback)
try:
    import tkinter as tk
    _root = tk.Tk()
    SCREEN_WIDTH = _root.winfo_screenwidth()
    SCREEN_HEIGHT = _root.winfo_screenheight()
    _root.withdraw()
    _root.destroy()
except Exception:
    SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720

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

    # compute text width for dynamic background size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    max_w = 0
    for line in text_lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        if w > max_w:
            max_w = w
    box_width = max_w + padding * 2
    img_w = img.shape[1]
    # ensure background doesn't overflow image width
    box_right = min(x - 10 + box_width, img_w - 10)
    cv2.rectangle(overlay, (x - 10, y - 30), (box_right, y + bg_height - 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (x, y + i * line_height),
                    font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

# ---------------------------
# Load TensorFlow model
# ---------------------------
model_path = os.path.join("models", "best_daltonization_model.h5")
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

# Set filter enabled by default so the deuteranopia simulation is applied on the camera
enhanced_mode = True
use_model = False
intensity = 100  # initial slider value

# ---------------------------
# Trackbar callback
# ---------------------------
def update_intensity(val):
    global intensity
    intensity = val

cv2.namedWindow("Smart Color Aid")
# set trackbar max to 100 so intensity maps directly to percent
cv2.createTrackbar("Intensity", "Smart Color Aid", intensity, 100, update_intensity)

# ---------------------------
# Main loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Always compute a full deuteranopia simulation (used for display and model input)
    deuter_sim = simulate_deuteranopia(frame.copy())

    # use intensity as a percentage in [0,1]
    alpha = float(intensity) / 100.0
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Middle window: show full deuteranopia simulation when enhanced_mode is ON (no intensity applied)
    deuter_frame = deuter_sim.copy() if enhanced_mode else frame.copy()

    # Right window: adjust based on intensity
    if use_model:
        # run model on the full simulated image (deuter_sim)
        model_pred = apply_model_to_image(deuter_sim, model)

        # Blend chroma channels in LAB between simulated and model-predicted using alpha
        lab_deut = cv2.cvtColor(deuter_sim, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_model = cv2.cvtColor(model_pred, cv2.COLOR_BGR2LAB).astype(np.float32)
        blended_lab = lab_deut.copy()
        blended_lab[..., 1] = lab_deut[..., 1] * (1.0 - alpha) + lab_model[..., 1] * alpha
        blended_lab[..., 2] = lab_deut[..., 2] * (1.0 - alpha) + lab_model[..., 2] * alpha
        blended_lab = np.clip(blended_lab, 0, 255).astype(np.uint8)
        enhanced_result = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)

        # Final displayed model_frame: blend original <-> enhanced_result according to alpha
        model_frame = cv2.addWeighted(frame, 1.0 - alpha, enhanced_result, alpha, 0.0)
    else:
        # No model: right window is a blend between original and the full deuteranopia simulation
        model_frame = cv2.addWeighted(frame, 1.0 - alpha, deuter_sim, alpha, 0.0)

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

    # scale combined image to fit the screen if necessary (leave a small margin)
    margin = 0.95
    max_w = int(SCREEN_WIDTH * margin)
    max_h = int(SCREEN_HEIGHT * margin)
    h, w = combined.shape[:2]
    scale_w = max_w / w if w > 0 else 1.0
    scale_h = max_h / h if h > 0 else 1.0
    scale = min(1.0, scale_w, scale_h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        combined_display = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        combined_display = combined

    cv2.imshow("Smart Color Aid", combined_display)

    # Keyboard controls: 'q' to quit, 'e' toggles enhanced/model mode, 'm' toggles model only
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        enhanced_mode = not enhanced_mode
        use_model = enhanced_mode
        print(f"Enhanced mode toggled: {enhanced_mode}")
    elif key == ord('m'):
        # allow toggling model independently if desired
        use_model = not use_model
        print(f"Model usage toggled: {use_model}")

cap.release()
cv2.destroyAllWindows()
