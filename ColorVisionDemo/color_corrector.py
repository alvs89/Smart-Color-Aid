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
        [1.0,     0.0,     0.0],
        [0.494207, 0.0,   1.24827],
        [0.0,     0.0,     1.0]
    ])
    LMS = np.dot(img.reshape(-1, 3), rgb2lms.T)
    sim_LMS = np.dot(LMS, deuteranopia.T)
    sim_RGB = np.dot(sim_LMS, lms2rgb.T)
    sim_RGB = np.clip(sim_RGB, 0, 1)
    simulated = (sim_RGB.reshape(img.shape) * 255).astype(np.uint8)

    simulated = cv2.cvtColor(simulated, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(simulated)
    b = cv2.add(b, 25)
    a = cv2.subtract(a, 3)
    l = cv2.add(l, 6)
    simulated = cv2.merge([l, a, b])
    simulated = cv2.cvtColor(simulated, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(simulated, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 0.75)
    hsv = cv2.merge([h, s, v])
    simulated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    simulated = cv2.convertScaleAbs(simulated, alpha=1.05, beta=0)
    return simulated

def simulate_deuteranomaly(image, severity=0.6):
    img = image.astype(np.float32) / 255.0
    deuteranomaly = np.array([
        [1.0, 0.0, 0.0],
        [severity * 0.494207, 1 - severity, severity * 1.24827],
        [0.0, 0.0, 1.0]
    ])
    LMS = np.dot(img.reshape(-1, 3), rgb2lms.T)
    sim_LMS = np.dot(LMS, deuteranomaly.T)
    sim_RGB = np.dot(sim_LMS, lms2rgb.T)
    sim_RGB = np.clip(sim_RGB, 0, 1)
    simulated = (sim_RGB.reshape(img.shape) * 255).astype(np.uint8)

    simulated = cv2.cvtColor(simulated, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(simulated)
    b = cv2.add(b, 10)
    a = cv2.subtract(a, 2)
    l = cv2.add(l, 4)
    simulated = cv2.merge([l, a, b])
    simulated = cv2.cvtColor(simulated, cv2.COLOR_LAB2BGR)
    simulated = cv2.convertScaleAbs(simulated, alpha=1.03, beta=2)
    return simulated

def daltonize(image, enhance_luminance=True):
    """
    Daltonize an image for deuteranopia using perceptual compensation.
    Steps:
      1. Simulate deuteranopia perception.
      2. Compute error between original and simulated.
      3. Reinject lost chromatic information into visible channels.
    """

    # --- Step 1: Convert to float and simulate ---
    img = image.astype(np.float32) / 255.0
    LMS = np.dot(img.reshape(-1, 3), rgb2lms.T)

    # Deuteranopia projection matrix (missing M-cone info)
    deuteranopia = np.array([
        [1.0,     0.0,     0.0],
        [0.494207, 0.0,   1.24827],
        [0.0,     0.0,     1.0]
    ])
    sim_LMS = np.dot(LMS, deuteranopia.T)
    sim_RGB = np.dot(sim_LMS, lms2rgb.T)
    sim_RGB = np.clip(sim_RGB, 0, 1).reshape(img.shape)

    # --- Step 2: Compute error ---
    error = img - sim_RGB

    # --- Step 3: Compensation ---
    # Reinject lost red–green information into blue–yellow channels.
    correction = np.zeros_like(error)
    correction[..., 1] = 0.7 * error[..., 0]  # add red difference to green
    correction[..., 2] = 0.7 * error[..., 0]  # add red difference to blue

    daltonized = np.clip(img + correction, 0, 1)

    # --- Step 4: Optional mild contrast/saturation enhancements ---
    daltonized = (daltonized * 255).astype(np.uint8)
    hsv = cv2.cvtColor(daltonized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    h = h.astype(np.float32)
    h[(h > 20) & (h < 50)] += 8   # shift yellows slightly toward orange
    h[(h > 90) & (h < 130)] -= 8  # shift cyans/blues slightly toward aqua
    h = np.clip(h, 0, 179).astype(np.uint8)
    
    s = cv2.multiply(s, 1.1)
    v = cv2.convertScaleAbs(v, alpha=1.05, beta=0)
    
    hsv = cv2.merge([h, np.clip(s, 0, 255), np.clip(v, 0, 255)])
    daltonized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if enhance_luminance:
        lab = cv2.cvtColor(daltonized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        daltonized = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
    return daltonized


# ---------------------------
# Model helper
# ---------------------------
def apply_model_to_image(image, model, input_size=(128, 128)):
    orig_h, orig_w = image.shape[:2]
    patch = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
    patch = patch.astype("float32") / 255.0
    patch = np.expand_dims(patch, axis=0)
    pred = model.predict(patch)[0]
    pred = np.clip(pred, 0.0, 1.0)
    pred = (pred * 255.0).astype(np.uint8)
    return cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

# ---------------------------
# Load model and image
# ---------------------------
model_path = r"E:\Programs\3rdyear\Smart-Color-Aid\models\daltonization_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found: " + model_path)
model = tf.keras.models.load_model(model_path, compile=False)

original = cv2.imread("images/plate1.png")
if original is None:
    raise FileNotFoundError("Image not found. Check path!")

original = cv2.resize(original, (300, 300))

# ---------------------------
# Generate variants
# ---------------------------
deut_anomaly = simulate_deuteranomaly(original, 0.6)
deut_nopia = simulate_deuteranopia(original)
model_corrected = apply_model_to_image(deut_nopia, model)

# ---------------------------
# Prepare LAB for blending
# ---------------------------
lab_orig = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
lab_corr = cv2.cvtColor(model_corrected, cv2.COLOR_BGR2LAB)

# ---------------------------
# Create interactive window
# ---------------------------
window_name = "Normal | Deuteranomaly | Deuteranopia | Model-corrected"

def update_slider(val):
    alpha = val / 100.0
    alpha = np.clip(alpha, 0, 2.0)
    blended_lab = lab_orig.copy()
    blended_lab[..., 1] = cv2.addWeighted(lab_orig[..., 1], 1 - alpha, lab_corr[..., 1], alpha, 0)
    blended_lab[..., 2] = cv2.addWeighted(lab_orig[..., 2], 1 - alpha, lab_corr[..., 2], alpha, 0)
    blended_bgr = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)
    
    # Combine horizontally for direct comparison
    combined = np.hstack((original, deut_anomaly, deut_nopia, blended_bgr))
    cv2.imshow(window_name, combined)

cv2.namedWindow(window_name)
cv2.createTrackbar("Intensity", window_name, 100, 100, update_slider)
update_slider(100)

cv2.waitKey(0)
cv2.destroyAllWindows()
