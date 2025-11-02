import cv2
import numpy as np
import pyttsx3

# 1️⃣ Load image
img = cv2.imread("ColorVisionDemo/images/plate1.png")
if img is None:
    raise FileNotFoundError("Image not found. Check your path!")

img = cv2.resize(img, (300, 300))

# 2️⃣ Accurate Deuteranopia simulation (Machado et al. + perceptual tuning)
def simulate_deuteranopia(image):
    img = image.astype(np.float32) / 255.0

    # --- RGB → LMS conversion matrix ---
    rgb2lms = np.array([
        [17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.0299566, 0.184309, 1.46709]
    ])

    # --- LMS → RGB back conversion matrix ---
    lms2rgb = np.linalg.inv(rgb2lms)

    # --- Deuteranopia confusion matrix (Machado et al.) ---
    deuteranopia = np.array([
        [1.0,     0.0,     0.0],
        [0.494207, 0.0,   1.24827],
        [0.0,     0.0,     1.0]
    ])

    # Convert RGB → LMS
    LMS = np.dot(img.reshape(-1, 3), rgb2lms.T)

    # Apply Deuteranopia confusion
    sim_LMS = np.dot(LMS, deuteranopia.T)

    # Convert LMS → RGB
    sim_RGB = np.dot(sim_LMS, lms2rgb.T)

    # Normalize and clip
    sim_RGB = np.clip(sim_RGB, 0, 1)

    simulated = (sim_RGB.reshape(img.shape) * 255).astype(np.uint8)

    # --- Tone tuning to match brownish reference (warmer gamma) ---
    simulated = cv2.cvtColor(simulated, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(simulated)
    
    b = cv2.add(b, 25)   # more yellow tone
    a = cv2.subtract(a, 3)  # reduce magenta cast
    l = cv2.add(l, 6)  # brighten overall
    
    simulated = cv2.merge([l, a, b])
    simulated = cv2.cvtColor(simulated, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(simulated, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 0.75)  # reduce saturation by 25%
    hsv = cv2.merge([h, s, v])
    simulated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    simulated = cv2.convertScaleAbs(simulated, alpha=1.05, beta=0)
    return simulated

# 3️⃣ Generate the simulated image
deut_sim = simulate_deuteranopia(img)

def daltonize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.add(a, 15)       # shift red-green axis
    b = cv2.subtract(b, 10)  # shift blue-yellow axis
    corrected = cv2.merge([l, a, b])
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

corrected = daltonize(deut_sim)

# 5️⃣ Combine images side by side
combined = np.hstack((img, deut_sim, corrected))

# 6️⃣ Display output
cv2.imshow("Normal | Color-blind Simulation | Corrected", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7️⃣ Optional: voice feedback
engine = pyttsx3.init()
engine.say("Color correction completed.")
engine.runAndWait()