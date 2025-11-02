import cv2
import numpy as np
import pyttsx3

# 1️⃣ Load image
img = cv2.imread("ColorVisionDemo/images/plate2.png")
if img is None:
    raise FileNotFoundError("Image not found. Check your path!")

# 2️⃣ Define color-blindness matrix (Deuteranopia example)
deuteranopia_matrix = np.array([
    [0.367, 0.861, -0.228],
    [0.280, 0.673,  0.047],
    [-0.012, 0.043, 0.969]
])

def simulate_colorblindness(image, matrix):
    """Apply matrix transform to simulate color blindness."""
    img = image / 255.0
    transformed = img.dot(matrix.T)
    transformed = np.clip(transformed, 0, 1)
    return (transformed * 255).astype(np.uint8)

# 3️⃣ Simulate color-blind view
simulated = simulate_colorblindness(img, deuteranopia_matrix)

# 4️⃣ Apply basic color correction (Daltonization-style)
def daltonize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.add(a, 15)       # shift red-green axis
    b = cv2.subtract(b, 10)  # shift blue-yellow axis
    corrected = cv2.merge([l, a, b])
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

corrected = daltonize(simulated)

# 5️⃣ Combine images side by side
combined = np.hstack((img, simulated, corrected))

# 6️⃣ Display output
cv2.imshow("Normal | Color-blind Simulation | Corrected", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7️⃣ Optional: voice feedback
engine = pyttsx3.init()
engine.say("Color correction completed.")
engine.runAndWait()
