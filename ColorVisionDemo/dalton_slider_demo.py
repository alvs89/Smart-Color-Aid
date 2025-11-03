import cv2
import numpy as np
import tensorflow as tf
import os

print(os.path.exists("models/daltonization_model.h5"))

# 1️⃣ Load your trained model
model_path = r"E:\Programs\3rdyear\Smart-Color-Aid\models\daltonization_model.h5"
print(os.path.exists(model_path))
model = tf.keras.models.load_model(model_path, compile=False)

# 2️⃣ Load and preprocess an image
original = cv2.imread("images/plate1.png")  # replace with your image path
input_img = cv2.resize(original, (128, 128))  # resize to match your model input
input_img = input_img.astype("float32") / 255.0  # normalize
input_img = np.expand_dims(input_img, axis=0)   # add batch dimension

# 3️⃣ Predict corrected image (fully daltonized)
corrected = model.predict(input_img)[0]  # remove batch dimension
corrected = (corrected * 255).astype(np.uint8)
corrected = cv2.resize(corrected, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)  # restore original size

lab_orig = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
lab_corr = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)

# 4️⃣ Function to update blended output
def update_slider(val):
    """Blend colors while preserving original brightness."""
    alpha = val / 100.0

    # Blend only the color channels (a,b), keep original lightness (L)
    blended_lab = lab_orig.copy()
    blended_lab[..., 1] = cv2.addWeighted(lab_orig[..., 1], 1 - alpha, lab_corr[..., 1], alpha, 0)
    blended_lab[..., 2] = cv2.addWeighted(lab_orig[..., 2], 1 - alpha, lab_corr[..., 2], alpha, 0)

    # Convert back to BGR
    blended_bgr = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)
    display_img = cv2.resize(blended_bgr, (0, 0), fx=0.5, fy=0.5)  # 50% smaller
    cv2.imshow("Personalized Daltonization", display_img)

# 5️⃣ Create OpenCV window and slider
cv2.namedWindow("Personalized Daltonization")
cv2.createTrackbar("Intensity", "Personalized Daltonization", 100, 100, update_slider)  # default 100%
update_slider(100)  # initial display

cv2.waitKey(0)
cv2.destroyAllWindows()
