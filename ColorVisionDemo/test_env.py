import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("\n🚀 Environment test starting...\n")

# --- TensorFlow test ---
print(f"TensorFlow version: {tf.__version__}")
print("TensorFlow devices available:", tf.config.list_physical_devices())

# Simple tensor test
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[2.0, 0.0], [1.0, 3.0]])
print("\nMatrix multiplication test:\n", tf.matmul(a, b).numpy())

# --- NumPy test ---
arr = np.array([1, 2, 3, 4, 5])
print("\nNumPy mean test:", np.mean(arr))

# --- OpenCV test ---
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:] = (0, 255, 0)  # green square
cv2.imwrite("test_image.jpg", img)
print("\nOpenCV test image saved as test_image.jpg")

# --- Matplotlib test ---
plt.plot([0, 1, 2], [0, 1, 4])
plt.title("Matplotlib Test Plot")
plt.savefig("test_plot.png")
print("Matplotlib test plot saved as test_plot.png")

# --- Scikit-learn test ---
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression().fit(X, y)
print("\nScikit-learn prediction for 5:", model.predict([[5]])[0])

print("\n✅ All libraries ran successfully!")
