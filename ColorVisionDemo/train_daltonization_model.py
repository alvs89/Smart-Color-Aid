import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# ==============================================
# Load dataset (direct OpenCV-based loader)
# ==============================================
DATASET_DIR = "../dataset/caltech-101/101_ObjectCategories"
IMG_SIZE = (128, 128)

def load_dataset(path, limit_per_class=40):
    X = []
    print("📂 Loading images from:", path)

    for folder in os.listdir(path):
        subdir = os.path.join(path, folder)
        if not os.path.isdir(subdir):
            continue

        count = 0
        for file in os.listdir(subdir):
            if file.lower().endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(subdir, file))
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    X.append(img)
                    count += 1
                    if count >= limit_per_class:
                        break

    X = np.array(X, dtype=np.float32) / 255.0
    print(f"✅ Loaded {len(X)} images from {len(os.listdir(path))} folders")
    return X

X = load_dataset(DATASET_DIR, limit_per_class=30)

# ==============================================
# Simulate deuteranopia (simplified)
# ==============================================
def simulate_deuteranopia_batch(images):
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.494207, 0.0, 1.24827],
        [0.0, 0.0, 1.0]
    ])
    reshaped = images.reshape(-1, 3)
    sim = np.dot(reshaped, M.T)
    return np.clip(sim.reshape(images.shape), 0, 1)

print("🎨 Simulating color-deficient dataset...")
X_deficient = simulate_deuteranopia_batch(X)

# ==============================================
# Split dataset
# ==============================================
split = int(len(X) * 0.8)
X_train, X_val = X_deficient[:split], X_deficient[split:]
Y_train, Y_val = X[:split], X[split:]

print(f"🧩 Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# ==============================================
# Define U-Net-style autoencoder
# ==============================================
def build_daltonization_model():
    inputs = tf.keras.layers.Input(shape=(128, 128, 3))

    # Encoder
    x1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D(2)(x1)
    x2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling2D(2)(x2)

    # Bottleneck
    b = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D(2)(b)
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    u2 = tf.keras.layers.UpSampling2D(2)(c1)
    c2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u2)

    outputs = tf.keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')(c2)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_daltonization_model()
model.summary()

if os.path.exists("../models/daltonization_model.h5"):
    print("Loading existing model weights...")
    model.load_weights("../models/daltonization_model.h5")
    
# ==============================================
# Train model
# ==============================================
EPOCHS = 40
BATCH_SIZE = 16

print("Training started...")

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="../models/best_daltonization_model.h5",
    monitor='val_loss',
    mode = 'min',
    verbose=1
)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks = [checkpoint]
)

# ==============================================
# Save model
# ==============================================
os.makedirs("../models", exist_ok=True)
model.save("../models/daltonization_model.h5")
print("✅ Model saved as daltonization_model.h5")

# ==============================================
# Evaluate model on validation set
# ==============================================
val_loss, val_mae = model.evaluate(X_val, Y_val, batch_size=16)
print(f"Validation MSE Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

# ==============================================
# Visual inspection of random samples
# ==============================================

slider_intensity = 0.5  # 0.0 = original, 1.0 = full correction
num_samples = 10       # number of random validation samples to show
indices = random.sample(range(len(X_val)), num_samples)

print("🎨 Showing random validation samples with blended personalization...")

for idx in indices:
    original = Y_val[idx]
    deficient = X_val[idx]
    
    # Predict corrected image
    predicted = model.predict(np.expand_dims(deficient, axis=0))[0]
    
    # Apply slider/blending
    personalized = cv2.addWeighted(original, 1 - slider_intensity, predicted, slider_intensity, 0)
    
    # Plot side by side
    plt.figure(figsize=(16,4))
    
    plt.subplot(1,4,1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(1,4,2)
    plt.title("Deficient")
    plt.imshow(deficient)
    plt.axis('off')
    
    plt.subplot(1,4,3)
    plt.title("Predicted")
    plt.imshow(predicted)
    plt.axis('off')
    
    plt.subplot(1,4,4)
    plt.title(f"Personalized ({slider_intensity*100:.0f}%)")
    plt.imshow(personalized)
    plt.axis('off')
    
    plt.show()