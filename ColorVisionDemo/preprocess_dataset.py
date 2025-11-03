import os
import cv2
import numpy as np

# Path to your dataset folder
DATASET_DIR = "../dataset/caltech-101/101_ObjectCategories"
IMG_SIZE = (128, 128)  # resize all images for faster processing

def load_images(dataset_dir, limit_per_class=50):
    X, y = [], []
    classes = [cls for cls in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cls))]

    for label, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        print(f"📂 Loading {cls}...")
        count = 0

        for file in os.listdir(cls_dir):
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(cls_dir, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0  # normalize to [0, 1]
                X.append(img)
                y.append(label)
                count += 1
                if count >= limit_per_class:
                    break

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"\n✅ Loaded {len(X)} images from {len(classes)} classes.")
    return X, y

if __name__ == "__main__":
    X, y = load_images(DATASET_DIR)
    print("Sample image shape:", X[0].shape)
