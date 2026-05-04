import cv2
import numpy as np
import tensorflow as tf
import json

# ----------------------------
# Load model
# ----------------------------
model = tf.keras.models.load_model("ASL.h5")
print("Model loaded")

# ----------------------------
# Load labels
# ----------------------------
with open("labels (2).json", "r") as f:
    labels = json.load(f)

print("Labels:", labels)

# ----------------------------C:\Users\91949\OneDrive\Desktop\SignScribe\test_images\15.png
# Load image
# ----------------------------
IMAGE_PATH = "test_images/28.png"   # <-- change if needed
IMG_SIZE = 64

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("Image not found. Check the path.")

# Show original image
cv2.imshow("Original Image", img)

# ----------------------------
# Preprocess image
# ----------------------------
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
img_normalized = img_resized / 255.0
img_input = np.expand_dims(img_normalized, axis=0)

# ----------------------------
# Predict
# ----------------------------
prediction = model.predict(img_input, verbose=0)
idx = np.argmax(prediction)
confidence = prediction[0][idx] * 100
label = labels[idx]

print(f"Prediction: {label.upper()} ({confidence:.2f}%)")

# ----------------------------
# Display result
# ----------------------------
cv2.putText(
    img,
    f"{label.upper()} ({confidence:.1f}%)",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
