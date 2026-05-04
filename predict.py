import cv2
import numpy as np
import tensorflow as tf
import json

# ----------------------------
# Load model and labels
# ----------------------------
model = tf.keras.models.load_model("asl_model.keras")

with open("labels (2).json", "r") as f:
    labels = json.load(f)

print("Model loaded")
print("Labels:", labels)

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

IMG_SIZE = 64

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Fixed ROI box
    x1, y1, x2, y2 = 100, 100, 350, 350
    roi = frame[y1:y2, x1:x2]

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Place hand inside box",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    if roi.size != 0:
        # BGR → RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)
        idx = np.argmax(prediction)
        confidence = prediction[0][idx] * 100
        label = labels[idx]

        cv2.putText(
            frame,
            f"{label.upper()} ({confidence:.1f}%)",
            (x1, y2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    cv2.imshow("ASL Recognition (Press Q to Quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
