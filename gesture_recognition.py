import tensorflow as tf
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from dataAugmentation import train_generator

# Load the trained model
model = tf.keras.models.load_model('hand_gesture_model.h5')

# Function to preprocess the frame
def preprocess_frame(frame):
    # Resize the frame to 64x64 as expected by the model
    resized_frame = cv2.resize(frame, (64, 64))
    # Normalize pixel values to range [0, 1]
    normalized_frame = resized_frame / 255.0
    # Expand dimensions to match the model's input shape
    return np.expand_dims(normalized_frame, axis=0)

# Initialize the hand detector from cvzone
detector = HandDetector(detectionCon=0.8, maxHands=1)  # Adjust confidence and maxHands as needed

# Start capturing from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Dictionary to map class indices to gesture labels
gesture_map = {v: k for k, v in train_generator.class_indices.items()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Detect hand in the frame
    hands, img = detector.findHands(frame, flipType=False)

    if hands:
        # Get the bounding box from the detected hand
        hand = hands[0]
        bbox = hand['bbox']  # Bounding box: [x, y, w, h]

        # Extract the hand region using the bounding box
        x, y, w, h = bbox
        hand_region = frame[y:y+h, x:x+w]

        # Preprocess the hand region for model prediction
        preprocessed_frame = preprocess_frame(hand_region)

        # Predict the gesture using the trained model
        predictions = model.predict(preprocessed_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Get the corresponding gesture label
        predicted_gesture = gesture_map[predicted_class]

        # Display the predicted gesture on the frame
        cv2.putText(frame, predicted_gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw the bounding box on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
