import cv2
import torch
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define inventory
inventory = {
    "boots": {"price": 100, "stock": 10},
    "flip_flops": {"price": 20, "stock": 30},
    "loafers": {"price": 75, "stock": 15},
    "sandals": {"price": 50, "stock": 20},
    "sneakers": {"price": 90, "stock": 5},
    "soccer_shoes": {"price": 120, "stock": 8}
}

# Load the trained classifier model and class indices
model = load_model('/Users/jimildigaswala/Desktop/gameproject/Assets/saved_model.h5')
class_indices = np.load('class_indices.npy', allow_pickle=True).item()
classes = {v: k for k, v in class_indices.items()}  # Reverse mapping for display

# Initialize MediaPipe Objectron
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils
objectron = mp_objectron.Objectron(static_image_mode=False, max_num_objects=5,
                                   min_detection_confidence=0.5, model_name='Shoe')

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects
    results = objectron.process(rgb_frame)

    # Process and display detected objects
    if results.detected_objects:
        for detected_object in results.detected_objects:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)

            # Extract bounding box coordinates
            landmarks = [(landmark.x, landmark.y) for landmark in detected_object.landmarks_2d.landmark]
            x_min = int(min(landmarks, key=lambda x: x[0])[0] * frame.shape[1])
            y_min = int(min(landmarks, key=lambda x: x[1])[1] * frame.shape[0])
            x_max = int(max(landmarks, key=lambda x: x[0])[0] * frame.shape[1])
            y_max = int(max(landmarks, key=lambda x: x[1])[1] * frame.shape[0])

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Extract ROI and preprocess for model
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                roi_resized = cv2.resize(roi, (224, 224))
                roi_array = img_to_array(roi_resized) / 255.0
                roi_array = np.expand_dims(roi_array, axis=0)

                # Predict footwear type with confidence check
                pred = model.predict(roi_array)
                confidence = np.max(pred)  # Get the confidence score
                if confidence > 0.6:  # Set confidence threshold
                    class_id = np.argmax(pred)
                    class_name = classes[class_id]

                    # Fetch inventory details
                    inventory_info = inventory.get(class_name, {"price": "N/A", "stock": "N/A"})
                    text = f"{class_name.capitalize()} | Price: ${inventory_info['price']} | Stock: {inventory_info['stock']}"
                else:
                    # Display unknown if confidence is low
                    text = "Unknown | Price: N/A | Stock: N/A"

                # Display inventory information
                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Footwear Detection', frame)

    # Press 'ESC' to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()