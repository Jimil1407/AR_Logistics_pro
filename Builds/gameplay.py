import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

inventory = {
    "boots": {"price": 100, "stock": 10},
    "flip_flops": {"price": 20, "stock": 30},
    "loafers": {"price": 75, "stock": 15},
    "sandals": {"price": 50, "stock": 20},
    "sneakers": {"price": 90, "stock": 5},
    "soccer_shoes": {"price": 120, "stock": 8}
}

base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(len(inventory), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

class_names = list(inventory.keys())

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils
objectron = mp_objectron.Objectron(static_image_mode=False, max_num_objects=5,
                                   min_detection_confidence=0.5, model_name='Shoe')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = objectron.process(rgb_frame)

    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)

            landmarks = [(landmark.x, landmark.y) for landmark in detected_object.landmarks_2d.landmark]
            x_min = int(min(landmarks, key=lambda x: x[0])[0] * frame.shape[1])
            y_min = int(min(landmarks, key=lambda x: x[1])[1] * frame.shape[0])
            x_max = int(max(landmarks, key=lambda x: x[0])[0] * frame.shape[1])
            y_max = int(max(landmarks, key=lambda x: x[1])[1] * frame.shape[0])

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                roi_resized = cv2.resize(roi, (224, 224))
                roi_array = img_to_array(roi_resized)
                roi_array = preprocess_input(roi_array)
                roi_array = np.expand_dims(roi_array, axis=0)

                pred = model.predict(roi_array)
                confidence = np.max(pred)
                if confidence > 0.7:
                    class_id = np.argmax(pred)
                    class_name = class_names[class_id]
                    inventory_info = inventory.get(class_name, {"price": "N/A", "stock": "N/A"})
                    text = f"{class_name.capitalize()} | Price: ${inventory_info['price']} | Stock: {inventory_info['stock']}"
                else:
                    text = "unknown | Price: N/A | Stock: N/A"

                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Footwear Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()