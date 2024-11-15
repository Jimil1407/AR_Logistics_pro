# Jimil Digaswala 21BAI1001
# AR Logistics Pro

## Brief Description of the Project
This project is a unique combination of Augmented Reality (AR) and Machine Learning (ML) for detecting and classifying various types of footwear in real-time using a live camera feed. We leverage AR for visually enhancing the detection process by drawing landmarks and bounding boxes around detected footwear, and ML for classifying the detected items into specific footwear categories.

The system is built using OpenCV, Mediapipe, and TensorFlow, with the MobileNet model utilized for classifying footwear types such as boots, flip flops, loafers, sandals, sneakers, and soccer shoes. The integration of AR techniques allows us to visually represent the detected objects in the video stream, making the user experience more engaging and interactive. Additionally, we provide real-time information about the detected footwear, including price and stock availability from an inventory.

### Key Highlights
- **Augmented Reality (AR)**: We use Mediapipe's Objectron model to draw 3D landmarks and bounding boxes around detected footwear, giving a visual AR effect that enhances user interaction.
- **Machine Learning (ML)**: A fine-tuned MobileNet model classifies the detected footwear, and the system provides corresponding inventory information.
- **Inventory Information**: Price and stock details are displayed, making the system suitable for real-time inventory management.

## Instructions for Running/Using the Project

### Environment Setup
1. Ensure you have Python installed on your system.
2. Install the required libraries using the following command:
   ```bash
   pip install opencv-python mediapipe tensorflow numpy

## Running the Project
1. Run the Python script in your terminal or preferred IDE:
   ```bash
   python footwear_detection.py
2. The script will open a window showing a live video feed from your webcam.
3. The system will detect footwear, draw AR bounding boxes and landmarks around them, and display classification details, including price and stock.

### Exiting the Program
1. Press the Esc key to close the application window.

### Additional Information
We have included a Jupyter Notebook file, Game_da.ipynb, which provides:

1. Classification Metrics: Detailed evaluation metrics for the model's performance.
1. Saved Model: A pre-trained model (saved_model.h5) is available in the Assets folder, which you can use for local data classification.

### Using the Saved Model

#### Step-by-Step Guide:

1. **First Way: Using the Default Model**
   - This method uses the model defined within the script to perform real-time detection and classification.
   - **Steps to Follow**:
     1. Make sure you have the required libraries installed (`opencv-python`, `mediapipe`, `tensorflow`, `numpy`).
     2. Open your terminal or your preferred IDE.
     3. Run the following command to execute the script:
        ```bash
        python footwear_detection.py
        ```
     4. The script will access your webcam and start detecting footwear in real-time.
     5. Footwear objects will be enclosed in AR bounding boxes, and their type, price, and stock details will be displayed on the screen.

2. **Second Way: Using the Saved Model (`saved_model.h5`)**
   - This method involves loading a pre-trained model from the `Assets` folder to perform classification. This is useful if you want to use the model on local data or if you prefer to customize the system.
   - **Steps to Follow**:
     1. Locate the `saved_model.h5` file in the `Assets` folder of the project directory.
     2. Modify the script to load the saved model:
        - Use the following code snippet to load the saved model:
          ```python
          from tensorflow.keras.models import load_model
          model = load_model('Assets/saved_model.h5')
          ```
     3. You can then run the modified script to classify detected footwear using the pre-trained model.
     4. This approach allows you to:
        - Use the model for your specific datasets.
        - Evaluate and test the model on different inputs as needed.
     5. The script will still use the same AR techniques for drawing landmarks and displaying classification information.

**Note**: The choice between these two methods depends on your use case. If you prefer real-time detection, use the default method. If you need to customize or test on specific datasets, using the saved model is recommended.
