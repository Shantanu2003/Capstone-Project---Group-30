import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('cnn_model.keras')  # Load the CNN model trained for gesture recognition
print("Model loaded from cnn_model.keras")

# Image dimensions used during training (ensure these match your dataset)
image_x, image_y = 300, 300  # Correct input size for the model (300x300)

# Define the list of gesture classes (adjust as per your dataset)
gesture_classes = ["Class1", "Class2", "Class3", "Class4"]  # Replace with your actual class names

def preprocess_image(frame):
    """
    Preprocess the frame from the webcam to the correct format for the model.
    Converts to grayscale, resizes, and normalizes the image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale (single channel)
    gray = cv2.resize(gray, (image_x, image_y))  # Resize the image to the input size expected by the model (300x300)
    gray = np.reshape(gray, (1, image_x, image_y, 1))  # Reshape the image to the correct format for the model (1, 300, 300, 1)
    gray = gray.astype('float32') / 255.0  # Normalize the pixel values to range [0, 1]
    return gray  # Return the preprocessed image

def predict_gesture(model, frame):
    """
    Predict the gesture from the webcam frame using the pre-trained model.
    """
    preprocessed_image = preprocess_image(frame)  # Preprocess the frame to match model input size
    prediction = model.predict(preprocessed_image)  # Get model predictions
    predicted_class_idx = np.argmax(prediction)  # Find the index of the class with the highest probability
    return predicted_class_idx, prediction  # Return the predicted class index and prediction probabilities

# Open the webcam (camera index 0)
cap = cv2.VideoCapture(0)  # Initialize the webcam (index 0 typically refers to the default webcam)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()  # Exit if the webcam cannot be accessed

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        print("Error: Failed to capture image.")
        break  # Exit the loop if there's an error capturing the imagecam

    # Predict gesture from the captured frame
    predicted_class_idx, prediction = predict_gesture(model, frame)  # Get model's prediction

    # Get the predicted class label based on the predicted class index
    predicted_label = gesture_classes[predicted_class_idx]

    # Display the predicted label on the frame (at position (10, 30))
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the webcam feed with the prediction label
    cv2.imshow("Webcam Feed", frame)  # Display the live webcam feed

    # Exit loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows after exiting the loop
cap.release()  # Release the webcam capture
cv2.destroyAllWindows()  # Close any OpenCV windows that were opened
