import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure TensorFlow uses only the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_image_size():
    """
    Dynamically determine the size of the images in the gestures dataset.
    Assumes that the images in the `gestures` directory are uniform in size.
    """
    gesture_dirs = glob('gestures/*')  # Get all subdirectories in 'gestures'
    if not gesture_dirs:
        raise FileNotFoundError("No gesture directories found in 'gestures'. Ensure the dataset is correctly arranged.")

    # Look for the first valid image file in any subdirectory
    for gesture_dir in gesture_dirs:
        image_files = glob(f'{gesture_dir}/*.jpg')  # Change to your file format if not .jpg
        if image_files:
            img = cv2.imread(image_files[0], 0)  # Read in grayscale
            if img is None:
                raise FileNotFoundError(f"Cannot load image {image_files[0]}. Check file integrity.")
            return img.shape

    raise FileNotFoundError("No valid images found in the 'gestures' dataset.")


def get_num_of_classes():
    """
    Count the number of gesture classes based on subdirectories in `gestures`.
    """
    gesture_dirs = glob('gestures/*')
    if not gesture_dirs:
        raise FileNotFoundError("No gesture directories found in 'gestures'.")
    return len(gesture_dirs)


def load_dataset(image_x, image_y):
    """
    Load images and labels from the `gestures` directory.
    Preprocess the data and return training and validation sets.
    """
    images = []
    labels = []
    gesture_dirs = glob('gestures/*')

    for idx, gesture_dir in enumerate(gesture_dirs):
        image_files = glob(f'{gesture_dir}/*.jpg')  # Change to your file format if needed
        for image_file in image_files:
            img = cv2.imread(image_file, 0)  # Load in grayscale
            if img is not None:
                img = cv2.resize(img, (image_x, image_y))  # Ensure consistent size
                images.append(img)
                labels.append(idx)  # Use the directory index as the label

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int32")

    # Normalize images and one-hot encode labels
    images = images / 255.0  # Normalize pixel values
    images = np.reshape(images, (images.shape[0], image_x, image_y, 1))  # Add channel dimension
    labels = to_categorical(labels, num_classes=len(gesture_dirs))

    # Split into training and validation sets (80-20 split)
    split_idx = int(0.8 * len(images))
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    return train_images, train_labels, val_images, val_labels


def cnn_model(image_x, image_y, num_classes):
    """
    Define and compile the CNN model.
    """
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train():
    """
    Train the CNN model using the gestures dataset.
    """
    try:
        image_x, image_y = get_image_size()
        num_classes = get_num_of_classes()

        print(f"Image size: {image_x}x{image_y}")
        print(f"Number of gesture classes: {num_classes}")

        # Load the dataset
        train_images, train_labels, val_images, val_labels = load_dataset(image_x, image_y)

        # Create the CNN model
        model = cnn_model(image_x, image_y, num_classes)

        # Data augmentation setup for better generalization
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(train_images)

        # Define the checkpoint callback to save the best model
        filepath = "cnn_model_best.keras"  # Updated file extension
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        callbacks_list = [checkpoint, early_stop]

        # Train the model
        model.fit(
            datagen.flow(train_images, train_labels, batch_size=64),
            validation_data=(val_images, val_labels),
            epochs=15,
            callbacks=callbacks_list
        )

        # Evaluate the model
        scores = model.evaluate(val_images, val_labels, verbose=0)
        print(f"CNN Error: {100 - scores[1] * 100:.2f}%")

        # Clear the session to avoid memory issues
        K.clear_session()

    except Exception as e:
        print(f"Error during training: {e}")


# Run the training process
if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Error: {e}")
