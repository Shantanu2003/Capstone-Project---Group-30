import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os


# Function to load images and their corresponding labels
def pickle_images_labels():
    images_labels = []  # List to store images and their corresponding labels
    # Glob pattern to fetch all .jpg files from subfolders under 'gestures'
    images = glob("gestures/*/*.jpg")
    images.sort()  # Sort the images to ensure they are in order

    # Debugging: Print out the found images to ensure the paths are correct
    print(f"Found {len(images)} images in the folder.")

    # Loop over each image in the directory
    for image in images:
        print(f"Processing image: {image}")

        # Extract the label by getting the parent folder name, which is the gesture category
        label = image.split(os.sep)[-2]  # Get the folder name, which is the label

        # Debugging: Print out the label to ensure it's extracted correctly
        print(f"Extracted label: {label}")

        # Read the image in grayscale (if it's a skeleton landmark image, this is common)
        img = cv2.imread(image, 0)

        if img is not None:  # Check if the image is loaded properly
            # Append the image and its corresponding label to the list
            images_labels.append((np.array(img, dtype=np.uint8), label))
        else:
            print(f"Failed to load image at {image}")

    return images_labels


# Load the images and their labels by calling the function
images_labels = pickle_images_labels()

# Debugging: Check if images_labels is populated
if not images_labels:
    print("No images found or loaded. Exiting.")
else:
    print("Images and labels successfully loaded.")

# Shuffle the data to randomize the order
images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))

# Unzip the images and labels into separate lists
images, labels = zip(*images_labels)
print("Length of images_labels", len(images_labels))

# Split the dataset into train, test, and validation sets (5:1:1 ratio)
# Train set will contain 5/6 of the total images
train_images = images[:int(5 / 6 * len(images))]
print("Length of train_images", len(train_images))

# Save the train images to a pickle file
with open("train_images", "wb") as f:
    pickle.dump(train_images, f)
del train_images  # Delete the train_images variable to free memory

# Train labels will be the same length as the train images
train_labels = labels[:int(5 / 6 * len(labels))]
print("Length of train_labels", len(train_labels))

# Save the train labels to a pickle file
with open("train_labels", "wb") as f:
    pickle.dump(train_labels, f)
del train_labels  # Delete the train_labels variable to free memory

# Test set will contain the next 1/6 of the total images (after 5/6 for training)
test_images = images[int(5 / 6 * len(images)):int(11 / 12 * len(images))]
print("Length of test_images", len(test_images))

# Save the test images to a pickle file
with open("test_images", "wb") as f:
    pickle.dump(test_images, f)
del test_images  # Delete the test_images variable to free memory

# Test labels will be the same length as the test images
test_labels = labels[int(5 / 6 * len(labels)):int(11 / 12 * len(images))]
print("Length of test_labels", len(test_labels))

# Save the test labels to a pickle file
with open("test_labels", "wb") as f:
    pickle.dump(test_labels, f)
del test_labels  # Delete the test_labels variable to free memory

# Validation set will contain the remaining 1/12 of the images (after 11/12 for training/testing)
val_images = images[int(11 / 12 * len(images)):]
print("Length of val_images", len(val_images))

# Save the validation images to a pickle file
with open("val_images", "wb") as f:
    pickle.dump(val_images, f)
del val_images  # Delete the val_images variable to free memory

# Validation labels will be the same length as the validation images
val_labels = labels[int(11 / 12 * len(labels)):]
print("Length of val_labels", len(val_labels))

# Save the validation labels to a pickle file
with open("val_labels", "wb") as f:
    pickle.dump(val_labels, f)
del val_labels  # Delete the val_labels variable to free memory
