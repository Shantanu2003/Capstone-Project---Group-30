import cv2
import os
import numpy as np
import random

# Function to get the image size of the first image
def get_image_size(sample_folder):
    image_files = [f for f in os.listdir(sample_folder) if f.endswith('.jpg')]
    if not image_files:
        return None
    img = cv2.imread(os.path.join(sample_folder, image_files[0]), 0)
    return img.shape if img is not None else None

# Main function
gestures_path = 'gestures'
gesture_folders = [d for d in os.listdir(gestures_path) if os.path.isdir(os.path.join(gestures_path, d))]
gesture_folders.sort()

image_size = get_image_size(os.path.join(gestures_path, gesture_folders[0]))
if not image_size:
    print("No valid images found in the dataset.")
    exit()

image_x, image_y = image_size
full_img = None

for gesture_folder in gesture_folders:
    gesture_path = os.path.join(gestures_path, gesture_folder)
    image_files = [f for f in os.listdir(gesture_path) if f.endswith('.jpg')]

    if not image_files:
        print(f"No images found in {gesture_folder}.")
        continue

    # Select a random image from available files
    random_image = random.choice(image_files)
    img_path = os.path.join(gesture_path, random_image)
    img = cv2.imread(img_path, 0)

    if img is None:
        print(f"Failed to load image: {img_path}")
        continue

    if full_img is None:
        full_img = img
    else:
        # Stack horizontally or vertically as needed
        full_img = np.vstack((full_img, img)) if full_img.shape[1] == img.shape[1] else np.hstack((full_img, img))

if full_img is not None and full_img.size > 0:
    cv2.imshow("gestures", full_img)
    cv2.imwrite('full_img.jpg', full_img)
    cv2.waitKey(0)
else:
    print("No valid images to display.")
