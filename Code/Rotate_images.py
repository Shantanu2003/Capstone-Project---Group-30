import cv2
import os

def flip_images():
    # Path to the gestures folder
    gest_folder = "gestures"

    # Loop through each category (gesture) in the gestures folder
    for g_id in os.listdir(gest_folder):
        # Full path of the gesture category
        g_path = os.path.join(gest_folder, g_id)

        # Check if the directory exists (it should be a folder containing images)
        if os.path.isdir(g_path):
            # Get all image files in the folder
            image_files = [f for f in os.listdir(g_path) if f.startswith('Image_') and f.endswith('.jpg')]

            # Process each image file
            for image_file in image_files:
                # Construct the full path of the image
                path = os.path.join(g_path, image_file)

                # Check if the image exists before proceeding
                if os.path.exists(path):
                    # Construct the new image path with an offset of 1200
                    new_image_name = f"{image_file.split('.')[0]}_flipped.jpg"
                    new_path = os.path.join(g_path, new_image_name)

                    # Read the image in grayscale
                    img = cv2.imread(path, 0)

                    # Check if the image is read correctly
                    if img is not None:
                        # Flip the image horizontally (1 = horizontal flip)
                        flipped_img = cv2.flip(img, 1)

                        # Save the flipped image
                        cv2.imwrite(new_path, flipped_img)
                        print(f"Flipped and saved image: {new_path}")
                    else:
                        print(f"Error reading image: {path}")
                else:
                    print(f"Image not found: {path}")
        else:
            print(f"Skipping non-directory: {g_id}")

# Call the function to flip the images
flip_images()
