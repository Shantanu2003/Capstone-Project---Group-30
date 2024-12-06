import cv2
import numpy as np
import pickle, os, sqlite3, random

# Define image dimensions for resizing
image_x, image_y = 50, 50


# Function to load the precomputed hand histogram (for gesture recognition)
def get_hand_hist():
    try:
        # Open and load the histogram data for gesture recognition
        with open(
                "C:/Users/Shantanu/Capstone Project/Code/hist",
                "rb") as f:
            hist = pickle.load(f)
        return hist
    except FileNotFoundError:
        print("Error: 'hist' file not found. Ensure you have created the histogram file.")
        exit()  # Exit if histogram file not found


# Function to initialize and create necessary folders and database
def init_create_folder_database():
    # Create 'gestures' folder if it doesn't exist
    if not os.path.exists("gestures"):
        os.mkdir("gestures")

    # Create the 'gesture_db.db' database if it doesn't exist
    if not os.path.exists("C:/Users/Shantanu/Capstone Project/Code/gesture_db.db"):
        conn = sqlite3.connect("C:/Users/Shantanu/Capstone Project/Code/gesture_db.db")
        # Create a table for storing gesture data
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()


# Function to create a folder for storing gesture images
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


# Function to store gesture information into the database
def store_in_db(g_id, g_name):
    conn = sqlite3.connect("C:/Users/Shantanu/Capstone Project/Code/gesture_db.db")
    # Insert gesture record into the database
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (g_id, g_name)
    try:
        conn.execute(cmd)
    except sqlite3.IntegrityError:
        # If the g_id already exists, give the option to update it
        choice = input("g_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
            conn.execute(cmd)
        else:
            print("Doing nothing...")
            return  # If user chooses not to update, exit the function
    conn.commit()


# Function to retrieve all gesture data from the database
def get_gestures_from_db():
    conn = sqlite3.connect("C:/Users/Shantanu/Capstone Project/Code/gesture_db.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM gesture")
    rows = cursor.fetchall()  # Fetch all records
    return rows


# Function to store images of gestures in a folder
def store_images(g_id):
    total_pics = 1200  # Total number of pictures to capture
    hist = get_hand_hist()  # Load the hand histogram for recognition
    cam = cv2.VideoCapture(0)  # Use the default camera (index 0)

    # Check if the camera is accessible
    if cam.read()[0] == False:
        print("Camera not found. Exiting...")
        exit()  # Exit if camera is not available

    # Define region for gesture capturing
    x, y, w, h = 300, 100, 300, 300

    # Create folder to store images if it doesn't exist
    create_folder("gestures/" + str(g_id))

    pic_no = 0  # Initialize picture number counter
    flag_start_capturing = False  # Flag to start/stop capturing images
    frames = 0  # Frame counter

    while True:
        # Capture a frame from the webcam
        img = cam.read()[1]
        img = cv2.flip(img, 1)  # Flip the image horizontally (mirror effect)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

        # Back project the histogram to the current frame
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # Disc kernel for morphological transformation
        cv2.filter2D(dst, -1, disc, dst)  # Apply filter to enhance the image

        blur = cv2.GaussianBlur(dst, (11, 11), 0)  # Gaussian blur to smooth the image
        blur = cv2.medianBlur(blur, 15)  # Median blur to remove noise

        # Apply thresholding to segment the hand gesture
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh, thresh, thresh))  # Merge into 3 channels
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        thresh = thresh[y:y + h, x:x + w]  # Crop the region of interest

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:  # If contours are found
            contour = max(contours, key=cv2.contourArea)  # Get the largest contour
            if cv2.contourArea(contour) > 10000 and frames > 50:  # Only consider significant contours
                x1, y1, w1, h1 = cv2.boundingRect(contour)  # Get bounding box for the contour
                pic_no += 1  # Increment the picture number
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]  # Crop the hand region
                # Pad image to make it square if necessary
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.resize(save_img, (image_x, image_y))  # Resize to fixed size
                rand = random.randint(0, 10)
                if rand % 2 == 0:  # Randomly flip the image for data augmentation
                    save_img = cv2.flip(save_img, 1)
                # Display capturing message
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                # Save the captured image
                cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)

        # Draw a rectangle around the region of interest
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", img)  # Show the camera feed
        cv2.imshow("thresh", thresh)  # Show the thresholded image

        # Listen for keypresses to control the capturing process
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):  # Toggle the start/stop capturing
            if flag_start_capturing == False:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing == True:
            frames += 1
        if pic_no == total_pics:  # Stop once enough pictures are captured
            break


# Function to update or create new gesture records in the database
def update_or_create_gesture():
    # Get user input for gesture ID and name
    g_id = input("Enter gesture no.: ")
    g_name = input("Enter gesture name/text: ")

    # Retrieve all existing gestures from the database
    existing_gestures = get_gestures_from_db()
    # Check if the gesture ID already exists
    if any(str(g[0]) == g_id for g in existing_gestures):
        choice = input(f"Gesture ID {g_id} already exists. Do you want to update the gesture name? (y/n): ")
        if choice.lower() == 'y':
            store_in_db(g_id, g_name)  # Update the gesture record
    else:
        store_in_db(g_id, g_name)  # Add a new gesture record

    store_images(g_id)  # Store images for the new gesture


# Start the process
init_create_folder_database()  # Initialize folders and database
update_or_create_gesture()  # Add or update gesture data and images
