# Project Description
# Hand tracking for sign language using Classification ML

import os
import mediapipe as mp
import cv2
import pickle
import matplotlib.pyplot as plt


# Initialize MediaPipe solutions for drawing utilities, hands tracking, and drawing styles
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Create a MediaPipe Hands object with specific configurations for hand tracking
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

# Path to the training dataset
KAGGLE_DATASET_TRAIN_DIR = '/path/to/dataset'

# Initialize lists to store data and labels
data = []
labels = []

# Iterate over each directory in the dataset (each representing a different sign)
for dir_ in os.listdir(KAGGLE_DATASET_TRAIN_DIR):
    dir_path = os.path.join(KAGGLE_DATASET_TRAIN_DIR, dir_)

    # Check if the path is a directory
    if os.path.isdir(dir_path):
        counter = 0  # Counter to limit the number of images processed per sign

        # Process only the first 750 images in each sub-directory
        for img_path in os.listdir(dir_path):
            temp_data = []  # Temporary list to store landmark data

            # Read the image using OpenCV
            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Process the image to find hand landmarks
            results = hands.process(img_rgb)

            # Check if any hand landmarks are found
            if results.multi_hand_landmarks and counter < 750:
                counter += 1

                # Iterate over each detected hand
                for num, hand in enumerate(results.multi_hand_landmarks):
                    for i in range(len(hand.landmark)):
                        x = hand.landmark[i].x
                        y = hand.landmark[i].y
                        temp_data.extend([x, y])  # Append landmark coordinates to temp_data

                    # Draw hand landmarks on the image
                    mp_drawing.draw_landmarks(img_rgb, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(), mp_drawing.DrawingSpec())

                # Append the processed data and label if the landmark count is within the expected range
                if len(temp_data) <= 42:
                    data.append(temp_data)
                    labels.append(os.path.basename(dir_))
                    # # Displaying the processed image with drawn landmarks
                    # plt.imshow(img_rgb)
                    # # Showing the plot
                    # plt.show()

            # Break the loop once 750 images have been processed
            if counter >= 750:
                break

# Serialize and save the processed data and labels
with open("data.pickle", "wb") as file:
    pickle.dump({"data": data, "labels": labels}, file)
