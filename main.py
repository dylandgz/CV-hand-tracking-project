# Project Description
# Hand tracking for sign language using Classification ML

import os
import mediapipe as mp
import cv2
import pickle
import matplotlib.pyplot as plt  # Corrected import

# Importing MediaPipe solutions for drawing utilities, hands tracking and drawing styles
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Creating a MediaPipe Hands object for hand tracking with specified configurations
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

# Path to the training dataset
KAGGLE_DATASET_TRAIN_DIR = '/Users/dylandominguez/PycharmProjects/CV-hand-tracking-project/ASL_alphabet_dataset/asl_alphabet_train/asl_alphabet_train'
data = []
labels = []
# Iterating over each directory in the dataset
for dir_ in os.listdir(KAGGLE_DATASET_TRAIN_DIR):
    # Constructing the full path of the sub-directory
    dir_path = os.path.join(KAGGLE_DATASET_TRAIN_DIR, dir_)
    # print(dir_path)
    # Checking if the path is indeed a directory
    if os.path.isdir(dir_path):
        # Processing only the first image in each sub-directory
        for img_path in os.listdir(dir_path)[:1]:
            temp_data = []
            # Reading the image using OpenCV
            img = cv2.imread(os.path.join(dir_path, img_path))
            # Converting the image from BGR to RGB format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Processing the image to find hand landmarks
            results = hands.process(img_rgb)
            # Checking if any hand landmarks are found
            if results.multi_hand_landmarks:
                # Iterating over each detected hand
                for num, hand in enumerate(results.multi_hand_landmarks):
                    for i in range(len(hand.landmark)):
                        # print(hand.landmark[i])
                        x = hand.landmark[i].x
                        y = hand.landmark[i].y
                        temp_data.append(x)
                        temp_data.append(y)
                    # Drawing hand landmarks on the image
                    mp_drawing.draw_landmarks(img_rgb, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(),
                                              mp_drawing.DrawingSpec())
            data.append(temp_data)
            print(os.path.basename(dir_))
            labels.append(os.path.basename(dir_))

            file = open("data.pickle", "wb")
            pickle.dump({"data": data, "labels": labels}, file)
            file.close()

            # Displaying the processed image with drawn landmarks
            # plt.imshow(img_rgb)
            # Showing the plot
            # plt.show()
