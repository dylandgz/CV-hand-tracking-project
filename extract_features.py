import math
import os
import mediapipe as mp
import cv2
import pickle

#calclate landmark distance using distance formula
def calculateDistance(landmark1, landmark2):
    x1 = landmark1.x
    y1 = landmark1.y
    x2 = landmark2.x
    y2 = landmark2.y
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def featureExtractionCoordinates(num_instances_per_sign, num_parameters, confidence, data_dir, clean_data_dir,
                                 extra_parameters):
    print("Extracting features from images")
    # Initialize MediaPipe and its functions
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # number of instances control the training size
    instance_counter = 1
    total_instances = num_instances_per_sign * 25

    # Create a MediaPipe Hands object
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=confidence)

    # Initialize lists to store data and labels
    data = []
    labels = []

    # Iterate over each directory in the file directory
    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)

        # make sure that the path is valid
        if os.path.isdir(dir_path):
            # Counter to limit the number of images processed per hand sign
            per_sign_instance_counter = 0

            for img_path in os.listdir(dir_path):
                temp_data = []  # Temporary list to store landmark data

                # Read the image using OpenCV
                img = cv2.imread(os.path.join(dir_path, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

                # Process the image to find hand landmarks
                results = hands.process(img_rgb)

                # Check if any hand landmarks are found and we are still within the number of instances
                if results.multi_hand_landmarks and per_sign_instance_counter < num_instances_per_sign:

                    # Iterate over each landmarks
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        for i in range(len(hand.landmark)):
                            x = hand.landmark[i].x
                            y = hand.landmark[i].y
                            # Append landmark coordinates to temp_data
                            temp_data.extend([x, y])

                        # Draw hand landmarks on the image
                        mp_drawing.draw_landmarks(img_rgb, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(),
                                                  mp_drawing.DrawingSpec())
                    if extra_parameters == True:
                        # get distance between landmark 4 and 10
                        temp_data.append(calculateDistance(hand.landmark[4], hand.landmark[10]))
                        # get distance between landmark 4 and 6
                        temp_data.append(calculateDistance(hand.landmark[4], hand.landmark[6]))

                    # Append the processed data and label if the landmark count otherwise do not use
                    # some images may not have all landmarks
                    if len(temp_data) == num_parameters:
                        per_sign_instance_counter += 1
                        data.append(temp_data)
                        labels.append(os.path.basename(dir_))
                        print(f"{instance_counter}/{total_instances} {(instance_counter / total_instances) * 100:.2f}%")
                        instance_counter += 1

                if per_sign_instance_counter >= num_instances_per_sign:
                    break

    # Save the processed data and labels
    with open(f"{clean_data_dir}", "wb") as file:
        pickle.dump({"data": data, "labels": labels}, file)