import math

import mediapipe as mp
import cv2
import numpy as np
import pickle


def calculateDistance(landmark1, landmark2):
    x1 = landmark1.x
    y1 = landmark1.y

    x2 = landmark2.x
    y2 = landmark2.y
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def LiveTestingModel(model_dir, num_parameters, extra_parameters):
    print("Testing Live")
    # Load the trained model and scale
    model_data = pickle.load(open(model_dir, 'rb'))
    model = model_data['model']
    scaler = model_data['scaler']  # Load the scaler

    # Initialize MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # video stream
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            # List to store landmark data
            data_aux = []
            x_ = []
            y_ = []
            ret, frame = cap.read()
            # Get the shape of the frame
            height, width, _ = frame.shape

            # Convert the frame from BGR to RGB color space
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Flip the image horizontally for a mirror effect
            image = cv2.flip(image, 1)
            # Set the flag to false before processing
            image.flags.writeable = False
            # Process the image to find hand landmarks
            results = hands.process(image)
            # Set the flag back to true after processing
            image.flags.writeable = True
            # Convert the image back to BGR color space
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render the hand landmarks on the image if any are detected
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                     circle_radius=2), )
                    for i in range(len(hand.landmark)):
                        x = hand.landmark[i].x
                        y = hand.landmark[i].y
                        data_aux.extend([x, y])
                        x_.append(x)
                        y_.append(y)

                # Calculate bounding box coordinates
                x1 = int(min(x_) * width)
                y1 = int(min(y_) * height)
                x2 = int(max(x_) * width)
                y2 = int(max(y_) * height)

                if extra_parameters == True:
                    # get distance between landmark 4 and 10
                    data_aux.append(calculateDistance(hand.landmark[4], hand.landmark[10]))
                    # get distance between landmark 4 and 6
                    data_aux.append(calculateDistance(hand.landmark[4], hand.landmark[6]))

                # Prepare the data for prediction
                X_test = np.asarray(data_aux)
                if X_test.size == num_parameters:
                    # Scale the data
                    X_test_scaled = scaler.transform([X_test])

                    probabilities = model.predict_proba(X_test_scaled)[0]
                    highest_prob_index = np.argmax(probabilities)
                    prediction = model.classes_[highest_prob_index]
                    confidence = probabilities[highest_prob_index]

                    #confidence
                    confidence_percent = confidence * 100
                    print(f"Prediction: {prediction}, Confidence: {confidence_percent:.2f}%")

                    # Draw the bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(image, f"{prediction} ({confidence_percent:.2f}%)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.3, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, "Press Q for exit", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            cv2.imshow('Hand Tracking', image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
