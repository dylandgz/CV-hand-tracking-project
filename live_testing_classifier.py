# import pickle
#
# import cv2
# import mediapipe as mp
# import numpy as np
#
# # model_dict = pickle.load(open('./model.p', 'rb'))
# # model = model_dict['model']
#
# cap = cv2.VideoCapture(0)
#
# # Importing MediaPipe solutions for drawing utilities, hands tracking and drawing styles
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Creating a MediaPipe Hands object for hand tracking with specified configurations
# hands = mp_hands.Hands(static_image_mode=0.8, min_detection_confidence=0.2)
#
# while True:
#     ret, frame = cap.read()
#
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Processing the image to find hand landmarks
#     results = hands.process(frame_rgb)
#     # Checking if any hand landmarks are found
#     # RGB 2 BGR
#     frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
#
#     if results.multi_hand_landmarks:
#         # Drawing hand landmarks on the image
#         for num, hand in enumerate(results.multi_hand_landmarks):
#             mp_drawing.draw_landmarks(frame_rgb, hand, mp_hands.HAND_CONNECTIONS,
#                                       mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                                       mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
#                                       )
#
#     cv2.imshow('frame', frame)
#     cv2.waitKey(25)
#
# cap.release()
# cvv2.destroyAllWindows()


import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import pickle

model = pickle.load(open('./model.pickle', 'rb'))
model = model['model']

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        H, W, _ = frame.shape

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        # print(results)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)
                for i in range(len(hand.landmark)):
                    x = hand.landmark[i].x
                    y = hand.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)  # Use max for the second point
            y2 = int(max(y_) * H)  # Use max for the second point
            X_test = np.asarray(data_aux)

            if X_test.size == 42:
                prediction = model.predict([X_test])
                print(prediction)
                # Draw the rectangle using the corrected coordinates
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(image, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
