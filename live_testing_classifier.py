import mediapipe as mp
import cv2
import numpy as np
import pickle

def LiveTestingModel():
    print("Testing Live")
    # Load the trained model from the pickle file
    model = pickle.load(open('./model.pickle', 'rb'))
    model = model['model']

    # Initialize MediaPipe solutions for drawing utilities and hand tracking
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Hands object
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            data_aux = []  # List to store landmark data
            x_ = []  # List to store x-coordinates of landmarks
            y_ = []  # List to store y-coordinates of landmarks
            ret, frame = cap.read()
            H, W, _ = frame.shape  # Get the shape of the frame

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
                x1 = int(min(x_) * W)
                y1 = int(min(y_) * H)
                x2 = int(max(x_) * W)
                y2 = int(max(y_) * H)

                # Prepare the data for prediction
                X_test = np.asarray(data_aux)
                if X_test.size == 42:
                    probabilities = model.predict_proba([X_test])[0]
                    highest_prob_index = np.argmax(probabilities)
                    prediction = model.classes_[highest_prob_index]
                    confidence = probabilities[highest_prob_index]

                    # Format confidence as a percentage
                    confidence_percent = confidence * 100
                    print(f"Prediction: {prediction}, Confidence: {confidence_percent:.2f}%")

                    # Draw the bounding box and display the prediction with confidence
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(image, f"{prediction} ({confidence_percent:.2f}%)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Display the image in a window titled 'Hand Tracking'
            cv2.imshow('Hand Tracking', image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()