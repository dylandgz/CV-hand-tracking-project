import mediapipe as mp
import cv2
import os


def createData(num_data_frames, dataset_name):
    classes = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
        10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
        19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
    }

    if not os.path.exists(dataset_name):
        os.mkdir(dataset_name)

    number_of_classes = 24
    dataset_size = num_data_frames

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        for i in range(number_of_classes):
            # Create a directory for each class
            class_path = os.path.join(dataset_name, str(classes[i]))
            if not os.path.exists(class_path):
                os.mkdir(class_path)

            # Wait for 'Q' press to start capturing for the class
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f'Class {classes[i]}: Press "Q" to start capturing', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Hand Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            counter = 0
            while counter < dataset_size:
                ret, frame = cap.read()
                if not ret:
                    continue  # Skip the frame if not captured properly

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                print(f'Capture class {classes[i]} {(counter/dataset_size)*100}%')

                if results.multi_hand_landmarks:
                    for hand in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                        )

                cv2.imshow('Hand Tracking', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Allow early exit if 'q' is pressed

                filename = '{}{}.jpg'.format(classes[i], counter)
                cv2.imwrite(os.path.join(class_path, filename), frame)
                counter += 1

    cap.release()
    cv2.destroyAllWindows()


createData(500, 'created_ASL_dataset_right_hand')
