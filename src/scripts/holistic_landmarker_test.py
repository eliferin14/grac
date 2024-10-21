# Import necessary libraries
import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    refine_face_landmarks = False,
    static_image_mode = False,
)

# Open a webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frames from the webcam
    ret, frame = cap.read()
    if not ret:
        print("frames not found")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the Holistic model
    results = holistic.process(rgb_frame)

    # Draw pose, face, and hands landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.face_landmarks:
        #mp.solutions.drawing_utils.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        pass
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Display the annotated frame
    cv2.imshow("output", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()