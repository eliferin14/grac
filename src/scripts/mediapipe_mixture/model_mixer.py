import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Load the gesture embedder model
gesture_embedder = tf.lite.Interpreter('gesture_embedder.tflite')
ge_input = gesture_embedder.get_input_details()
ge_output = gesture_embedder.get_output_details()
gesture_embedder.allocate_tensors()

# Load the gesture_classifier model
gesture_classifier = tf.lite.Interpreter('custom_gesture_classifier.tflite')
gc_input = gesture_classifier.get_input_details()
gc_output = gesture_classifier.get_output_details()
gesture_classifier.allocate_tensors()
 
if True:
    print(f"\nEmbedder input: {ge_input}")
    print(f"\nEmbedder output: {ge_output}")
    print(f"\nClassifier input: {gc_input}")
    print(f"\nClassifier output: {gc_output}")
    
    
# Define a function to convert landmarks to numpy arrays
def convert_to_numpy(landmarks):
    """Converts the landmark list to a numpy matrix of 3d points

    Args:
        hand_world_landmarks (_type_): result of the hand recognition process
    """        ''''''
    hand_landmarks_matrix = np.zeros((1,21,3))
    i = 0
    for landmark in landmarks.landmark:
        hand_landmarks_matrix[0,i,0] = landmark.x
        hand_landmarks_matrix[0,i,1] = landmark.y
        hand_landmarks_matrix[0,i,2] = landmark.z
        i+=1
        
    return hand_landmarks_matrix.astype(np.float32)
    
    
    
    
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:
    
    while cap.isOpened():
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
            for hand_landmarks, hand_world_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks, results.multi_handedness):
                # Convert to numpy arrays
                hand_landmarks_np = convert_to_numpy(hand_landmarks)
                assert hand_landmarks_np.shape == (1,21,3)
                hand_world_landmarks_np = convert_to_numpy(hand_world_landmarks)
                assert hand_world_landmarks_np.shape == (1,21,3)
                handedness_np = np.array([[handedness.classification[0].index]]).astype(np.float32)
                assert handedness_np.shape == (1,1)
                
                # Gesture embedder
                gesture_embedder.set_tensor(ge_input[0]['index'], hand_landmarks_np)
                gesture_embedder.set_tensor(ge_input[1]['index'], handedness_np)
                gesture_embedder.set_tensor(ge_input[2]['index'], hand_world_landmarks_np)
                
                gesture_embedder.invoke()
                
                embedded_gesture = gesture_embedder.get_tensor(ge_output[0]['index'])
                #print(embedded_gesture)
                
                # Gesture classifier
                gesture_classifier.set_tensor(gc_input[0]['index'], embedded_gesture)
                
                gesture_classifier.invoke()
                
                gestures_likelihood = gesture_classifier.get_tensor(gc_output[0]['index'])
                print(gestures_likelihood)
                
                
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()