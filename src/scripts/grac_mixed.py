import cv2
import argparse
import mediapipe as mp
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.pose import PoseLandmark
from copy import deepcopy
import threading
import zipfile

from fps_counter import FPS_Counter
from gesture_transition_manager import GestureTransitionManager

# Command line arguments
parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("--camera_id", type=int, default=0, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
parser.add_argument("-grmd", "--gesture_recognizer_model_directory", type=str, default="training/exported_model", help="Path to the gesture recognition model")
parser.add_argument("-mgdc", "--minimum_gesture_detection_confidence", type=float, default=0.5, help="Minimum confidence for a gesture to be accepted")
parser.add_argument("-gtt", "--gesture_transition_timer", type=float, default=0.5, help="Timer required for a new grsture to be registered")
parser.add_argument("--draw_landmarks", type=bool, default=True, help="If true draw the landmarks on the output frame")

# Hands and pose models packages
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Cosmetics
right_color = (255,0,0) # Blue in BGR
left_color = (0,0,255)  # Red in BGR
rh_drawing_specs = DrawingSpec((0,0,255))   # Blue in RGB
lh_drawing_specs = DrawingSpec((255,0,0))   # Red in RGB

# List of landmarks to exclude from the drawing
excluded_landmarks = [
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    PoseLandmark.LEFT_PINKY,
    PoseLandmark.RIGHT_PINKY,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.RIGHT_THUMB
    ]
pose_color = (0,255,0)
pose_drawing_specs = DrawingSpec(pose_color)





class GRAC():
    
    def __init__(self, model_directory, mgdc):
        
        # Define constants
        self.RIGHT = 0
        self.LEFT = 1
        
        # Allocate the hand landmarker 
        self.hand_landmarker = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        
        # Extract the gesture recognizer .tflite models
        with zipfile.ZipFile(model_directory+"/gesture_recognizer.task", 'r') as zip_ref:
            zip_ref.extractall(model_directory)
        with zipfile.ZipFile(model_directory+"/hand_gesture_recognizer.task", 'r') as zip_ref:
            zip_ref.extractall(model_directory)
        gesture_embedder_path = model_directory+'/gesture_embedder.tflite'
        gesture_classifier_path = model_directory+'/custom_gesture_classifier.tflite'
        
        # Allocate the gesture embedder and its info
        self.gesture_embedder = tf.lite.Interpreter(gesture_embedder_path)
        self.gesture_embedder_input = self.gesture_embedder.get_input_details()
        self.gesture_embedder_output = self.gesture_embedder.get_output_details()
        self.gesture_embedder.allocate_tensors()
        
        # Allcoate the gesture classifier and its info
        self.gesture_classifier = tf.lite.Interpreter(gesture_classifier_path)
        self.gesture_classifier_input = self.gesture_classifier.get_input_details()
        self.gesture_classifier_output = self.gesture_classifier.get_output_details()
        self.gesture_classifier.allocate_tensors()
        
        # Load the labels
        labels_path = model_directory+'/labels.txt'
        file = open(labels_path, 'r')
        labels = file.readlines()
        self.labels = [line.strip() for line in labels]
        
        # Allocate the pose landmarker
        
        # Initialize results to None
        self.right_hand_gesture, self.left_hand_gesture = None, None
        
        return
        
        
        
        
    def process(self, frame, use_threading=False):
        
        if frame is None:
            self.logger.warning("Empty frame received")
            return
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if use_threading:
        
            # Duplicate the frame for the two processes
        
            # Set as write only
        
            # Call the two functions with different threads
        
            # Wait for both functions to return
            pass
            
        else:
            self._process_hands(rgb_frame)
            pass
            
        
        # Save results (?)
        
        return
        
        
        
        
    def _process_hands(self, rgb_frame):
        
        # Call the hand landmarker model
        results = self.hand_landmarker.process(rgb_frame)
        
        # Process the results
        if results.multi_hand_landmarks:
            
            for hand_landmarks, hand_world_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks, results.multi_handedness):
                
                # Convert the output of the hands model to tensors
                hand_landmarks_tensor = self._convert_results_to_tensor(hand_landmarks)
                hand_world_landmarks_tensor = self._convert_results_to_tensor(hand_world_landmarks)
                handedness_tensor = np.array([[handedness.classification[0].index]]).astype(np.float32)
                
                # Call the gesture embedder model
                self.gesture_embedder.set_tensor(self.gesture_embedder_input[0]['index'], hand_landmarks_tensor)
                self.gesture_embedder.set_tensor(self.gesture_embedder_input[1]['index'], handedness_tensor)
                self.gesture_embedder.set_tensor(self.gesture_embedder_input[2]['index'], hand_world_landmarks_tensor)
                
                self.gesture_embedder.invoke()
                
                embedded_gesture = self.gesture_embedder.get_tensor(self.gesture_embedder_output[0]['index'])
                
                # Call the gesture classifier model                
                self.gesture_classifier.set_tensor(self.gesture_classifier_input[0]['index'], embedded_gesture)
                
                self.gesture_classifier.invoke()
                
                gestures_likelihood = self.gesture_classifier.get_tensor(self.gesture_classifier_output[0]['index'])
                
                # Select the most likely gesture
                gesture_id = np.argmax(gestures_likelihood)
                gesture = self.labels[gesture_id]
                
                # Save the gesture to the appropriate varaible
                if handedness.classification[0].index == self.RIGHT:
                    self.right_hand_gesture = gesture
                else:
                    self.left_hand_gesture = gesture
                    
        else:
            self.right_hand_gesture, self.left_hand_gesture = None, None
        
        return self.right_hand_gesture, self.left_hand_gesture
    
    
    
    
    def _process_pose(self, rgb_frame):
        return
        
    
    
    
    def _detect_hands(self, rgb_frame):
        
        # Call the hand landmarker to detect hands
        
        # Process the results
        
        # Save to class fields
        
        return
    
    
    
    def _recognize_gestures(self, hand_landmarks, hand_world_landmarks, handedness):
        
        # Prepare the data for the model
        
        # Call the model
        
        # Select the most likely gesture
        
        return
    
    
    
    
    def _detect_pose(self, rgb_frame):
        
        # Call the pose landmarker
        
        # Process the results
        
        # Save to class fields
        
        return 
    
    
    
    
    def _convert_results_to_tensor(self, landmarks):
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
    
    
    
    
    def get_hand_gestures(self):
        return self.right_hand_gesture, self.left_hand_gesture
        
        
        
    
    def draw_results(self, frame):
        """Draw the hand and pose landmarks in the provided frame

        Args:
            frame (opencv_image): Frame where the landmarks are drawn
        """        ''''''
        if self.holistic_landmarks is None:
            return
        
        # Remove unwanted landmarks from the drawing process
        # Since a change of the properties of the landmark is required, create an independent copy of the landmark set
        filtered_landmarks = deepcopy(self.holistic_landmarks.pose_landmarks)
        for idx, landmark in enumerate(filtered_landmarks.landmark):
            if idx in excluded_landmarks:
                landmark.presence = 0
        
        # Draw pose, face, and hands landmarks on the frame
        if self.holistic_landmarks.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, filtered_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=pose_drawing_specs)
        if self.holistic_landmarks.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, self.holistic_landmarks.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=lh_drawing_specs)
        if self.holistic_landmarks.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, self.holistic_landmarks.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=rh_drawing_specs)
    
    
    
    
    def add_text(self, frame, text_list, row_height):
        """Adds text to a frame. 
        The text is passed as a list of frame_text (which is a named tuple)

        Args:
            frame (opencv image): frame where to add the text
            text_list (list[frame_text]): Set of text to be added
            row_height (int): height of each row (pixels) 
        """        ''''''
        
        for i, frame_text in enumerate(text_list):
            
            # If the value is a float, truncate to 2 decimals
            value_str = f"{frame_text.value:.2f}" if type(frame_text.value) == float else f"{frame_text.value}"
            
            # If the name is None, do not use the :
            name_str = "" if frame_text.name is None else f"{frame_text.name}: "
            
            # Draw the text on the frame
            frame = cv2.putText(frame, name_str+value_str, org=(25, (i+1)*row_height), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=frame_text.color, thickness=2, lineType=cv2.LINE_AA)
            
        
    
    
    
    
    
    
    
    

if __name__ == "__main__":    
    
    args = parser.parse_args()
    print(args)
    
    grac = GRAC(
        model_directory=args.gesture_recognizer_model_directory,
        mgdc=args.minimum_gesture_detection_confidence 
    )
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(args.camera_id)
    assert cam.isOpened()
    
    # Fps counter
    fps_counter = FPS_Counter()
    
    # Gesture managers
    rightGTR = GestureTransitionManager(transition_timer=args.gesture_transition_timer)
    leftGTR = GestureTransitionManager(transition_timer=args.gesture_transition_timer)
    
    # Named tuple for text to print on image
    frame_text = namedtuple('FrameText', ['name', 'value', 'color'])
    
    frame_number = 0
    rhg, lhg = 0, 0
    rhw_posList = []
    
    
    
    # Loop 
    while cam.isOpened():
        # Update fps
        fps = fps_counter.get_fps()
        
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            continue        
        
        # Flip image horizontally
        frame = cv2.flip(frame, 1) 
        
        # Detect landmarks
        grac.process(frame)    
        rhg, lhg = grac.get_hand_gestures()
        
        # Filter gestures
        rht, filtered_rhg = rightGTR.gesture_change_request(rhg)
        lht, filtered_lhg = leftGTR.gesture_change_request(lhg)
        
        # Draw hands and pose
        if args.draw_landmarks:
            pass
            #grac.draw_results(frame)   
        # 3D plot of hands
        #grac.mpgr.plot_hands_3d()  
        
        # Add info as text        
        text_list = []
        text_list.append(frame_text('FPS', fps, (0,255,0)))
        if rhg is not None: text_list.append(frame_text('Right', filtered_rhg, right_color))
        if lhg is not None: text_list.append(frame_text('Left', filtered_lhg, left_color))
        grac.add_text(frame, text_list, row_height=30)
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
    # Release the camera
    cam.release()
    cv2.destroyAllWindows()