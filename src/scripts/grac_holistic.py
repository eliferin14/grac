import cv2
import argparse
import mediapipe as mp
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.pose import PoseLandmark
from copy import deepcopy

from fps_counter import FPS_Counter
from custom_model.custom_model_wrapper import CustomGestureRecognizer
from gesture_transition_manager import GestureTransitionManager

# Command line arguments
parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("--camera_id", type=int, default=0, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
parser.add_argument("-grmp", "--gesture_recognizer_model_path", type=str, default="custom_model/model/gesture_classifier.tflite", help="Path to the gesture recognition model")
parser.add_argument("-glp", "--gesture_list_path", type=str, default="custom_model/gesture_list.csv", help="Path to the gesture list file")
parser.add_argument("-mgdc", "--minimum_gesture_detection_confidence", type=float, default=0.5, help="Minimum confidence for a gesture to be accepted")
parser.add_argument("-gtt", "--gesture_transition_timer", type=float, default=0.5, help="Timer required for a new grsture to be registered")
parser.add_argument("--draw_landmarks", type=bool, default=True, help="If true draw the landmarks on the output frame")

# Holistic model solution
mp_holistic = mp.solutions.holistic

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
body_color = (0,255,0)
body_drawing_specs = DrawingSpec(body_color)





class GRAC():
    """Wrapper class that combines holistic landmark detection and gestire recognition for hands
    """    ''''''
    
    def __init__(self, model_path, gesture_list_path, mgdc):
        
        # Create a holistic model
        self.holistic = mp_holistic.Holistic(
                refine_face_landmarks = False,
                static_image_mode = False,
            )
        
        # Create the gesture recognition model
        self.cgr = CustomGestureRecognizer(
            model_path=model_path,
            gesture_list_path=gesture_list_path,
            minimum_gesture_detection_confidence=mgdc
            )
        
        # Initialise class fields
        self.holistic_landmarks = None
        
        
        
        
    def detect(self, frame):
        """Call the holistic landmark detection

        Args:
            frame (opencv image): frame where the models will look for hands and pose
        """        ''''''
        
        if frame is None:
            self.logger.warning("Empty frame received")
            return
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        self.holistic_landmarks = self.holistic.process(rgb_frame)
        
        
    
    
    def recognize(self):        
        """Call the custom gesture recognition model

        Returns:
            [str, str]: Labels of the recognized gestures: [left, right]
        """        ''''''
        if self.holistic_landmarks is None:
            return
        
        rhg_id = self.cgr.recognize(self.holistic_landmarks.right_hand_landmarks, 'Right')
        self.rhg = self.cgr.get_gesture_name(rhg_id)
        
        lhg_id = self.cgr.recognize(self.holistic_landmarks.left_hand_landmarks, 'Left')
        self.lhg = self.cgr.get_gesture_name(lhg_id)
        
        #print(f"Left: {self.lhg}\tRight: {self.rhg}")
        
        return self.lhg, self.rhg
        
        
        
        
    def get_gestures(self):
        """Returns the last recognized gestures

        Returns:
            [str, str]: Labels of the recognized gestures: [left, right]
        """        ''''''
        return self.lhg, self.rhg        
        
        
        
    
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
            mp.solutions.drawing_utils.draw_landmarks(frame, filtered_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=body_drawing_specs)
        if self.holistic_landmarks.face_landmarks:
            #mp.solutions.drawing_utils.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            pass
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
        model_path=args.gesture_recognizer_model_path,
        gesture_list_path=args.gesture_list_path,
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
        grac.detect(frame)
        
        # Recognize gesture
        grac.recognize()
        rhg, lhg = grac.get_gestures()        
        
        # Filter gestures
        rht, filtered_rhg = rightGTR.gesture_change_request(rhg)
        lht, filtered_lhg = leftGTR.gesture_change_request(lhg)
        
        # Draw hands and pose
        if args.draw_landmarks:
            grac.draw_results(frame)   
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