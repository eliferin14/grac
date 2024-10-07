import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

# Some parameters for the model (default ones)
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode         # I want the live feed options -> for continuous videos

# Default model path
default_model_path = 'gesture_recognizer.task'

# Define styles
right_hand_drawing_specs = mp.solutions.drawing_styles.DrawingSpec(
    color=(255, 0, 0)
)

left_hand_drawing_specs = mp.solutions.drawing_styles.DrawingSpec(
    color=(0,0,255)
)

# Hand connections: pair of landmarks where a line is present
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), 
    (13, 14), (14, 15), (15, 16), 
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# Gesture recognizer class
class Mediapipe_GestureRecognizer():
    
    def __init__(self, 
        model_path = default_model_path, 
        mode = 1, 
        min_hand_detection_confidence = 0.3,
        min_hand_presence_confidence = 0.3,
        min_tracking_confidence = 0.3,
        show_hand_plot = True
    ):
        
        self.logger = logging.getLogger()
        self.logger.info("Creating Mediapipe_GestureRecognizer instance")
        
        # Define a variable that will store the results of gesture recognition
        self.results = None
        
        # Create options object
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path), 
            num_hands=2, 
            min_hand_detection_confidence = min_hand_detection_confidence,
            min_hand_presence_confidence = min_hand_presence_confidence,
            min_tracking_confidence = min_tracking_confidence
            )
        
        # Select running mode
        self.mode = mode
        if self.mode == 2:
            options.running_mode = VisionRunningMode.LIVE_STREAM
            options.result_callback = self.save_result_callback
            self.logger.info("Live stream mode selected")
        elif self.mode == 1:
            options.running_mode = VisionRunningMode.VIDEO
            self.logger.info("Video mode selected")
        elif self.mode == 0:
            options.running_mode = VisionRunningMode.IMAGE
            self.logger.info("Image mode selected")
        else:
            self.logger.error("Invalid mode selected")
            
        # Create the recognizer object
        self.recognizer = GestureRecognizer.create_from_options(options)
        
        # Create plot
        self.show_plot = show_hand_plot
        if self.show_plot:                
            plt.ion()
            self.hands_fig = plt.figure()
            self.left_hand_ax = self.hands_fig.add_subplot(121, projection='3d')
            self.left_hand_ax.set_title("Left hand")
            self.right_hand_ax = self.hands_fig.add_subplot(122, projection='3d')
            self.right_hand_ax.set_title("Right hand")

        
    
    def draw_results_old(self, frame):
        
        if self.results is None:
            return frame
        
        # results is an object with all the data of the hands
        # results.hand_landmarks is a list of sets of landmark. One set per hand detected
        for landmarks in self.results.hand_landmarks:
            # result is a list of landmarks of a single hand
            # Draw the pose landmarks
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([ landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks ])
            
            # Define the styles
            style = solutions.drawing_styles.get_default_hand_landmarks_style()
            
            solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    style
                )
            
        return frame
    
    
    
    def draw_results(self, frame, bb=True):
        
        if self.results is None:
            return frame
        
        # See the bottom of this script to see the structure of the data
        #print(len(self.results.hand_landmarks))
        # NOTE here we use pixel coordinates
        for i, landmarks in enumerate(self.results.hand_landmarks):
            
            handedness = self.results.handedness[i]
            #print(f"i: {i}, handedness: {handedness[0].display_name}")
            
            # Copy the points to a protobuffer
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([ landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks ])
            #print(type(hand_landmarks_proto))
            
            # Specify the style
            style = solutions.drawing_styles.get_default_hand_landmarks_style()
            if handedness[0].index == 0:    # Right hand
                style = right_hand_drawing_specs
            else:
                style = left_hand_drawing_specs
            
            # Draw the landmarks
            solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    style
                )
            
            # Draw the bounding box
            min_x = min([landmark.x for landmark in landmarks])
            max_x = max([landmark.x for landmark in landmarks])
            min_y = min([landmark.y for landmark in landmarks])
            max_y = max([landmark.y for landmark in landmarks])
            
            width = int((max_x - min_x) * frame.shape[1])
            height = int((max_y - min_y) * frame.shape[0])
            
            top_left_x = int(min_x * frame.shape[1])
            top_left_y = int(min_y * frame.shape[0])
            cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), (0, 255, 0), 2)


            
        return frame
            
    def convert_results_to_numpy(self):
        
        if self.results is None:
            return
        
        # Create an empty array
        # 2 hands, 21 points, 3 coordinates
        self.results_numpy = np.zeros(shape=(2, 21, 3), dtype=float)
        
        for i, landmarks in enumerate(self.results.hand_landmarks):
            
            # Check if right or left hand
            handedness = self.results.handedness[i][0].index   # right: 0, left: 1
            #print(handedness)
            
            # Copy each point in the numpy array
            for j, landmark in enumerate(landmarks):
                self.results_numpy[handedness, j, 0] = landmark.x
                self.results_numpy[handedness, j, 1] = landmark.y
                self.results_numpy[handedness, j, 2] = landmark.z               
            
    
    def detect_hands(self, frame, timestamp):
        
        if frame is None:
            self.logger.warning("Empty frame received")
            return
        
        # Choose which function to call based on mode
        if self.mode == 2:
            self.detect_hands_live_stream(frame, timestamp)
        elif self.mode == 1:
            self.detect_hands_video(frame, timestamp)
        elif self.mode == 0:
            self.detect_hands_image(frame)
    
    def detect_hands_live_stream(self, frame, timestamp):
        
        # Convert to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect hands and call the callback function
        self.recognizer.recognize_async(mp_image, timestamp)        
        
    
    
    def save_result_callback(self, results:GestureRecognizerResult, output_image:mp.Image, timestamp_ms:int):
        
        # Save the result in a variable, so it can be processed by other functions
        self.results = results
        
    
    
    def detect_hands_video(self, frame, timestamp):
        
        # Convert to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect hands
        self.results = self.recognizer.recognize_for_video(mp_image, timestamp)
        
    
    
    def detect_hands_image(self, frame):
        
        # Convert to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect hands
        self.results = self.recognizer.recognize(mp_image)
        
        
        
        
    
        
if __name__ == '__main__':
    
    # Init hand detector object
    mpgr = Mediapipe_GestureRecognizer()
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(1)
    #cam.open(2)
    assert cam.isOpened()
    
    cam.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cam.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    
    frame_number = 0
    while cam.isOpened():
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            mpgr.logger.warning("Skipping empty frame")
            continue
        
        # Detect hands
        mpgr.detect_hands(frame, frame_number)
        frame_number += 1
        
        # Draw hands
        mpgr.draw_results(frame)     
        
        # Flip image horizontally
        frame = cv2.flip(frame, 1)
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
        # 3D plot of hands
        mpgr.plot_hands_3d()
        
        if mpgr.results is not None and False:
            break
        
    # Release the camera
    cam.release()
    cv2.destroyAllWindows()
    
    # Get info about results
    if False:
        print(f"results: {mpgr.results}")
        print(f"results.gestures: {mpgr.results.gestures}")
        print(f"results.handedness: {type(mpgr.results.handedness)}")
        print(f"results.hand_landmarks: {mpgr.results.hand_landmarks}")
        print(f"results.hand_world_landmarks: {mpgr.results.gestures}")
        print(f"style: {solutions.drawing_styles.get_default_hand_landmarks_style()}")
    
''' GestureRecogizerResult structure:
results
    gestures
    handedness
    hand_landmarks
    hand_world_landmarks
'''
    