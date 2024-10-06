import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import logging
logging.basicConfig(level=logging.DEBUG)

# Some parameters for the model (default ones)
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode         # I want the live feed options -> for continuous videos

# Default model path
default_model_path = 'gesture_recognizer.task'

# Gesture recognizer class
class Mediapipe_GestureRecognizer():
    
    def __init__(self, model_path = default_model_path):
        
        self.logger = logging.getLogger()
        self.logger.info("Creating Mediapipe_GestureRecognizer instance")
        
        # Define a variable that will store the results of gesture recognition
        self.results = None
        
        # Load the options
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.save_result_callback,  
            num_hands=2)
        
        self.recognizer = GestureRecognizer.create_from_options(options)
        
    def save_result_callback(self, results:GestureRecognizerResult, output_image:mp.Image, timestamp_ms:int):
        
        # Save the result in a variable, so it can be processed by other functions
        self.results = results
        
    def draw_results(self, frame):
        
        if self.results is None:
            return frame
        
        # results is an object with all the data of the hands
        # results.hand_landmarks is a list of sets of landmark. One set per hand detected
        for landmarks in self.results.hand_landmarks:
            # result is a list of landmarks of a single hand
            # Draw the pose landmarks
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([ landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks ])
            solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style()
                )
            
        return frame
    
    def detect_hands_async(self, frame, timestamp):
        
        if frame is None:
            self.logger.warning("Empty frame received")
            return
        
        # Convert to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect hands and call the callback function
        self.recognizer.recognize_async(mp_image, timestamp)
        
        
    
        
if __name__ == '__main__':
    # Init hand detector object
    mpgr = Mediapipe_GestureRecognizer()
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(1)
    
    frame_number = 0
    while cam.isOpened():
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            mpgr.logger.warning("Skipping empty frame")
            continue
        
        # Flip image horizontally
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        mpgr.detect_hands_async(frame, frame_number)
        frame_number += 1
        
        # Draw hands
        mpgr.draw_results(frame)
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
    # Release the camera
    cam.release()
    cv2.destroyAllWindows()