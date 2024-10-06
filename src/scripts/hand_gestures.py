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
            result_callback=self.print_result,  
            num_hands=2)
        
        self.recognizer = GestureRecognizer.create_from_options(options)
        
    def print_result(self, results:GestureRecognizerResult, output_image:mp.Image, timestamp_ms:int):
        
        # Save the result in a variable, so it can be processed by other functions
        self.results = results
        
    def draw_results(self, frame):
        
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
    
    def main(self):
        
        # Initialize the camera feed
        cam = cv2.VideoCapture(0)
        # Check that camera is actually open
        if not cam.isOpened():
            self.logger.fatal("Cannot open camera")
            exit()
        self.logger.debug("Camera opened correctly")
            
        # Get width and height of the captured frames
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.debug(f"Frame size: [{frame_width}, {frame_height}]")
        
        # Loop for processing frames
        timestamp = 0
        while cam.isOpened():
            timestamp += 1
            
            # Capture frame
            ret, frame = cam.read()
            if not ret:
                self.logger.warning("Skipping empty frame")
                continue
            
            # Convert frame to mp.Image
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Detect hands and call the callback function
            self.recognizer.recognize_async(image, timestamp)
            
            # Process the results
            if self.results is not None:
                
                # Draw on the frame
                frame = self.draw_results(frame)
            
            # Display frame
            cv2.imshow("Live feed", frame)            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("Closing Camera Stream")
                break
            
        cam.release()
        
        
    
        
if __name__ == '__main__':
    mpgr = Mediapipe_GestureRecognizer()
    mpgr.main()