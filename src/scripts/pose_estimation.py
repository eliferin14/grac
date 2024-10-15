import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode





class Mediapipe_PoseLandmarker():
    
    def __init__(self,
        model_path, 
        mode = 1, 
        min_pose_detection_confidence = 0.3,
        min_pose_presence_confidence = 0.3,
        min_tracking_confidence = 0.3,
        show_pose_plot = True
    ):
        self.logger = logging.getLogger()
        self.logger.info("Creating Mediapipe_PoseLandmarker instance")
        
        # Create the variable that will store the results
        self.results = None        
        
        # Create options object
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path), 
            num_poses=1, 
            min_pose_detection_confidence = min_pose_detection_confidence,
            min_pose_presence_confidence = min_pose_presence_confidence,
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
        self.landmarker = PoseLandmarker.create_from_options(options)        
        
        # Create plot
        self.show_plot = show_pose_plot
        if self.show_plot:                
            plt.ion()
            self.pose_fig = plt.figure()
            self.pose_ax = self.pose_fig.add_subplot(projection='3d')
            self.pose_ax.set_title("Pose")
            
        self.pose_coordinates = None
            
            
    def detect_pose(self, frame, timestamp):
        
        if frame is None:
            self.logger.warning("Empty frame received")
            return
        
        # Convert to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Choose which function to call based on mode
        if self.mode == 2:
            self.landmarker.detect_async(mp_image, timestamp)   
        elif self.mode == 1:             
            self.results = self.landmarker.detect_for_video(mp_image, timestamp)
        elif self.mode == 0:
            self.results = self.landmarker.detect(mp_image)
            
        self.convert_to_numpy()
        #print(self.pose_coordinates[16])         
            
    
    def save_result_callback(self, results:PoseLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
        
        # Save the result in a variable, so it can be processed by other functions
        self.results = results
        
        
        
    def convert_to_numpy(self):
        #print(self.results)
        if self.results is None:
            return
        
        self.pose_coordinates = np.zeros(shape=(33, 3), dtype=float)
        
        for landmarks in self.results.pose_world_landmarks:
            for i, landmark in enumerate(landmarks):
                self.pose_coordinates[i, 0] = landmark.x
                self.pose_coordinates[i, 1] = landmark.y
                self.pose_coordinates[i, 2] = landmark.z
                
    
    
    def draw_pose(self, frame):
        if self.results is None:
            return frame
        
        pose_landmarks_list = self.results.pose_landmarks
        
        for i in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[i]
            
            # Convert to protobuffer
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([ landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks ])
            
            # Draw onto the frame
            solutions.drawing_utils.draw_landmarks(
                frame,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
                )
            
        return frame
    
    def draw_mask(self, frame): 
        return       
        if self.results.segmentation_masks is None:
            return frame
        
        segmentation_mask = self.results.segmentation_masks[0].numpy_view()
        frame = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
        
        
        
    def get_point_by_index(self, landmark_id):
        pass
        
        
            
        
        
        
        
        
if __name__ == "__main__":
    
    # Create the object
    mppl = Mediapipe_PoseLandmarker('pose_landmarker_full.task')
    
    # Open camera
    cam = cv2.VideoCapture(0)    
    assert cam.isOpened()
    
    # Loop
    frame_number = 0
    while cam.isOpened():
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            mppl.logger.warning("Skipping empty frame")
            continue
        
        # Detect pose
        frame_number += 1
        mppl.detect_pose(frame, frame_number)
        
        # Draw pose
        frame = mppl.draw_pose(frame)
        
        # Flip image horizontally
        frame = cv2.flip(frame, 1)
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
    # Release the camera
    cam.release()
    cv2.destroyAllWindows()
        
        
