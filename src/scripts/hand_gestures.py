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

# Define styles
right_hand_drawing_specs = mp.solutions.drawing_styles.DrawingSpec( color=(255, 0, 0) )
left_hand_drawing_specs = mp.solutions.drawing_styles.DrawingSpec( color=(0,0,255) )

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
    '''Class that provides a simplified interface for the mediapipe gesture_recognizer task
    '''
    
    def __init__(self, 
        model_path, 
        mode = 1, 
        min_hand_detection_confidence = 0.3,
        min_hand_presence_confidence = 0.3,
        min_tracking_confidence = 0.3,
        show_hand_plot = True
    ):
        """_summary_

        Args:
            model_path (str): path to the model .task file
            mode (int, optional): running mode of the model. 0:IMAGE, 1:VIDEO, 2:LIVE_STREAM. Defaults to 1.
            min_hand_detection_confidence (float, optional): The minimum confidence score for the hand detection to be considered successful in palm detection model. Defaults to 0.3.
            min_hand_presence_confidence (float, optional): The minimum confidence score of hand presence score in the hand landmark detection model. In Video mode and Live stream mode of Gesture Recognizer, if the hand presence confident score from the hand landmark model is below this threshold, it triggers the palm detection model. Otherwise, a lightweight hand tracking algorithm is used to determine the location of the hand(s) for subsequent landmark detection. Defaults to 0.3.
            min_tracking_confidence (float, optional): The minimum confidence score for the hand tracking to be considered successful. This is the bounding box IoU threshold between hands in the current frame and the last frame. In Video mode and Stream mode of Gesture Recognizer, if the tracking fails, Gesture Recognizer triggers hand detection. Otherwise, the hand detection is skipped. Defaults to 0.3.
            show_hand_plot (bool, optional): If true a 3D matplotlib representation of the hands is produced. Defaults to True.
        """        ''''''
        
        self.logger = logging.getLogger()
        self.logger.info("Creating Mediapipe_GestureRecognizer instance")
        
        # Define a variable that will store the results of gesture recognition
        self.results = None
        self.right_hand_coordinates, self.left_hand_coordinates = None, None
        self.right_hand_gesture, self.left_hand_gesture = None, None
        
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
    
    
    
    def draw_results(self, frame, draw_hands=True, draw_bb=True):
        """Function that draws and shows the detection and classification results

        Args:
            frame (opencv image): Frame on which the results are drawn
            draw_hands (bool, optional): if True the landmarks are drawn. Defaults to True.
            draw_bb (bool, optional): if True the bounding box of each hand is drawn. Defaults to True.

        Returns:
            opencv image : processed image ready to be shown
        """        ''''''
        
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
            
            if draw_hands:    
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
            if draw_bb:
                min_x = min([landmark.x for landmark in landmarks])
                max_x = max([landmark.x for landmark in landmarks])
                min_y = min([landmark.y for landmark in landmarks])
                max_y = max([landmark.y for landmark in landmarks])
                
                width = int((max_x - min_x) * frame.shape[1])
                height = int((max_y - min_y) * frame.shape[0])
                
                top_left_x = int(min_x * frame.shape[1])
                top_left_y = int(min_y * frame.shape[0])
                color = (255,0,0) if handedness[0].index==0 else (0,0,255)
                cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), color, 2)
            
        return frame
                
            
    
    def draw_labels(self, frame, height=50):
        """Writes the names of the detected gestures on the image

        Args:
            frame (opencv image): input frame
            height (int, optional): Height of the text. Defaults to 50.

        Returns:
            opencv image: processed frame
        """        ''''''
        
        right_label = self.right_hand_gesture.category_name if self.right_hand_gesture is not None else "None"
        left_label = self.left_hand_gesture.category_name if self.left_hand_gesture is not None else "None"
        frame = cv2.putText(frame, left_label, org=(50, height), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
        frame = cv2.putText(frame, right_label, org=(50, 2*height), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
            
        return frame
        
    
    
    def plot_hands_3d(self):
        """Produces a 3D plot of the hands landmarks using matplotlib
        
        The landmarks are memorized as a field of the class, and are updated at every step
        """        ''''''
        
        if not self.show_plot:
            self.logger.debug("Calling plotting function but plot is disabled")
            return
        
        #plt.cla()
        #self.ax.scatter(xs, ys, zs, marker=m)
        
        """ if self.right_hand_coordinates is not None:
            for landmark in self.right_hand_coordinates:
                self.ax.scatter(landmark[0], landmark[1], landmark[2]) """
                
        """ if self.right_hand_coordinates is not None:
            self.right_hand_ax.scatter(*zip(*self.right_hand_coordinates)) """
        
        self.plot_hand(self.right_hand_ax, self.right_hand_coordinates, 'b', 'k', "Right hand")
        self.plot_hand(self.left_hand_ax, self.left_hand_coordinates, 'r', 'k', "Left hand")

        self.hands_fig.canvas.draw()
        self.hands_fig.canvas.flush_events()
        
        
        
        
    def plot_hand(self, ax, coord, landmark_color, line_color, title):
        """Function that draws a hand in a plt.ax object

        Args:
            ax (plt.ax): Axis where to draw the hand plot
            coord (21x3 numpy array): Set of 3D coordinates of the 21 landmarks of a hand
            landmark_color : Color for the points
            line_color : Color for the lines
            title : Title of the plot
        """        ''''''
        
        # Set the ax as active
        plt.sca(ax)
        
        # Clear the plot
        plt.cla()        
        
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        if coord is None:
            return
        
        # Plot the new points
        ax.scatter(*zip(*coord), c=landmark_color)
        ax.scatter(0,0,0, c='g')
        
        # Plot the connections
        for i,j in connections:
            ax.plot([coord[i,0], coord[j,0]], [coord[i,1], coord[j,1]], [coord[i,2], coord[j,2]], c=line_color)
            
    
    
    def convert_results_to_numpy(self):
        """Takes the output object of the model and extract the landmarks world coordinates into a 21x3 numpy array
        """        ''''''
        
        if self.results is None:
            return
        
        # Create two empty arrays: one for right and one for left hand
        self.right_hand_coordinates = None
        self.left_hand_coordinates = None        
        
        self.right_hand_gesture = None
        self.left_hand_gesture = None
        
        for i, landmarks in enumerate(self.results.hand_world_landmarks):
            
            # Check if right or left hand
            handedness = self.results.handedness[i][0].index   # right: 0, left: 1
            #print(handedness)
            
            # Copy results to numpy arrays
            if handedness == 0:
                self.right_hand_coordinates = self.copy_coordinates_to_numpy_array(landmarks)
                self.right_hand_gesture = self.results.gestures[i][0]
            else:
                self.left_hand_coordinates = self.copy_coordinates_to_numpy_array(landmarks)
                self.left_hand_gesture = self.results.gestures[i][0]
            
                
    
    
    def copy_coordinates_to_numpy_array(self, src):
        dst = np.zeros(shape=(21, 3), dtype=float)
        for i, landmark in enumerate(src):
            dst[i, 0] = landmark.x
            dst[i, 1] = landmark.y
            dst[i, 2] = landmark.z          
        return dst  
    
    
    
    def detect_hands(self, mp_image:mp.Image, timestamp:int):
        """Function that calls the gesture_recognizer model

        Args:
            mp_image (mp.Image): frame where the model will detect hands
            timestamp (int): increasing number. Required if VIDEO or LIVE_STREAM mode are selected

        Returns:
            2 21x3 numpy arrays: right and left hand coordinates
        """        ''''''
        
        # Choose which function to call based on mode
        if self.mode == 2:
            self.recognizer.recognize_async(mp_image, timestamp)   
        elif self.mode == 1:             
            self.results = self.recognizer.recognize_for_video(mp_image, timestamp)
        elif self.mode == 0:
            self.results = self.recognizer.recognize(mp_image)
            
        # Copy landmarks coordinates to numpy array
        self.right_hand_coordinates, self.left_hand_coordinates = None, None
        self.convert_results_to_numpy()
        
        self.logger.debug(f"Right hand: {self.right_hand_coordinates}")
        self.logger.debug(f"Left hand: {self.left_hand_coordinates}")
        
        return self.right_hand_coordinates, self.left_hand_coordinates
    
        
    
    
    def save_result_callback(self, results:GestureRecognizerResult, output_image:mp.Image, timestamp_ms:int):
        
        # Save the result in a variable, so it can be processed by other functions
        self.results = results
        
    
    
    
    def get_point_by_index(self, hand:int, index:int):
        if self.results is None:
            return
        
        if index < 0 or index > 20:
            return
        
        if hand == 0 and self.right_hand_coordinates is not None: # right
            return self.right_hand_coordinates[index]
        elif hand == 1 and self.left_hand_coordinates is not None:
            return self.left_hand_coordinates[index]
        else:
            return
        
    
    
    def get_right_hand_gesture(self):
        """Get the gesture of the right hand"""
        return self.right_hand_gesture
    
    def get_left_hand_gesture(self):
        '''Get the gesture of the left hand'''
        return self.left_hand_gesture
    
    def get_hand_gestures(self):
        '''Get the gestures of right and left hands'''
        return self.right_hand_gesture, self.left_hand_gesture
    
    
    
    
    def get_hands_coordinates(self):
        '''Get the 3D coordinates of the hand landmarks'''
        return self.right_hand_coordinates, self.left_hand_coordinates
        
        
    
        
if __name__ == '__main__':
    
    # Init hand detector object
    mpgr = Mediapipe_GestureRecognizer()
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(1)
    #cam.open(2)
    assert cam.isOpened()
    
    cam.set(cv2.CV_CAP_PROP_FRAME_WIDTH,640);
    cam.set(cv2.CV_CAP_PROP_FRAME_HEIGHT,480);
    
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
    