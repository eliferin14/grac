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

from gesture_utils.fps_counter import FPS_Counter
from gesture_utils.gesture_filter import GestureFilter
#from landmark_normalizer import normalize_landmarks

# Command line arguments
parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("--camera_id", type=int, default=0, help="ID of camera device. Run v4l2-ctl --list-devices to get more info")
parser.add_argument("-grmd", "--gesture_recognizer_model_directory", type=str, default="training/exported_model", help="Path to the gesture recognition model")
parser.add_argument("-gtt", "--gesture_transition_timer", type=float, default=0.5, help="Timer required for a new grsture to be registered")
parser.add_argument("--draw_hands", type=bool, default=True, help="If true draw the hands landmarks on the output frame")
parser.add_argument("--draw_pose", type=bool, default=True, help="If true draw the pose landmarks on the output frame")

# Hands and pose models packages
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Cosmetics
right_color = (255,0,0) # Blue in BGR
left_color = (0,0,255)  # Red in BGR
rh_drawing_specs = DrawingSpec(right_color)   # Blue in RGB
lh_drawing_specs = DrawingSpec(left_color)   # Red in RGB

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





# Gesture dataclass
from dataclasses import dataclass
@dataclass
class HandData:
    landmarks: np.array = None
    gesture: str = None
    handedness: str = None
    palm_origin: np.array = None
    
# Pose connections
pose_connections = [
    (11, 12),  # Shoulders
    (11, 13),  # Left shoulder to left elbow
    (12, 14),  # Right shoulder to right elbow
    (13, 15),  # Left elbow to left wrist
    (14, 16),  # Right elbow to right wrist
    (15, 17),  # Left wrist to left pinky
    (15, 19),  # Left wrist to left index
    (15, 21),  # Left wrist to left thumb
    (16, 18),  # Right wrist to right pinky
    (16, 20),  # Right wrist to right index
    (16, 22),  # Right wrist to right thumb
    (11, 23),  # Left shoulder to left hip
    (12, 24),  # Right shoulder to right hip
    (23, 25),  # Left hip to left knee
    (24, 26),  # Right hip to right knee
    (25, 27),  # Left knee to left ankle
    (26, 28),  # Right knee to right ankle
    (27, 29),  # Left ankle to left heel
    (28, 30),  # Right ankle to right heel
    (29, 31),  # Left heel to left foot index
    (30, 32),  # Right heel to right foot index
]


def get_landmark_by_id(landmark_list, landmark_id):
    if landmark_list:
        return np.array( [landmark_list.landmark[landmark_id].x, 
                            landmark_list.landmark[landmark_id].y, 
                            landmark_list.landmark[landmark_id].z, ])
    
    
    
    
    
    
# 3D plotting function
def draw_reference_system(ax, length=1.0):
    """
    Draws the base reference system (x, y, z axes) as arrows in a 3D Matplotlib plot.
    
    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to draw on.
        length (float): Length of the arrows to draw for each axis.
    """
    # Draw x-axis arrow in red
    ax.quiver(0, 0, 0, length, 0, 0, color='red', arrow_length_ratio=0.1)
    
    # Draw y-axis arrow in green
    ax.quiver(0, 0, 0, 0, length, 0, color='green', arrow_length_ratio=0.1)
    
    # Draw z-axis arrow in blue
    ax.quiver(0, 0, 0, 0, 0, length, color='blue', arrow_length_ratio=0.1)
    
    # Optionally, add labels for each axis
    ax.text(length, 0, 0, 'X', color='red')
    ax.text(0, length, 0, 'Y', color='green')
    ax.text(0, 0, length, 'Z', color='blue')
    

def get_bounding_cube(x, y, z, corner_thickness=50):
    """
    Plots the smallest cube containing all data points on the provided axis
    and marks the corners with specified thickness.

    Parameters:
    ax: matplotlib.axes.Axes3D
        The 3D axis on which to plot the cube.
    x: array-like
        The x-coordinates of the points.
    y: array-like
        The y-coordinates of the points.
    z: array-like
        The z-coordinates of the points.
    corner_thickness: int
        The thickness of the corners' scatter plot.
    """
    # Calculate the minimum and maximum values for x, y, and z
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    # Determine the range and the size of the cube
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Calculate the midpoints
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    mid_z = (z_min + z_max) / 2

    # Set the limits for the cube
    #ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    #ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    #ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    # Create the cube's corners
    r = max_range / 2  # Half the side length of the cube
    corners = np.array([[mid_x - r, mid_y - r, mid_z - r],
                        [mid_x + r, mid_y - r, mid_z - r],
                        [mid_x + r, mid_y + r, mid_z - r],
                        [mid_x - r, mid_y + r, mid_z - r],
                        [mid_x - r, mid_y - r, mid_z + r],
                        [mid_x + r, mid_y - r, mid_z + r],
                        [mid_x + r, mid_y + r, mid_z + r],
                        [mid_x - r, mid_y + r, mid_z + r]])

    # Plot the corners of the cube
    #ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], color='r', s=corner_thickness)  # 's' sets the size of the markers
    
    return corners
    


def draw_pose_connections(ax, x, y, z):
    """
    Draws connections between body pose landmarks in 3D on the provided axis.
    
    Parameters:
    ax: matplotlib.axes.Axes3D
        The 3D axis on which to draw the connections.
    x: array-like
        The x-coordinates of the landmarks.
    y: array-like
        The y-coordinates of the landmarks.
    z: array-like
        The z-coordinates of the landmarks.
    """
    # Draw connections
    for start, end in pose_connections:
        ax.plot([x[start], x[end]], 
                [y[start], y[end]], 
                [z[start], z[end]], 
                color='k',
               )  # You can change the color as needed


    
    
def plot3D(ax, pose_landmarks, right_hand_landmarks, left_hand_landmarks):
    
    #print("Plotting function")
    
    # Clear plot
    ax.cla()
    
    # Plot the reference system
    draw_reference_system(ax)
    
    # Plot the pose landmarks
    if pose_landmarks:
        pose_x = [lm.x for lm in pose_landmarks.landmark]
        pose_y = [lm.y for lm in pose_landmarks.landmark]
        pose_z = [lm.z for lm in pose_landmarks.landmark]
        draw_pose_connections(ax, pose_x, pose_y, pose_z)
        ax.scatter(pose_x[11:], pose_y[11:], pose_z[11:], c='g', marker='o')
    
    
    # Plot bounding cube
    #pose_x.append(0)
    #pose_y.append(0)
    #pose_z.append(0)
    get_bounding_cube(ax, pose_x, pose_y, pose_z, corner_thickness=1)
    
    # Update the plot
    plt.draw()
    
    return





    
    





class GestureDetector():
    """Gesture for Robotic Arm Control
    """    ''''''
    
    def __init__(self, model_directory, transition_timer):
        
        # Define constants
        self.RIGHT = 1
        self.LEFT = 0
        
        # Allocate the hand landmarker 
        self.hand_landmarker = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # Extract the gesture recognizer .tflite models
        with zipfile.ZipFile(model_directory+"/custom_gesture_recognizer.task", 'r') as zip_ref:
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
        self.labels_path = model_directory+'/labels.txt'
        file = open(self.labels_path, 'r')
        labels = file.readlines()
        self.labels = [line.strip() for line in labels]
        
        # Allocate the pose landmarker
        self.pose_landmarker = mp_pose.Pose(
            model_complexity = 0,
            min_detection_confidence=0.5,
            enable_segmentation = False
        )
        
        # Initialize results to None
        self.right_hand_gesture, self.left_hand_gesture = None, None
        self.right_hand_landmarks, self.left_hand_landmarks = None, None
        self.right_hand_landmarks_matrix, self.right_hand_world_landmarks_matrix = None, None
        self.left_hand_landmarks_matrix, self.left_hand_world_landmarks_matrix = None, None
        self.pose_landmarks = None
        self.right_hand_data, self.left_hand_data = HandData(), HandData()
        
        # Initialize the filters
        self.rightGTR = GestureFilter(transition_timer=transition_timer)
        self.leftGTR = GestureFilter(transition_timer=transition_timer)
        
        
        
        
    def process(self, frame, use_threading=False):
        """Detects hands and pose in a frame. Also recognize hand gestures

        Args:
            frame (opencv image): input image
            use_threading (bool, optional): if true the hand detection and pose detection are run on different threads to imporve performance. Defaults to False.
        """        ''''''
        
        if frame is None:
            self.logger.warning("Empty frame received")
            return
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        if use_threading:
        
            # Create the two threads
            hands_thread = threading.Thread(target=self._process_hands, args=(rgb_frame,))
            pose_thread = threading.Thread(target=self._process_pose,args=(rgb_frame,))
            
            # Start the threads
            hands_thread.start()
            pose_thread.start()
        
            # Wait for both functions to return
            hands_thread.join()
            pose_thread.join()
            
        else:
            
            # Call the two functions sequentially
            self._process_hands(rgb_frame)
            self._process_pose(rgb_frame)
        
        
        
        
    def _process_hands(self, rgb_frame, ignore_orientation=False):
        """Detects hands and recognize gestures. The output of the models are stored in class variables

        Args:
            rgb_frame (opencv image): input frame in RGB format
        """        ''''''
        
        # Call the hand landmarker model
        results = self.hand_landmarker.process(rgb_frame)
        
        # Reset outputs to None
        self.right_hand_gesture, self.left_hand_gesture = None, None            
        self.right_hand_landmarks, self.left_hand_landmarks = None, None
        self.right_hand_data, self.left_hand_data = HandData(), HandData()
        
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
                
                # Save the landmarks and the gesture to the appropriate varaible
                if handedness.classification[0].index == self.RIGHT:
                    self.right_hand_landmarks_matrix = hand_landmarks_tensor[0]
                    self.right_hand_world_landmarks_matrix = hand_world_landmarks_tensor[0]
                    _, gesture = self.rightGTR.gesture_change_request(gesture)
                    self.right_hand_gesture = gesture
                    
                    #self.right_hand_landmarks = hand_landmarks
                    #self.right_hand_data = HandData(landmarks=hand_landmarks_tensor[0], gesture=gesture, handedness='Right')
                else:
                    self.left_hand_landmarks_matrix = hand_landmarks_tensor[0]
                    self.left_hand_world_landmarks_matrix = hand_world_landmarks_tensor[0]
                    _, gesture = self.leftGTR.gesture_change_request(gesture)
                    self.left_hand_gesture = gesture
                    
                    self.left_hand_landmarks = hand_landmarks
                    self.left_hand_data = HandData(landmarks=hand_landmarks_tensor[0], gesture=gesture, handedness='Left')
    
    
    
    
    def _process_pose(self, rgb_frame):
        """Detects pose landmarks. The output is stored in a class variable

        Args:
            rgb_frame (opencv image): input frame in RGB format
        """        ''''''
        
        # Call the landmarker
        results = self.pose_landmarker.process(rgb_frame)
        
        # Reset results to None
        self.pose_landmarks = None
        
        if results.pose_landmarks is None: return
        
        # Save raw results for drawing
        self.pose_landmarks_raw = results.pose_landmarks
        
        # Rotate to have a standin figure
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        rotated_landmarks = self._convert_results_to_matrix(results.pose_landmarks) @ R.T
        
        # Translate
        origin = (rotated_landmarks[11] + rotated_landmarks[12]) / 2
        rotated_landmarks -= origin
        self.pose_landmarks = rotated_landmarks
    
    
    
    
    def _convert_results_to_tensor(self, landmarks):
        """Converts the landmark list to a numpy matrix of 3d points

        Args:
            hand_world_landmarks (_type_): result of the hand recognition process as a tensor
        """        ''''''
        if landmarks is None: return
        
        num_landmarks = len(landmarks.landmark)
        hand_landmarks_matrix = np.zeros((1,num_landmarks,3))
        i = 0
        for landmark in landmarks.landmark:
            hand_landmarks_matrix[0,i,0] = landmark.x
            hand_landmarks_matrix[0,i,1] = landmark.y
            hand_landmarks_matrix[0,i,2] = landmark.z
            i+=1
            
        return hand_landmarks_matrix.astype(np.float32)  
    
    
    def _convert_results_to_matrix(self, landmarks):
        if landmarks is None: return
        return self._convert_results_to_tensor(landmarks)[0]
    
    
    
    
    def get_hand_gestures(self):
        """Returns the right and left hand gestures names

        Returns:
            str, str: right and left gesture names
        """        ''''''
        return self.right_hand_data.gesture, self.left_hand_data.gesture
    
    
    
    
        
        
        
    
    def draw_results(self, frame, draw_hands=True, draw_pose=True):
        """Draw the detected landmarks on the provided frame

        Args:
            frame (opencv image): where to draw the landmarks
            draw_hands (bool, optional): if true the hand landmarks are drawn. Defaults to True.
            draw_pose (bool, optional): if true the pose landmarks are drawn. Defaults to True.
        """        ''''''
        
        if draw_pose:
            if self.pose_landmarks is not None:
        
                # Remove unwanted landmarks from the drawing process
                # Since a change of the properties of the landmark is required, create an independent copy of the landmark set
                filtered_landmarks = deepcopy(self.pose_landmarks_raw)
                for idx, landmark in enumerate(filtered_landmarks.landmark):
                    if idx in excluded_landmarks:
                        landmark.presence = 0
                    
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, filtered_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=pose_drawing_specs,connection_drawing_spec=DrawingSpec(color=(128,128,128)))

        # Draw hands landmarks
        if draw_hands:
            if self.left_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, self.left_hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=lh_drawing_specs)
            if self.right_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, self.right_hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=rh_drawing_specs)
    
    
    
    
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
    
    grac = GestureDetector(
        model_directory=args.gesture_recognizer_model_directory,
        transition_timer=args.gesture_transition_timer
    )
    
    # Open the camera live feed and process the frames
    cam = cv2.VideoCapture(args.camera_id)
    assert cam.isOpened()
    
    # Fps counter
    fps_counter = FPS_Counter()
    
    # Named tuple for text to print on image
    frame_text = namedtuple('FrameText', ['name', 'value', 'color'])
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
        grac.process(frame, use_threading=True)    
        rhg, lhg = grac.get_hand_gestures()
        print(grac.left_hand_data)
        
        # Draw hands and pose
        grac.draw_results(frame, args.draw_hands, args.draw_pose)   
        # 3D plot of hands
        #grac.mpgr.plot_hands_3d()  
        
        # Add info as text        
        text_list = []
        text_list.append(frame_text('FPS', fps, (0,255,0)))
        if rhg is not None: text_list.append(frame_text('Right', rhg, right_color))
        if lhg is not None: text_list.append(frame_text('Left', lhg, left_color))
        grac.add_text(frame, text_list, row_height=30)
        
        # Display frame
        cv2.imshow("Live feed", frame)            
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
    # Release the camera
    cam.release()
    cv2.destroyAllWindows()