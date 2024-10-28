import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from landmark_normalizer import normalize_landmarks
else:    
    from .landmark_normalizer import normalize_landmarks

# Hand connections: pair of landmarks where a line is present
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), 
    (13, 14), (14, 15), (15, 16), 
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]



class CustomGestureRecognizer():
    
    def __init__(self,
                 model_path = "model/gesture_classifier.tflite",
                 gesture_list_path = "gesture_list.csv",
                 minimum_gesture_detection_confidence = 0.5,
                 show_plot=False
        ):
        """Class that provides a simple interface for the gesture recognition NN

        Args:
            model_path (str, optional): Path to the .tflite model file. Defaults to "model/gesture_classifier.tflite".
            gesture_list_path (str, optional): Path to the gesture list .csv file. Defaults to "gesture_list.csv".
            minimum_gesture_detection_confidence (float, optional): Minimum confidence required for a gesture to be recognized. Defaults to 0.5.
            show_plot (bool, optional): If true a plot of the detected hands will be produced. Defaults to False.
        """        ''''''
        
        # Load the model
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load the gesture list
        self.gesture_list = np.loadtxt(gesture_list_path, delimiter=',', dtype=str)
        
        # Set the minimum detection confidence
        self.mgdc = minimum_gesture_detection_confidence
        
        # Create the plot 
        self.show_plot = False
        if show_plot:
            self.show_plot = True
            plt.ion()
            self.hand_figure = plt.figure()
            self.hand_raw_ax = self.hand_figure.add_subplot(121, projection='3d')
            self.hand_normalized_ax = self.hand_figure.add_subplot(122, projection='3d')
        
        
        
        
        
    def convert_to_numpy(self, hand_world_landmarks):
        """Converts the landmark list to a numpy matrix of 3d points

        Args:
            hand_world_landmarks (_type_): result of the hand recognition process
        """        ''''''
        hand_landmarks_matrix = np.zeros((21,3))
        i = 0
        for landmark in hand_world_landmarks.landmark:
            hand_landmarks_matrix[i,0] = landmark.x
            hand_landmarks_matrix[i,1] = landmark.y
            hand_landmarks_matrix[i,2] = landmark.z
            i+=1
            
        return hand_landmarks_matrix
    
        
        
    
    def recognize(self, hand_world_landmarks, handedness):
        """Recognize gesture given the hand landmark coordinates

        Args:
            hand_world_landmarks (_type_): list of landmark coordinates
            handedness (str): handedness of the detected hand. Can be 'Left' or 'Right'
        """        ''''''
        
        if hand_world_landmarks is None:
            return
        
        # Convert to numpy matrix
        landmark_matrix = self.convert_to_numpy(hand_world_landmarks)
        
        # Normalize landmarks
        normalized_landmarks = normalize_landmarks(landmark_matrix, handedness)
        
        # Plot hands
        if self.show_plot:
            pass
            self.plot_hand(self.hand_raw_ax, landmark_matrix, 'b', 'k', 'Raw coordinates')
            self.plot_hand(self.hand_normalized_ax, normalized_landmarks, 'b', 'k', 'Raw coordinates')
            
        # Prepare the data for the model
        feature_vector = normalized_landmarks.reshape(-1)
        feature_vector = np.float32(np.array([feature_vector]))
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], feature_vector)
        self.interpreter.invoke()
        predict_result = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Extract the most likely gesture
        gesture_id = np.argmax(np.squeeze(predict_result))
        gesture_confidence = np.max(np.squeeze(predict_result))
        
        # Chack if confidence is high enough for the gesture to be accepted
        if gesture_confidence < 0.5:
            gesture_id = 0
        
        return gesture_id
    
    
    
    def get_gesture_name(self, gesture_id):
        """Translates the gesture ID into the corresponding gesture name

        Args:
            gesture_id (int): ID of the gesture

        Returns:
            str: Name of the gesture
        """        ''''''
        if gesture_id is None: return None
        return self.gesture_list[gesture_id]
    
    
    
    
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
        ax.scatter(0,0,0, c='m')
        
        # Plot the connections
        for i,j in connections:
            ax.plot([coord[i,0], coord[j,0]], [coord[i,1], coord[j,1]], [coord[i,2], coord[j,2]], c=line_color)
        
        

        
if __name__ == "__main__":
    
    import cv2
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    
    cgr = CustomGestureRecognizer(show_plot=True)
    
    # For webcam input:
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
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
                    
                for hand_world_landmarks, handedness in zip(results.multi_hand_world_landmarks, results.multi_handedness):
                    gesture_id = cgr.recognize(hand_world_landmarks, handedness.classification[0].label)
                    print(f"{handedness.classification[0].label}: {cgr.get_gesture_name(gesture_id)}")
                    
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()