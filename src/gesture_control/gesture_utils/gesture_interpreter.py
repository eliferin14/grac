import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from gesture_utils.gesture_detector import HandData, get_landmark_by_id
from gesture_utils.frameworks.joint_control import JointFrameworkManager


def get_vector(p1, p2):
    """Calculate the unity vector connecting the two points
    """    ''''''
    diff = p2-p1
    vector = diff / np.linalg.norm(diff)
    return vector



LEFT_WRIST = 15
RIGHT_WRIST = 16



class GestureInterpreter():
    
    frameworks = [JointFrameworkManager()]
        
    def __init__(self, labels_path, sequence_length=10):
        
        # Initialize sequences
        self.sequence_length = sequence_length
        self.right_hand_sequence = deque(maxlen=self.sequence_length)
        self.left_hand_sequence = deque(maxlen=self.sequence_length)
        
        # Load labels
        file = open(labels_path, 'r')
        labels = file.readlines()
        self.labels = [line.strip() for line in labels]
        
        # Generate the transition matrix
        transition_id = 0
        self.transitions = {}
        for gesture_from in self.labels:
            for gesture_to in self.labels:
                # Assign to each pair a unique identifier
                self.transitions[ (gesture_from, gesture_to) ] = transition_id
                transition_id += 1
                
        #for (state_from, state_to), identifier in self.transitions.items(): print(f"{state_from} -> {state_to}: {identifier}")
        
        self.left_p1 = None
        self.left_p2 = None
        
        self.selected_framework = 0
                
                
    def get_transition_id(self, gesture_from, gesture_to):
        return self.transitions.get((gesture_from, gesture_to), None)
                
    
    def interpret(self, right_hand_data: HandData, left_hand_data: HandData, pose_landmarks):
            
        self.right_hand_sequence.append(right_hand_data)
        self.left_hand_sequence.append(left_hand_data)
            
        #print(list(self.left_hand_sequence))
        #print(f"Right: ")
        
        # Check if there are enough gestures
        if len(self.right_hand_sequence) < 2: return
            
        """ # Get transition_id using the last two gestures
        left_tid = self.get_transition_id(self.left_hand_sequence[-2].gesture, self.left_hand_sequence[-1].gesture)
        
        # Do something depending on the transition
        
        if left_tid == self.get_transition_id('palm', 'fist'): 
            self.left_p1 = pose_landmarks[LEFT_WRIST]
            
        if left_tid == self.get_transition_id('fist', 'palm'):
            self.left_p2 = pose_landmarks[LEFT_WRIST]

            if self.left_p1 is not None and self.left_p2 is not None:
                vector = self.left_p2 - self.left_p1
                print(vector) """
                
        framework_manager = self.frameworks[self.selected_framework]
        framework_manager.interpret_gestures(self.right_hand_sequence[-1], self.left_hand_sequence[-1])
            
        
            
        
        