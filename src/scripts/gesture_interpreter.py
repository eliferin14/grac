import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from gesture_detector import HandData


def get_vector(p1, p2):
    diff = p2-p1
    vector = diff / np.linalg.norm(diff)
    return vector



class GestureInterpreter():
    
    def __init__(self, sequence_length=10):
        
        # Initialize sequences
        self.sequence_length = sequence_length
        self.right_hand_sequence = deque(maxlen=self.sequence_length)
        self.left_hand_sequence = deque(maxlen=self.sequence_length)
    
    def interpret(self, hand_data: HandData, left_hand_data: HandData):
        # Check if there is actually data
        if hand_data.landmarks is None:
            return
        
        # Add the datapoint to the sequence
        if hand_data.handedness == 'Right':
            self.right_hand_sequence.append(hand_data)
        elif hand_data.handedness == 'Left':
            self.left_hand_sequence.append(hand_data)
        else:
            print("Error: handedness is not \'Right\' nor \'Left\'")
            exit()
            
        #print(list(self.left_hand_sequence))
        #print(f"Right: ")
        
        