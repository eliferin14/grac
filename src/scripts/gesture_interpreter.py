import numpy as np
import matplotlib.pyplot as plt

from gesture_detector import HandData






class GestureInterpreter():
    
    right_hand_sequence: list = []
    left_hand_sequence: list = []
    
    def __init__(self):
        pass
    
    def interpret(self, hand_data: HandData):
        # Check if there is actually data
        if hand_data.landmarks is None:
            return
        
        # Add the datapoint to the sequence
        if hand_data.handedness == 'Right':
            self.right_hand_sequence.append(hand_data)
        else:
            self.left_hand_sequence.append(hand_data)
        
        
        print(f"Right: #{len(self.right_hand_sequence)};\tLeft: #{len(self.left_hand_sequence)}")