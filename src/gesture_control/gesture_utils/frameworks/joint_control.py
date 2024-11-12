import rospy
import numpy as np
import cv2

from gesture_utils.frameworks.base_framework import BaseFrameworkManager

from gesture_utils.ros_utils import convert_ROSpoints_to_matrix





class JointControlManager(BaseFrameworkManager):
    
    gesture_to_joint_list = np.array(['fist', 'one', 'two', 'three', 'four', 'palm'])
    
    def __init__(self):
        
        # In the beginning, no joint is selected
        self.selected_joint = None
        
        return 
    
    
    
    
    def interpret_gestures(self, right_gesture, left_gesture, callback=None):
        
        # The left hand selects the joint
        candidate_selected_joint = np.where(self.gesture_to_joint_list == left_gesture)[0]
        if candidate_selected_joint != self.selected_joint and candidate_selected_joint is not None:
            rospy.loginfo(f"Joint {candidate_selected_joint} selected")
            self.selected_joint = candidate_selected_joint
        
        #return super().interpret_gestures(right_gesture, left_gesture, callback)