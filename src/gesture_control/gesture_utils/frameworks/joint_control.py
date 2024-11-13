import rospy
import numpy as np
from functools import partial

from gesture_utils.frameworks.base_framework import BaseFrameworkManager

from gesture_utils.ros_utils import convert_ROSpoints_to_matrix

from sami.arm import Arm, EzPose



service_name = '/iris_sami/joints'

class JointControlManager(BaseFrameworkManager):
    
    gesture_to_joint_list = np.array(['fist', 'one', 'two', 'three', 'four', 'palm'])
    
    def __init__(self):
        
        # In the beginning, no joint is selected
        self.selected_joint = None
        
        return 
    
    
    
    
    def interpret_gestures(self, arm:Arm, right_gesture, left_gesture):
        
        # The left hand selects the joint
        candidate_selected_joint = np.where(self.gesture_to_joint_list == left_gesture)[0]
        print(candidate_selected_joint)
        
        # If no joint is selected, do nothing
        if candidate_selected_joint.size == 0:
            return partial(super().dummy_callback)
        
        if candidate_selected_joint != self.selected_joint:
            rospy.loginfo(f"Joint {candidate_selected_joint} selected")
            self.selected_joint = candidate_selected_joint
        
        # The right hand selects the angle
                
        return partial(arm.get_joints)
        