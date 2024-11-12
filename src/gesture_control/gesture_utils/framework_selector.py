import rospy
import numpy as np

from gesture_utils.frameworks.base_framework import BaseFrameworkManager
from gesture_utils.frameworks.joint_control import JointControlManager







class FrameworkSelector():
    
    framework_managers = [
        BaseFrameworkManager(),
        JointControlManager()
    ]
    
    def __init__(self):
        
        # The default framework is the base framework
        # NOTE for testing purposes the default is the joint control
        self.selected_framework_manager = self.framework_managers[1]
        
        
        
        
        
    def interpret_gestures(
        self,
        rh_gesture,
        lh_gesture,
        #...
    ):
        rospy.logdebug(f"Right: {rh_gesture}, Left: {lh_gesture}")
        
        # Call the menu selection
        
        # Select the desired framework
        
        # Call the framework manager and do something
        self.selected_framework_manager.interpret_gestures(rh_gesture, lh_gesture)