import rospy
import numpy as np

from gesture_utils.frameworks.base_framework import BaseFrameworkManager
from gesture_utils.frameworks.joint_control import JointFrameworkManager

from sami.arm import Arm, EzPose







class FrameworkSelector():
    
    framework_managers = [
        BaseFrameworkManager(),
        JointFrameworkManager()
    ]
    
    def __init__(self):
        
        # The default framework is the base framework
        # NOTE for testing purposes the default is the joint control
        self.selected_framework_manager = self.framework_managers[1]
        
        
        
        
        
    def interpret_gestures(
        self,
        arm: Arm,
        rh_gesture,
        lh_gesture,
        #...
    ):
        """This function asks to the selected framework manager for a function to execute. Then returns such function to the caller

        Args:
            rh_gesture (_type_): _description_
            lh_gesture (_type_): _description_

        Returns:
            _type_: _description_
        """        ''''''
        
        # Call the menu selection
        
        # Select the desired framework
        
        # Call the framework manager and do something
        callback = self.selected_framework_manager.interpret_gestures(arm, rh_gesture, lh_gesture)
        
        return callback