import rospy
from functools import partial

from sami.arm import Arm, EzPose






class BaseFrameworkManager:
    
    framework_name = "Base"
    
    def __init__(self):
        pass
    
    def dummy_callback(self):
        return
    
    def interpret_gestures(self, *args, **kwargs):
        
        """Given the left and right hand gestures and the landmarks, define the function that has to be run.
        Such function is returned to the caller for execution

        Args:
            right_gesture (_type_): _description_
            left_gesture (_type_): _description_
        """        ''''''
        
        #print(kwargs)
        
        arm = kwargs.get('arm')
        
        return partial(arm.get_joints)