import rospy
from functools import partial

from sami.arm import Arm, EzPose





def dummy_callback(p1, p2):
    rospy.loginfo(f"Right: {p1}, Left: {p2}")






class BaseFrameworkManager:
    
    def __init__(self):
        pass
    
    def dummy_callback(self):
        return
    
    def interpret_gestures(self, arm:Arm, right_gesture, left_gesture):
        """Given the left and right hand gestures and the landmarks, define the function that has to be run.
        Such function is returned to the caller for execution

        Args:
            right_gesture (_type_): _description_
            left_gesture (_type_): _description_
        """        ''''''
        
        
        
        return partial(arm.get_joints)