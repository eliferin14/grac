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
        raise NotImplementedError