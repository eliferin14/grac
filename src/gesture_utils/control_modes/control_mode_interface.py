import rospy
from functools import partial





class ControlModeInterface:
    
    framework_name = "Control Mode Interface"
    
    def empty_callback(self):
        pass
    
    def interpret_gestures(self, *args, **kwargs):        
        raise NotImplementedError