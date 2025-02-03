import rospy
import numpy as np
from functools import partial

from grac.src.gesture_utils.frameworks.control_mode_interface import ControlModeInterface

from gesture_utils.ros_utils import convert_ROSpoints_to_matrix

from sami.arm import Arm, ArmMotionChain, EzPose





# Framework to build a motion chain
# Possible actions: set initial position, add new position, clear, execute
# As usual the left hand choose what action to perform:
    # fist -> set initial
    # one -> add position
    # two -> add position from known list
    # four -> clear
    # palm -> execute
    
# 2 could be implemented using the right hand as a visual menu selector

useful_lhg = ['fist', 'one', 'four', 'palm']
useful_rhg = ['none', 'pick']



class ChainFrameworkManager(ControlModeInterface):
    
    framework_name = "Motion chain"
    
    # Motion chain for the arm
    chain = ArmMotionChain()
    
    
    
    def __init__(self):
        super().__init__()
        return
        
        
        
    def interpret_gestures(self, *args, **kwargs):
        
        # Extract parameters from kwargs
        arm = kwargs['arm']
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        rhl = kwargs['rhl']
        lhl = kwargs['lhl']
        
        if lhg not in useful_lhg:
            return partial(super().empty_callback)
        
        # 
    
        return 