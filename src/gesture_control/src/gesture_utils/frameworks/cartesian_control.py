import rospy
import numpy as np
from functools import partial

from gesture_utils.frameworks.base_framework import BaseFrameworkManager

from gesture_utils.ros_utils import convert_ROSpoints_to_matrix

from sami.arm import Arm, EzPose
from sensor_msgs.msg import JointState


    

class CartesianFrameworkManager(BaseFrameworkManager):
    
    framework_name = "World relative"
    
    # List that defines how dof are selected
    gesture_to_dof_list = np.array(['fist', 'one', 'two', 'three', 'four', 'palm'])
    
    
    
    
    def __init__(self):
        
        # Angle of each movement
        self.default_angle = np.pi/16
        self.default_step = 0.1
        
        # In the beginning, no joint is selected
        self.selected_dof = None
        
        # Initialise the variable that remebers previous gesture (of right hand)
        self.old_rhg = None
        
        # 
        self.move_once_flag = False
        
        return 
    
    
    
    
    
    def interpret_gestures(self, *args, **kwargs):        
        
        # Extract parameters from kwargs
        arm = kwargs['arm']
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        rhl = kwargs['rhl']
        lhl = kwargs['lhl']
        
        # Assert that everyithing is in place
               
        # The left hand selects the joint
        candidate_selected_dof = np.where(self.gesture_to_dof_list == lhg)[0]
        
        # If no joint is selected (left hand is not in any of the fist->palm gestures), do nothing
        if candidate_selected_dof.size == 0:
            return partial(super().dummy_callback)
        
        # Change selected joint
        if candidate_selected_dof != self.selected_dof: rospy.loginfo(f"Joint {candidate_selected_dof[0]} selected")
        self.selected_dof = candidate_selected_dof[0]
        
        # Get the workspace limits
        
        # The right hand selects the angle/distance
        angle = 0
        distance = 0
        if rhg == 'one': 
            distance = self.default_step
            angle = self.default_angle         # positive movement
        elif rhg == 'two':             
            distance = -self.default_step
            angle = -self.default_angle      # negative movement
        
        # In the first iteration old_rhg is None: change it and return immediately
        if self.old_rhg is None:
            self.old_rhg = rhg
            return partial(super().dummy_callback)
        
        # The movement happens only if the previous gesture was 'fist' and the new gesture is 'one' or 'two'
        if self.old_rhg != 'fist':
            rospy.loginfo(f"Not moving because old_rhg is {self.old_rhg}")
            self.old_rhg = rhg
            return partial(super().dummy_callback)
        self.old_rhg = rhg  # Save new gesture for next iteration
        if rhg != 'one' and rhg != 'two':
            rospy.loginfo(f"Not moving because old_rhg is {self.old_rhg} but rhg is {rhg}")
            return partial(super().dummy_callback)
        
        # Define the target position
        if self.selected_dof < 3: 
            delta = distance
        else:
            delta = angle
            
        # Create an list of all zeros
        target_values = [0,0,0,0,0,0]
        
        # Add the delta in the correct position and create the target with EzPose
        target_values[self.selected_dof] = delta
        target_dpose = EzPose(*target_values)
        
        """ # Assert the joint limits are respected 
        # TODO: try catch or equivalent
        if joint_position < joint_limits[0] or joint_position > joint_limits[1]:
            return partial(super().dummy_callback)
        #assert joint_position > joint_limits[0] and joint_position < joint_limits[1] """
        
        # Move 
        return partial(arm.move_pose_relative_world, dpose=target_dpose)
        
        if not self.move_once_flag:
            #self.move_once_flag = True
            return partial(arm.move_joints, joints=target_state)
                
        # Just to know we are in the correct place
        return partial(arm.get_joints)
        