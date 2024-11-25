import rospy
import numpy as np
from functools import partial

from gesture_utils.frameworks.base_framework import BaseFrameworkManager

from gesture_utils.ros_utils import convert_ROSpoints_to_matrix

from sami.arm import Arm, EzPose
from sensor_msgs.msg import JointState






def get_joint_limits(arm:Arm, index):
    joints_list = arm.arm_interface.moveg.get_joints()
    joint_name = joints_list[index]
    
    robot = arm.arm_interface.robot
    joint_limits = robot.get_joint(joint_name).bounds()
    
    return joint_limits
    

class JointFrameworkManager(BaseFrameworkManager):
    
    framework_name = "Joint relative"
    
    # List that defines how joints are selected
    gesture_to_joint_list = np.array(['fist', 'one', 'two', 'three', 'four', 'palm'])
    
    
    
    
    
    def __init__(self):
        
        # Angle of each movement
        self.default_angle = np.pi/16
        
        # In the beginning, no joint is selected
        self.selected_joint = None
        
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
        candidate_selected_joint = np.where(self.gesture_to_joint_list == lhg)[0]
        
        # If no joint is selected (left hand is not in any of the fist->palm gestures), do nothing
        if candidate_selected_joint.size == 0:
            return partial(super().dummy_callback)
        
        # Change selected joint
        if candidate_selected_joint != self.selected_joint: rospy.loginfo(f"Joint {candidate_selected_joint[0]} selected")
        self.selected_joint = candidate_selected_joint[0]
        
        # Get the joint limits
        joint_limits = get_joint_limits(arm, self.selected_joint)
        rospy.loginfo(f"Joint limits for joint {self.selected_joint}: {joint_limits}")
        
        # The right hand selects the angle
        angle = 0
        if rhg == 'one': angle = self.default_angle         # positive movement
        elif rhg == 'two': angle = -self.default_angle      # negative movement
        
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
        
        # Get current joint configuration
        current_joints = arm.get_joints()
        
        # The move_joints function requires a JointState object
        target_state = JointState()
        
        # Get joint name: get_joints() returns a list of names, one for each joint. Note that there are more that 6 "joints" in the movegroup
        joint_name = arm.arm_interface.moveg.get_joints()[self.selected_joint]
        
        # Define new position for the selected joint
        joint_position = angle + current_joints[self.selected_joint]
        print(f"Target joint position: {joint_position}")
        
        # Assert the joint limits are respected 
        # TODO: try catch or equivalent
        if joint_position < joint_limits[0] or joint_position > joint_limits[1]:
            return partial(super().dummy_callback)
        #assert joint_position > joint_limits[0] and joint_position < joint_limits[1]
        
        # Store the target in the JointState object
        target_state.name = [joint_name]
        target_state.position = [joint_position]
        
        # Move 
        return partial(arm.move_joints, joints=target_state)
        
        if not self.move_once_flag:
            #self.move_once_flag = True
            return partial(arm.move_joints, joints=target_state)
                
        # Just to know we are in the correct place
        return partial(arm.get_joints)
        