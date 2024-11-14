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
    
    framework_name = "Joint"
    
    gesture_to_joint_list = np.array(['fist', 'one', 'two', 'three', 'four', 'palm'])
    
    def __init__(self):
        
        # In the beginning, no joint is selected
        self.selected_joint = None
        
        self.move_once_flag = False
        
        return 
    
    
    
    
    
    def interpret_gestures(
        self, 
        arm: Arm, 
        right_gesture, 
        left_gesture,
        rh_landmarks,
        lh_landmarks):
               
        # The left hand selects the joint
        candidate_selected_joint = np.where(self.gesture_to_joint_list == left_gesture)[0]
        
        # If no joint is selected, do nothing
        if candidate_selected_joint.size == 0:
            return partial(super().dummy_callback)
        
        if candidate_selected_joint != self.selected_joint:
            rospy.loginfo(f"Joint {candidate_selected_joint[0]} selected")
        
        self.selected_joint = candidate_selected_joint[0]
            
        
        # Get the joint limits
        joint_limits = get_joint_limits(arm, self.selected_joint)
        rospy.loginfo(f"Joint limits for joint {self.selected_joint}: {joint_limits}")
        
        # The right hand selects the angle
        angle = np.pi/64
        
        # Get current joint configuration
        current_joints = arm.get_joints()
        #print(current_joints)
        #print(current_joints[self.selected_joint])
        
        # Define the new position
        target_joints = current_joints
        #print(type(target_joints[self.selected_joint]))
        target_joints[self.selected_joint] = angle + current_joints[self.selected_joint]
        
        # The move_joints function requires a JointState object
        target_state = JointState()
        joint_name = arm.arm_interface.moveg.get_joints()[self.selected_joint]
        joint_position = angle + current_joints[self.selected_joint]
        print(f"Target joint position: {joint_position}")
        assert joint_position > joint_limits[0] and joint_position < joint_limits[1]
        target_state.name = [joint_name]
        target_state.position = [joint_position]
        
        # Move
        if not self.move_once_flag:
            #self.move_once_flag = True
            return partial(arm.move_joints, joints=target_state)
                
        return partial(arm.get_joints)
        