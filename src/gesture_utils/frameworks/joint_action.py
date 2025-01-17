#!/usr/bin/env python

import rospy
import numpy as np
from functools import partial

from sami.arm import Arm
from gesture_utils.frameworks.base_framework import BaseFrameworkManager
from gesture_utils.frameworks.action_base_framework import ActionClientBaseFramework

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint





class JointActionFrameworkManager(ActionClientBaseFramework):
    
    framework_name = "Joint control (Action)"
    
    selected_joint_index = None
    
    
    
    def interpret_gestures(self, *args, **kwargs):
        
        
        # Extract parameters from kwargs
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
               
        
        
        
        
        ################ ACTION SELECTION ########################
        
        # The left hand selects the joint
        candidate_selected_joint = np.where(self.left_gestures_list == lhg)[0]
        
        # If no joint is selected (left hand is not in any of the fist->palm gestures), do nothing
        if candidate_selected_joint.size == 0:
            return partial(super().stop)
        
        # Change selected joint
        if candidate_selected_joint != self.selected_joint_index: rospy.loginfo(f"Joint {candidate_selected_joint[0]} selected")
        self.selected_joint_index = candidate_selected_joint[0]
        
        # The right hand selects the angle
        angle = 0
        if rhg == 'one': angle = self.angle_step         # positive movement
        elif rhg == 'two': angle = -self.angle_step      # negative movement
        else: return partial(super().stop)
        
        
        
        
        
        ################ TARGET DEFINITION #####################
        
        # Get the joint limits
        selected_joint_limits = self.joint_limits[self.selected_joint_index]
        
        # Get current joint configuration from the robot
        current_joints = self.group_commander.get_current_joint_values()
        
        # Calculate the target joint position
        target_joint_position = current_joints[self.selected_joint_index] + angle
        
        # Assert the joint limits are respected 
        if target_joint_position < selected_joint_limits[0] or target_joint_position > selected_joint_limits[1]:
            rospy.logwarn(f"Joint limits exceeded! {target_joint_position} not in [{selected_joint_limits[0]}, {selected_joint_limits[1]}]")
            return partial(super().stop)
        
        # Define the target joint positions
        target_joints = current_joints
        target_joints[self.selected_joint_index] = target_joint_position
        rospy.loginfo(target_joints)
        
        
        
        
        
        
        ################ SEND ACTION REQUEST ####################
        
        goal = self.generate_action_goal(target_joints, self.joint_names)
        action_request = partial(self.client.send_goal, goal=goal)        
        return action_request





    
    
    
    
    
    
if __name__ == "__main__":
    
    rospy.init_node("joint_actions_node")
    
    jafm = JointActionFrameworkManager()
    callback = jafm.interpret_gestures(lhg='fist', rhg='one')
    callback = jafm.interpret_gestures(lhg='none', rhg='one')
    print(callback)
    callback()