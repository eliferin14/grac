#!/usr/bin/env python

import rospy
import numpy as np
from functools import partial

from sami.arm import Arm
from gesture_utils.frameworks.base_framework import BaseFrameworkManager
from gesture_utils.frameworks.action_base_framework import ActionClientBaseFramework
from gesture_utils.frameworks.cartesian_world_action import CartesianActionFrameworkManager

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

import tf2_ros
from tf.transformations import quaternion_multiply, quaternion_about_axis, quaternion_matrix










class HandMimicFrameworkManager( CartesianActionFrameworkManager ):
    
    framework_name = "Mimic hand"
    
    left_gestures_list = ['one']
    
    
    def __init__(self, group_name="manipulator"):
        
        super().__init__(robot_name, group_name)
        
        # Initialise the service caller
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
        # Declare the starting points for hand and robot
        self.hand_starting_point, self.robot_starting_point = None, None
        self.is_mimicking = False
        
        
        
        
        
    def interpret_gestures(self, *args, **kwargs):
        
        # Extract parameters from kwargs
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        
        
        
        
        
        ############## ACTION SELECTION #######################
    
        # If the lhg is not in the list, do nothing
        if lhg not in self.left_gestures_list:
            return partial(self.stop)
        
        # If the robot is not already mimicking, define the starting points and flip the flag
        if not self.is_mimicking:
            
            # Get current pose of the robot
            robot_current_pose = self.group_commander.get_current_pose().pose
            
            # Get current pose of the hand
            hand_current_position = None
            hand_current_orientation = None
            
            
        
        