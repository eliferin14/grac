#!/usr/bin/env python3

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
    
    left_gestures_list = ['one', 'two', 'three', 'four']
    scaling_list_length = len(left_gestures_list)
    
    # TODO: Define this matrix properly
    camera_to_robot_tf = np.eye(3)
    
    
    def __init__(self, group_name="manipulator", min_scaling=0.001, max_scaling=1):
        
        super().__init__(group_name)
        
        # Initialise the service caller
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        
        # Declare the starting points for hand and robot
        self.hand_starting_position, self.robot_starting_position = None, None
        self.hand_starting_orientation, self.robot_starting_orientation = None, None
        self.is_mimicking = False
        
        # Calculate the scaling values
        scaling_list = np.logspace( np.log10(min_scaling), np.log10(max_scaling), self.scaling_list_length)
        rospy.loginfo(f"Scaling values: {scaling_list}")
        
        
        
        
        
    def interpret_gestures(self, *args, **kwargs):
        
        # Extract parameters from kwargs
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        pl = kwargs['pl']
        
        
        
        
        
        ############## ACTION SELECTION #######################
    
        # If the lhg is not in the list, do nothing
        # Check the right hand: move only when it is in 'fist'
        if lhg not in self.left_gestures_list    or   not rhg == 'fist' :
            self.is_mimicking = False
            return partial(self.stop)      
        
        # Read the hand position
        hand_current_position = np.copy(pl[15])              # Right wrist
        hand_current_orientation = None
        rospy.loginfo(hand_current_position)
        
        # If the robot is not already mimicking, define the starting points and flip the flag
        if not self.is_mimicking:
            
            rospy.logdebug("Saving starting postion of robot and hand")
            
            # This code is supposed to be executed only once, when the left hand is choosing the scaling AND the right hand is 'fist'
            
            # Get current pose of the robot and store it
            robot_pose = self.group_commander.get_current_pose().pose
            self.robot_starting_position, self.robot_starting_orientation = self.convert_pose_to_p_q(robot_pose)
            
            # Save the starting position of the hand
            self.hand_starting_position = hand_current_position
            self.hand_starting_orientation = None     
            
            # Flip the flag
            self.is_mimicking = True
            
            # Do nothing
            return partial(self.stop)
            
        
        
        
        
        ############## TARGET DEFINITION #######################
            
        # Select the scaling
        # TODO Do it in a fancy way with the left hand gesture
        scaling_factor = 0.2
        
        # Calculate the delta vector between the current and starting position of the right hand
        rospy.logdebug(f"Hand starting position: {self.hand_starting_position}")
        rospy.logdebug(f"Hand current position: {hand_current_position}")
        hand_delta_position = hand_current_position - self.hand_starting_position
        rospy.logdebug(f"Hand delta position: {hand_delta_position}")
        
        # Apply scaling to obtain the delta vector for the robot
        robot_delta_position = scaling_factor * hand_delta_position
        rospy.logdebug(f"Robot delta position before rotation: {robot_delta_position}")
        
        # Change of coordinates to have the axes of the camera aligned with the robot base frame
        robot_delta_position = self.camera_to_robot_tf @ robot_delta_position
        rospy.logdebug(f"Robot delta position: {robot_delta_position}")
        
        # Calculate the target pose for the robot (starting pose + scaled delta vector)
        robot_target_position = self.robot_starting_position + robot_delta_position
        robot_target_orientation = self.robot_starting_orientation
        robot_target_pose = self.convert_p_q_to_pose(robot_target_position, robot_target_orientation)
        rospy.logdebug(robot_target_pose)
        
        # Call the inverse kinematics service
        target_joints = self.compute_ik(robot_target_pose)
        rospy.logdebug(f"Target joints: {target_joints}")
        
        # Generate the goal 
        goal = self.generate_action_goal(target_joints, self.joint_names)
        
        # Send the goal to the action server
        action_request = partial(self.client.send_goal, goal=goal)        
        return action_request
            
            












if __name__ == "__main__":
    
    rospy.init_node("Hand_mimic", log_level=rospy.DEBUG)
    
    hmfm = HandMimicFrameworkManager()
    
    lhg = 'one'
    rhg = 'fist'
    pl = np.zeros((33,3), dtype=float)
    
    hmfm.interpret_gestures(lhg=lhg, rhg=rhg, pl=pl)
    
    pl[15] = np.random.uniform(-0.5,0.5,3)
    print(pl)
    
    callback = hmfm.interpret_gestures(lhg=lhg, rhg=rhg, pl=pl)
    callback()
    
    