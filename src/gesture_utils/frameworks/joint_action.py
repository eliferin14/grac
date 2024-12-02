import rospy
import numpy as np
from functools import partial

from sami.arm import Arm
from gesture_utils.frameworks.base_framework import BaseFrameworkManager

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint




class JointActionFrameworkManager(BaseFrameworkManager):
    
    framework_name = "Joint control (Action)"
    
    # List that defines how joints are selected
    gesture_to_joint_list = np.array(['fist', 'one', 'two', 'three', 'four', 'palm'])
    
    # To be substituted with a value from the parameter server
    angle_step = np.pi / 64
    time_step = 0.5
    
    
    
    
    def __init__(self, arm=Arm('ur10e_moveit', group='manipulator')):
        
            
        # Client for the action server
        self.client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        #   self.client.wait_for_server(timeout=3)
    
        # Get joint names and limits from the arm object
        self.joint_names = arm.arm_interface.moveg.get_active_joints()
        self.joint_limits = JointActionFrameworkManager.get_joint_limits(arm.arm_interface.robot, self.joint_names)
        
        # Initialise variables
        self.old_rhg = None
        self.selected_joint_index = None
    
    
    def interpret_gestures(self, *args, **kwargs):
    
        # Extract parameters from kwargs
        arm = kwargs['arm']
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        
        
        
        
        ################ ACTION SELECTION ########################
        
        # The left hand selects the joint
        candidate_selected_joint = np.where(self.gesture_to_joint_list == lhg)[0]
        
        # If no joint is selected (left hand is not in any of the fist->palm gestures), do nothing
        if candidate_selected_joint.size == 0:
            return partial(super().dummy_callback)
        
        # Change selected joint
        if candidate_selected_joint != self.selected_joint_index: rospy.loginfo(f"Joint {candidate_selected_joint[0]} selected")
        self.selected_joint_index = candidate_selected_joint[0]
        
        # The right hand selects the angle
        angle = 0
        if rhg == 'one': angle = self.angle_step         # positive movement
        elif rhg == 'two': angle = -self.angle_step      # negative movement
        else: return partial(super().dummy_callback)
        
        
        
        
        ################# GOAL DEFINITION ######################
        
        # Get the joint limits
        selected_joint_limits = self.joint_limits[self.selected_joint_index]
        
        # Get current joint configuration from the robot
        current_joints = arm.get_joints()
        
        # Calculate the target joint position
        target_joint_position = current_joints[self.selected_joint_index] + angle
        
        # Assert the joint limits are respected 
        if target_joint_position < selected_joint_limits[0] or target_joint_position > selected_joint_limits[1]:
            rospy.logwarn(f"Joint limits exceeded! {target_joint_position} not in [{selected_joint_limits[0]}, {selected_joint_limits[1]}]")
            return partial(super().dummy_callback)
        
        # Define the target joint positions
        target_joints = current_joints
        target_joints[self.selected_joint_index] = target_joint_position
        
        # Define the trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start = rospy.Duration(self.time_step)
        
        # Define the trajectory object
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.points = [point]

        # Send the initial goal with no points
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        
        
        
        
        # Return the call to the action server
        return partial(self.client.send_goal, goal=goal)
    
    
    
    
    
    # You can call this function with JointActionFrameworkManager.get_joint_limits()
    def get_joint_limits(robot_commander, joint_names):
        
        joint_limits = []
        for joint in joint_names:
            #print(robot_commander.get_joint(joint).bounds())
            joint_limits.append( robot_commander.get_joint(joint).bounds() )
            
        return joint_limits