#!/usr/bin/env python

import rospy
import numpy as np
from functools import partial

from gesture_utils.frameworks.base_framework import BaseFrameworkManager

import moveit_commander

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from actionlib_msgs.msg import GoalStatus




class ActionClientBaseFramework(BaseFrameworkManager):
    
    framework_name = "Abstract action server framework"
    
    left_gestures_list = np.array(['fist', 'one', 'two', 'three', 'four', 'palm'])
    
    # To be substituted with a value from the parameter server
    angle_step = np.pi / 64
    position_step = 0.01
    time_step = 0.5
    
    
    
    
    
    def __init__(self, robot_name="ur10e_moveit", group_name="manipulator"):
        
        super().__init__()
        
        # Client for the action server
        self.client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        
        # Create robot and movegroup commanders
        self.robot_commander = moveit_commander.RobotCommander()
        self.group_commander = moveit_commander.MoveGroupCommander("manipulator")
        
        # Extract joint names and limits
        self.joint_names = self.group_commander.get_active_joints()
        self.joint_limits = self.get_joint_limits()
        
        rospy.loginfo(self.joint_names)
        rospy.loginfo(self.joint_limits)
        
        
        
        
    def interpret_gestures(self, *args, **kwargs):
        raise NotImplementedError
    
    
    
    
    
    def generate_action_goal(self, target_joints_configuration, joints_names):
        
        if not self.check_joint_limits(target_joints_configuration):
            rospy.logwarn("Joint limits not respected")
            return partial(self.dummy_callback)
        
        point = JointTrajectoryPoint()
        point.positions = target_joints_configuration
        point.time_from_start = rospy.Duration(self.time_step)
        
        trajectory = JointTrajectory()
        trajectory.joint_names = joints_names
        trajectory.points = [point]
        
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory        
        
        #rospy.loginfo(goal)
        
        return goal
    
    
    
    
    
    def stop(self):
        
        # Cancel the goal trajectory
        state = self.client.get_state()
        if state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
            self.client.cancel_goal()
    
    
        
    
    
    def get_joint_limits(self):
        
        joint_limits = []
        for joint in self.joint_names:
            #print(robot_commander.get_joint(joint).bounds())
            joint_limits.append( self.robot_commander.get_joint(joint).bounds() )
            
        return joint_limits
    
    
    
    
    def check_joint_limits(self, joint_target):
        
        result = True
        
        for t, l in zip(joint_target, self.joint_limits):
            if t < l[0] or t > l[1]:
                result = False
        
        return result

    
    
    
    
    
if __name__ == "__main__":
    
    rospy.init_node("action_client_node")
    ac = ActionClientBaseFramework()