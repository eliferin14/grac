#!/usr/bin/env python3

import rospy
import numpy as np
from functools import partial

from gesture_utils.frameworks.control_mode_interface import ControlModeInterface

#import moveit_commander
from moveit_commander import MoveGroupCommander, RobotCommander

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from actionlib_msgs.msg import GoalStatus




class ActionBasedControlMode(ControlModeInterface):
    
    framework_name = "Abstract action server framework"
    
    left_gestures_list = np.array(['fist', 'one', 'two', 'three', 'four', 'palm'])
    
    # To be substituted with a value from the parameter server
    joint_velocity = np.pi/12 # [rad/s]
    ee_velocity = 0.1 # [m/s]
        
    # Create robot and movegroup commanders
    robot_commander = RobotCommander()
    group_commander = MoveGroupCommander("manipulator")   

    # Action client
    live_mode = False
    try:
        live_mode = rospy.get_param('/detection_node/live')
    except KeyError:
        pass
    server_name = '/scaled_pos_traj_controller/follow_joint_trajectory' if live_mode else '/arm_controller/follow_joint_trajectory'
    client = actionlib.SimpleActionClient(server_name, FollowJointTrajectoryAction) 
    
    
    
    
    
    def __init__(self, group_name="manipulator", time_step=0.1, max_velocity_scaling=0.3):
        """Initiaslise the RobotCommander and the MoveGroupCommander. 
        Saves the names of the joints and the joint limits in as instance variables

        Args:
            group_name (str, optional): Name of he movegroup, as specified in the URDF. Defaults to "manipulator".
        """        ''''''
        
        super().__init__()

        #self.client = actionlib.SimpleActionClient(self.server_name, FollowJointTrajectoryAction)
        
        # Set the timestep
        self.time_step = time_step
        
        # Set the maximum allowable velocity scaling factor
        self.max_velocity_scaling = max_velocity_scaling
        
        # Extract joint names and limits
        self.joint_names = self.group_commander.get_active_joints()
        self.joint_position_limits, self.joint_velocity_limits = self.get_joint_limits()
        
        rospy.loginfo(f"Joint names: {self.joint_names}")
        rospy.loginfo(f"Joint position limits: {self.joint_position_limits}")
        rospy.loginfo(f"Joint velocity limits: {self.joint_velocity_limits}")
        
        
        
        
        
        
        
    def interpret_gestures(self, *args, **kwargs):
        raise NotImplementedError
    
    
    
    
    
    def generate_action_goal(self, target_joints_configuration, joints_names, time=-1):
        """Generate the goal object to be sent to the action server

        Args:
            target_joints_configuration (_type_): Array of joint positions
            joints_names (_type_): Array of joint names

        Returns:
            FollowJointTrajectoryGoal: the goal object
        """        ''''''
        
        if not self.check_joint_limits(target_joints_configuration):
            rospy.logwarn("Joint limits not respected")
            return FollowJointTrajectoryGoal()
        
        point = JointTrajectoryPoint()
        point.positions = target_joints_configuration

        time_step = self.time_step if time==-1 else time
        point.time_from_start = rospy.Duration(time_step)
        
        trajectory = JointTrajectory()
        trajectory.joint_names = joints_names
        trajectory.points = [point]
        
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory        
        
        #rospy.loginfo(goal)
        
        return goal
    
    
    
    
    
    def stop(self):
        """Clears the active or pending goal in the action server
        """        
        
        # Cancel the goal trajectory
        state = self.client.get_state()
        if state in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
            self.client.cancel_all_goals()
            current_joints = self.group_commander.get_current_joint_values()
            goal = self.generate_action_goal(current_joints, self.joint_names, time=0.05)
            self.client.send_goal(goal)
    
        
    
    
    def get_joint_limits(self):
        """Get the joint maximum and minimum posiition from the movegroup

        Returns:
            list: the list containing the limits
        """            
        
        joint_position_limits = []
        joint_velocity_limits = []
        for joint in self.joint_names:
            #print(robot_commander.get_joint(joint).bounds())
            joint_position_limits.append( self.robot_commander.get_joint(joint).bounds() )
            
            # Get velocity constraints by calling the service
            joint_velocity_limits.append(rospy.get_param('/robot_description_planning/joint_limits/{}/max_velocity'.format(joint)))
            
        return joint_position_limits, joint_velocity_limits
    
    
    
    
    def check_joint_limits(self, joint_target):
        """Compares the target values to the joint limits

        Args:
            joint_target (_type_): list of target positions

        Returns:
            bool: True if each joint is within its bounds
        """        
        
        result = True

        if joint_target is None:
            return False
        
        for t, l in zip(joint_target, self.joint_position_limits):
            if t < l[0] or t > l[1]:
                result = False
        
        return result
    
    
    
    
    
    
    def get_velocity_scaling(self, lhl, rhl, mapping='linear', **kwargs):
        
        scaling = 0
        
        # 0 < hands distance < 1
        lh_wrist_coord = lhl[0][0:1]
        rh_wrist_coord = rhl[0][0:1]
        hands_distance = np.linalg.norm( rh_wrist_coord-lh_wrist_coord )
        
        if mapping == 'linear':
            scaling = hands_distance

        elif mapping == 'logarithmic':
            a = kwargs['a']
            b = kwargs['b']
            c = kwargs['c']
            d = kwargs['d']

            scaling = c + (d-c)*(np.log(hands_distance) - np.log(a))/(np.log(b) - np.log(a))

        elif mapping == 'exponential':
            a = kwargs['a']
            b = kwargs['b']
            c = kwargs['c']
            d = kwargs['d']

            scaling = c + (d-c)*(np.exp(hands_distance) - np.exp(a))/(np.exp(b) - np.exp(a))

        assert scaling > 0
            
        return scaling

    
    
    
    
    
if __name__ == "__main__":
    
    rospy.init_node("action_client_node")
    ac = ActionBasedControlMode()