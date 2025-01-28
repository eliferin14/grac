#!/usr/bin/env python3

import rospy
import numpy as np
from functools import partial
import time

from gesture_utils.frameworks.base_framework import BaseFrameworkManager
from gesture_utils.frameworks.action_base_framework import ActionClientBaseFramework
from gesture_utils.frameworks.cartesian_action import CartesianActionFrameworkManager

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

import tf2_ros
from tf.transformations import quaternion_multiply, quaternion_about_axis, quaternion_matrix

from geometry_msgs.msg import Point






class ExponentialMovingAverage():

    def __init__(self, tau):
        """
        Initializes the filter.
        
        Args:
            tau (float): Time constant controlling the filter's smoothness.
        """
        self.tau = tau
        self.prev_filtered = None
        self.prev_time = None

    def update(self, value, current_time):
        """
        Updates the filter with a new value and timestamp.

        Args:
            value (float): The new input value.
            current_time (float): The current timestamp.

        Returns:
            float: The filtered value.
        """
        if self.prev_time is None:
            # First call: Initialize the filter with the input value
            self.prev_filtered = value
            self.prev_time = current_time
            return self.prev_filtered

        # Calculate time difference
        delta_t = current_time - self.prev_time

        # Calculate time-normalized decay factor
        alpha = 1 - np.exp(-delta_t / self.tau)

        # Update the filtered value
        self.prev_filtered = alpha * value + (1 - alpha) * self.prev_filtered

        # Update previous time
        self.prev_time = current_time

        return self.prev_filtered
    
    def reset(self):
        self.prev_time = None
        self.prev_filtered = None


class PositionFilter():

    def __init__(self, tau_x, tau_y, tau_z, tau_rx, tau_ry, tau_rz):

        # Initialise all the filters
        self.filters = [
            ExponentialMovingAverage(tau_x), 
            ExponentialMovingAverage(tau_y), 
            ExponentialMovingAverage(tau_z),
            #ExponentialMovingAverage(tau_rx), 
            #ExponentialMovingAverage(tau_ry), 
            #ExponentialMovingAverage(tau_rz)
        ]

    def update(self, new_position, current_time):
        self.filtered_value = []

        for i, f in enumerate(self.filters):
            self.filtered_value.append( f.update(new_position[i], current_time) )

        assert len(self.filtered_value) == len(new_position)

        return self.filtered_value
    
    def reset(self):
        for f in self.filters:
            f.reset()

    







class HandMimicFrameworkManager( CartesianActionFrameworkManager ):
    
    framework_name = "Mimic hand"
    
    left_gestures_list = ['one', 'two', 'three', 'four']
    scaling_list_length = len(left_gestures_list)

    # Initialise filters
    position_filter = PositionFilter(0.1,0.1,0.1,0.001,0.001,0.001)
    
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

        self.publlisher = rospy.Publisher('Hand_tracking', Point, queue_size=10)
        
        
        
        
        
    def interpret_gestures(self, *args, **kwargs):
        
        # Extract parameters from kwargs
        rhg = kwargs['rhg']
        lhg = kwargs['lhg']
        pl = kwargs['pl']
        rhwl = kwargs['rhwl']
        
        
        
        
        
        ############## ACTION SELECTION #######################
    
        # If the lhg is not in the list, do nothing
        # Check the right hand: move only when it is in 'fist'
        if lhg not in self.left_gestures_list    or   not rhg == 'fist' :
            # Reset the flag
            self.is_mimicking = False

            # Reset all the filters
            self.position_filter.reset()

            return partial(self.stop)      
        
        # Read the hand position
        #hand_current_position = pl[15]               # Right wrist
        hand_current_position = rhwl[0]              # Right wrist
        hand_current_orientation = None
        
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

            # Store the starting time
            self.start_time = time.time()   
            
            # Flip the flag
            self.is_mimicking = True
            
            # Do nothing
            return partial(self.stop)
            
        
        
        
        
        ############## TARGET DEFINITION #######################
            
        # Select the scaling
        # TODO Do it in a fancy way with the left hand gesture
        scaling_factor = 0.2

        # Measure the time that has passed from the beginning of the mimicking (needed for filtering)
        current_time = time.time() - self.start_time

        # Apply filtering to the hand position
        #filtered_position = self.position_filter.update(hand_current_position, current_time)
        filtered_position = hand_current_position
        rospy.loginfo(f"{filtered_position[0]:.3f}\t{filtered_position[1]:.3f}\t{filtered_position[2]:.3f}")

        point = Point()
        point.x = filtered_position[0]
        point.y = filtered_position[1]
        point.z = filtered_position[2]
        self.publlisher.publish(point)

        return self.dummy_callback
        
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
    
    