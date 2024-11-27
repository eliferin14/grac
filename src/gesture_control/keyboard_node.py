#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from pynput.keyboard import Key, Listener
import numpy as np
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time
from gesture_utils.trajectory_buffer import TrajectoryBuffer
from gesture_utils.ik_utils import IK
import argparse
from tf.transformations import quaternion_multiply, quaternion_about_axis




parser = argparse.ArgumentParser(description="ROS Node with Argument Parsing")
parser.add_argument("--pose_control", '-p', action='store_true', help="Enable pose control instead of joint control")
args = parser.parse_args()
print(args)
    



# Initialize ROS node
rospy.init_node('keyboard_listener', anonymous=True)

# Define the client
client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
client.wait_for_server()

# Create arm object
from sami.arm import Arm, EzPose
arm = Arm('ur10e_moveit', group='manipulator')

# Inverse kinematics
ik_solver = IK()

# Define the trajectory object
trajectory = JointTrajectory()
joint_names = [
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
]
trajectory.joint_names = joint_names

# Send the initial goal with no points
goal = FollowJointTrajectoryGoal()
goal.trajectory = trajectory

# Send the initial empty trajectory
rospy.loginfo("Sending initial empty trajectory...")
client.send_goal(goal)
client.wait_for_result()

rospy.loginfo("Initial trajectory sent, now sending incremental points...")



# Define discrete steps
angle_step = np.pi * 1/64
position_step = 0.01




def on_press(key):

    try:
        # Print the key that is pressed
        rospy.loginfo(f'Key {key.char} pressed')
        
    except AttributeError:
        # Handle special keys
        return
    
    
    # Define the point
    point = JointTrajectoryPoint()
    point.time_from_start = rospy.Duration(0.3)
    
    
    if args.pose_control:
        
        # Get current pose
        current_pose = arm.get_pose()
        pose_target = current_pose
        current_orientation = pose_target.orientation
        current_quaternion = [current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w]
        q_final = current_quaternion
        q_rotation = [0,0,0,1]
        
        # Based on the key pressed build the target
        if key.char == 'w':
            pose_target.position.x += position_step
        elif key.char == 'd':
            pose_target.position.y += position_step
        elif key.char == 'e':
            pose_target.position.z += position_step
        elif key.char == 'r':
            axis = [1,0,0]
            q_rotation = quaternion_about_axis(angle_step, axis)
        elif key.char == 'p':
            axis = [0,1,0]
            q_rotation = quaternion_about_axis(angle_step, axis)
        elif key.char == 'y':
            axis = [0,0,1]
            q_rotation = quaternion_about_axis(angle_step, axis)
        elif key.char == 's':
            pose_target.position.x -= position_step
        elif key.char == 'a':
            pose_target.position.y -= position_step
        elif key.char == 'q':
            pose_target.position.z -= position_step
        elif key.char == 'R':
            axis = [1,0,0]
            q_rotation = quaternion_about_axis(-angle_step, axis)
        elif key.char == 'P':
            axis = [0,1,0]
            q_rotation = quaternion_about_axis(-angle_step, axis)
        elif key.char == 'Y':
            axis = [0,0,1]
            q_rotation = quaternion_about_axis(-angle_step, axis)
        else:
            return
        
        # Calculate orientation
        q_final = quaternion_multiply(q_rotation, current_quaternion)
        pose_target.orientation.x = q_final[0]
        pose_target.orientation.y = q_final[1]
        pose_target.orientation.z = q_final[2]
        pose_target.orientation.w = q_final[3]
        
        # Compute inverse kinematics
        joint_target = ik_solver.solve_ik(pose_target)
        point.positions = joint_target
        
    else:
        
        # Get current joint configuration
        current_joints = arm.get_joints()
        joint_target = current_joints
        pass
    
        # QWERTY -> joints
        if key.char == 'q':
            joint_target[0] += angle_step
        elif key.char == 'w':
            joint_target[1] += angle_step
        elif key.char == 'e':
            joint_target[2] += angle_step
        elif key.char == 'r':
            joint_target[3] += angle_step
        elif key.char == 't':
            joint_target[4] += angle_step
        elif key.char == 'y':
            joint_target[5] += angle_step
            
        elif key.char == 'Q':
            joint_target[0] -= angle_step
        elif key.char == 'W':
            joint_target[1] -= angle_step
        elif key.char == 'E':
            joint_target[2] -= angle_step
        elif key.char == 'R':
            joint_target[3] -= angle_step
        elif key.char == 'T':
            joint_target[4] -= angle_step
        elif key.char == 'Y':
            joint_target[5] -= angle_step
        else:
            return
        
        point.positions = joint_target
    
    
    # Build the trajectory
    trajectory.points = [point]
    
    # Build the goal
    goal.trajectory = trajectory
    
    # Send the goal
    rospy.loginfo(f"Sending updated trajectory with {len(trajectory.points)} points...")
    client.send_goal(goal)
    

def on_release(key):
    if key == Key.esc:
        # Stop listener when the 'esc' key is pressed
        rospy.loginfo("Exiting the keyboard listener.")
        return False

def start_listener():
    # Start the listener for keyboard inputs
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
        
        
        

if __name__ == '__main__':
    try:
        rospy.loginfo("Keyboard listener node started.")
        start_listener()  # Start the keyboard listener
    except rospy.ROSInterruptException:
        pass
