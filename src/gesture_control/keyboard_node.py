#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from pynput.keyboard import Key, Listener
import numpy as np
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

# Initialize ROS node
rospy.init_node('keyboard_listener', anonymous=True)
pub = rospy.Publisher('keyboard_input', String, queue_size=10)

# Create arm object
from sami.arm import Arm, EzPose
arm = Arm('ur10e_moveit', group='manipulator')

# Joint names
joint_names = [
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
]

# Define the client
client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
client.wait_for_server()

# Define the trajectory object
trajectory = JointTrajectory()
trajectory.joint_names = joint_names

# Send the initial goal with no points
goal = FollowJointTrajectoryGoal()
goal.trajectory = trajectory

# Send the initial empty trajectory
rospy.loginfo("Sending initial empty trajectory...")
client.send_goal(goal)
client.wait_for_result()

rospy.loginfo("Initial trajectory sent, now sending incremental points...")




t = time.time()

angle_step = np.pi * 1/64




def on_press(key):
    try:
        # Print the key that is pressed
        rospy.loginfo(f'Key {key.char} pressed')
        pub.publish(f'Key {key.char} pressed')
        
    except AttributeError:
        # Handle special keys
        rospy.loginfo(f'Special key {key} pressed')
        pub.publish(f'Special key {key} pressed')
        
    # Get current joint configuration
    current_joints = arm.get_joints()
    #print(current_joints)
    joint_target = current_joints
    
    try:
        key.char
    except AttributeError:
        return
        
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
        
    # Create a trajectory point
    point = JointTrajectoryPoint()
    point.positions = joint_target
    point.time_from_start = rospy.Duration(0.1)
    
    # Update trajectory object
    trajectory.points = [point]
    
    # Update goal and send
    goal.trajectory = trajectory
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
