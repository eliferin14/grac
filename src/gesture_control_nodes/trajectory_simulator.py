#!/usr/bin/env python3

import rospy
import numpy as np
import time
import argparse
import roslib
import os

from gesture_utils.control_modes.hand_mimic_control_mode import HandMimicControlMode







rospy.init_node('trajectory_simulator', anonymous=True)

hmcm = HandMimicControlMode()

parser = argparse.ArgumentParser()
parser.add_argument('-it', '--input-trajectory', type=str, default='test_input.npz')
parser.add_argument('-ot', '--output-trajectory', type=str, default='test_output.npz')

args = parser.parse_args()
rospy.loginfo(args)


# Find the absolute path
data_realtive_path = "data/trajectories"
package_path = roslib.packages.get_pkg_dir('gesture_control')
data_absolute_path = os.path.join(package_path, data_realtive_path)

input_filename = os.path.join(data_absolute_path, args.input_trajectory)
output_filename = os.path.join(data_absolute_path, args.output_trajectory)
rospy.loginfo(f"Input trajectory file: {input_filename}")
rospy.loginfo(f"Output trajectory file: {output_filename}")

# Load the input trajectory
loaded_data = np.load(input_filename, allow_pickle=True)
timestamps = loaded_data['timestamps']
trajectories_tensor = loaded_data['trajectories_tensor']
rospy.loginfo(f"Loaded trajectory tensor with shape {trajectories_tensor.shape}")
rospy.loginfo(f"Loaded timestamps array with shape {timestamps.shape}")

# Extract the robot target trajectory
robot_delta_trajectory = trajectories_tensor[:,5,:]
rospy.loginfo(f"Loaded robot target trajectory with shape {robot_delta_trajectory.shape}")


# Get current robot orientation (it will be needed to correctly generate the target)
robot_pose = hmcm.group_commander.get_current_pose().pose
robot_position, robot_orientation = hmcm.convert_pose_to_p_q(robot_pose)




# Loop to send the command to the robot, either real or simulated
for i, (t, robot_delta_position) in enumerate(zip(timestamps, robot_delta_trajectory)):
    
    # Start the chrono
    start_t = time.time()
    
    # Generate the target pose
    robot_position += robot_delta_position
    robot_target_pose = hmcm.convert_p_q_to_pose(robot_position, robot_orientation)
    
    # Call IK
    target_joints = hmcm.compute_ik(robot_target_pose)
    
    # Generate goal
    goal = hmcm.generate_action_goal(target_joints, hmcm.joint_names)
    
    # Send goal
    hmcm.client.send_goal(goal)
    
    # If last iteration don't wait
    if i >= timestamps.shape[0]-1:
        break
    
    # Wait so that the trajectory is replicated correctly
    delta_t = (timestamps[i+1] - t).to_sec()
    rospy.loginfo(f"[{i}]: [{t}], {delta_t} -> {robot_delta_position}")    
    while time.time() - start_t < delta_t: pass
    











