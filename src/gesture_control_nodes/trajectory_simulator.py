#!/usr/bin/env python3

import rospy
import numpy as np
import time
import argparse
import roslib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gesture_utils.control_modes.hand_mimic_control_mode import HandMimicControlMode






def plot_trajectories_with_bounding_cube(ax, trajectories, title, labels, colors, markers, draw_arrows=True):
    """
    Plots multiple 3D trajectories on the given ax and calculates the smallest bounding cube.
    
    Parameters:
    - ax: Matplotlib 3D axis object
    - trajectories: list or array of shape (num_trajectories, num_points, 3)
    """
    # Plot all trajectories
    for traj, label, color, marker in zip(trajectories,labels,colors, markers):
        x, y, z = traj[:,0], traj[:,1], traj[:,2]
        ax.plot(x, y, z, label=label, color=color, marker=marker)

        if draw_arrows:
            u, v, w = np.diff(x), np.diff(y), np.diff(z)
            #ax.quiver(x[:-1], y[:-1], z[:-1], u, v, w, color=color, arrow_length_ratio=0.5)

    # Compute the bounding cube
    all_points = np.vstack(trajectories)  # Flatten all trajectories into a single array
    min_vals = all_points.min(axis=0)
    max_vals = all_points.max(axis=0)
    center = (min_vals + max_vals) / 2
    max_range = (max_vals - min_vals).max() / 2  # Half the side length of the cube

    # Define cube vertices
    cube_vertices = np.array([
        [center[0] - max_range, center[1] - max_range, center[2] - max_range],
        [center[0] - max_range, center[1] - max_range, center[2] + max_range],
        [center[0] - max_range, center[1] + max_range, center[2] - max_range],
        [center[0] - max_range, center[1] + max_range, center[2] + max_range],
        [center[0] + max_range, center[1] - max_range, center[2] - max_range],
        [center[0] + max_range, center[1] - max_range, center[2] + max_range],
        [center[0] + max_range, center[1] + max_range, center[2] - max_range],
        [center[0] + max_range, center[1] + max_range, center[2] + max_range]
    ])

    # Plot cube vertices
    ax.scatter(cube_vertices[:, 0], cube_vertices[:, 1], cube_vertices[:, 2], color='red', s=0)

    # Ensure equal axis scaling
    #ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([center[0] - max_range, center[0] + max_range])
    ax.set_ylim([center[1] - max_range, center[1] + max_range])
    ax.set_zlim([center[2] - max_range, center[2] + max_range])

    # Axes names
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_title(title)
    ax.legend()






parser = argparse.ArgumentParser()
parser.add_argument('-it', '--input-trajectory', type=str, default='test_input')
#parser.add_argument('-ot', '--output-trajectory', type=str, default='test_output.npz')
parser.add_argument('-l', '--live', action="store_true")

args = parser.parse_args()
rospy.loginfo(args)


rospy.init_node('trajectory_simulator', anonymous=True)

if args.live:
    rospy.set_param('/detection_node/live', True)
else:
    rospy.set_param('/detection_node/live', False)

hmcm = HandMimicControlMode()


# Find the absolute path
data_realtive_path = "data/trajectories"
package_path = roslib.packages.get_pkg_dir('gesture_control')
data_absolute_path = os.path.join(package_path, data_realtive_path)

suffix = "_live.npz" if args.live else "_sim.npz" 
input_filename = os.path.join(data_absolute_path, args.input_trajectory+".npz")
output_filename = os.path.join(data_absolute_path, args.input_trajectory+suffix)
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
robot_target_trajectory = trajectories_tensor[:,3,:]
rospy.loginfo(f"Loaded robot target trajectory with shape {robot_target_trajectory.shape}")


# Create the empty trajectory object
# hand_raw, hand, robot_raw, robot_smoothed, robot_measured
output_trajectory_tensor = np.zeros((1,4,3))


# Get current robot orientation (it will be needed to correctly generate the target)
robot_pose = hmcm.group_commander.get_current_pose().pose
robot_position, robot_orientation = hmcm.convert_pose_to_p_q(robot_pose)
robot_target_position = robot_position

# Loop to send the command to the robot, either real or simulated
for i, (t, robot_delta_position) in enumerate(zip(timestamps, robot_delta_trajectory)):
    
    # Start the chrono
    start_t = time.time()
    
    # Measure the curernt robot position
    robot_position, _ = hmcm.convert_pose_to_p_q( hmcm.group_commander.get_current_pose().pose )
    
    # Generate the target pose
    robot_target_position += robot_delta_position
    robot_target_pose = hmcm.convert_p_q_to_pose(robot_target_position, robot_orientation)
    
    points = np.array([trajectories_tensor[i,0], trajectories_tensor[i,1], robot_target_position, robot_position])
    output_trajectory_tensor = np.append(output_trajectory_tensor, [points], axis=0)
    
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
    
# Remove the empty element created at the beginning
output_trajectory_tensor = output_trajectory_tensor[1:]
assert output_trajectory_tensor.shape[0] == timestamps.shape[0]
rospy.loginfo(output_trajectory_tensor.shape)

# Extract trajectories
hand_raw_trajectory = output_trajectory_tensor[:,0,:]
hand_filtered_trajectory = output_trajectory_tensor[:,1,:]
robot_raw_trajectory = output_trajectory_tensor[:,2,:]
robot_measured_trajectory = output_trajectory_tensor[:,3,:]


# Plot the trajectory
fig, (ax_hand, ax_robot) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

plot_trajectories_with_bounding_cube(ax_hand, 
                                     trajectories=[hand_raw_trajectory, hand_filtered_trajectory],
                                     title="Hand position",
                                     labels=['Raw', 'Filtered'],
                                     colors=['k','r'],
                                     markers=['', '']
                                     )

plot_trajectories_with_bounding_cube(ax_robot, 
                                     trajectories=[robot_raw_trajectory, robot_measured_trajectory],
                                     title="Robot position",
                                     labels=['Raw', 'Measured'],
                                     colors=['k', 'b'],
                                     markers=['x', '']
                                     )



# Show the plot
plt.tight_layout()
plt.show()

np.savez(output_filename, trajectory_tensor=output_trajectory_tensor)







