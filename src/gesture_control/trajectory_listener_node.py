#!/usr/bin/env python3

import rospy
import roslib.packages
import numpy as np
from geometry_msgs.msg import Point
from gesture_control.msg import trajectories
from moveit_commander import MoveGroupCommander
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default="trajectories.npy")
args = parser.parse_args()
print(args)

# Find the model directory absolute path
data_realtive_path = "data"
package_path = roslib.packages.get_pkg_dir('gesture_control')
data_absolute_path = os.path.join(package_path, data_realtive_path)
file_absolute_path = os.path.join(data_absolute_path, args.filename)

# Create the tensor
num_trajectories = 5
trajectories_tensor = np.zeros((1,num_trajectories,3))    # Time, trajectory, coordinate

# Initialize the commander
group_commander = MoveGroupCommander("manipulator")



def callback(msg):

    global trajectories_tensor

    # Create the numpy matrix
    trajectories_matrix = np.zeros((num_trajectories,3))

    # Fill the matrix with the received points
    #rospy.loginfo("Received {} points:".format(len(msg.points)))
    for i, point in enumerate(msg.points):
        #rospy.loginfo("Point {} -> x: {:.2f}, y: {:.2f}, z: {:.2f}".format(i, point.x, point.y, point.z))

        point_array = np.array([point.x, point.y, point.z])
        trajectories_matrix[i] = point_array

    # Poll the robot position
    robot_pose = group_commander.get_current_pose().pose
    point_array = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])

    # Save the matrix into the tensor
    trajectories_tensor = np.append(trajectories_tensor, [trajectories_matrix], axis=0)

    rospy.loginfo(f"Matrix dimension: {trajectories_matrix.shape}; Tensor dimensions: {trajectories_tensor.shape}")




def shutdown_hook():
    # Remove the first row
    global trajectories_tensor
    trajectories_tensor = trajectories_tensor[1:]

    # Save the file
    rospy.loginfo(f"Recorded {trajectories_tensor.shape[0]} points per trajectory")
    rospy.loginfo(f"Saving trajectories to {file_absolute_path}...")
    np.save(file_absolute_path, trajectories_tensor)




def listener():
    rospy.init_node('trajectory_listener', anonymous=True)
    rospy.Subscriber('/hand_mimic_trajectories', trajectories, callback)  # Replace with your actual topic
    rospy.on_shutdown(shutdown_hook)
    rospy.spin()




if __name__ == '__main__':
    listener()