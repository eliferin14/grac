#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import mediapipe as mp

from gesture_control.msg import plot
from gesture_utils.ros_utils import convert_ROSpoints_to_XYZarrays


# Define the figure
fig = plt.figure(figsize=(18,6))
pose_ax = fig.add_subplot(131, projection='3d')
left_ax = fig.add_subplot(132, projection='3d')
right_ax = fig.add_subplot(133, projection='3d')

# Set titles and stuff
pose_ax.set_title("Pose")
right_ax.set_title("Right hand")
left_ax.set_title("Left hand")

# Initialize data as empty lists
rhl_x, rhl_y, rhl_z = [], [], []
lhl_x, lhl_y, lhl_z = [], [], []
pl_x, pl_y, pl_z = [], [], []

# Initialize artist for the plot(s)
pl_scatter = pose_ax.scatter(pl_x, pl_y, pl_z, c='g')
lhl_scatter = left_ax.scatter(lhl_x, lhl_y, lhl_z, c='r')
rhl_scatter = right_ax.scatter(rhl_x, rhl_y, rhl_z, c='b')

pose_line_collection = Line3DCollection([], colors='gray')
pose_lines = pose_ax.add_collection3d(pose_line_collection)
right_line_collection = Line3DCollection([], colors='black')
right_lines = right_ax.add_collection3d(right_line_collection)
left_line_collection = Line3DCollection([], colors='black')
left_lines = left_ax.add_collection3d(left_line_collection)




# Define pose connection list and landmark blacklist
mp_pose = mp.solutions.pose
pose_connection_list = list(mp_pose.POSE_CONNECTIONS)
pose_landmarks_blacklist = [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22]

# Define hand connection list
mp_hands = mp.solutions.hands
hands_connection_list = list(mp_hands.HAND_CONNECTIONS)





def update_data_callback(msg):
    
    global rhl_x, rhl_y, rhl_z, lhl_x, lhl_y, lhl_z, pl_x, pl_y, pl_z
    
    # Extract data
    rhl_ros = msg.rh_landmarks
    lhl_ros = msg.lh_landmarks
    pl_ros = msg.pose_landmarks
    
    # Convert points array to numpy matrices
    rhl_x, rhl_y, rhl_z = convert_ROSpoints_to_XYZarrays(rhl_ros)
    lhl_x, lhl_y, lhl_z = convert_ROSpoints_to_XYZarrays(lhl_ros)
    pl_x, pl_y, pl_z = convert_ROSpoints_to_XYZarrays(pl_ros)
    
    # Rotate points by 90deg about x
    pl_x, pl_y, pl_z = rot_x_90(pl_x, pl_y, pl_z)
    rhl_x, rhl_y, rhl_z = rot_x_90(rhl_x, rhl_y, rhl_z)
    lhl_x, lhl_y, lhl_z = rot_x_90(lhl_x, lhl_y, lhl_z)
    
    rospy.logdebug("Data updated")
    
    
    
    
def update_scatter_and_lines(x, y, z, connections, blacklist=[]):
    
    # Create connection list
    lines = []
    for start, end in connections:
        
        # Skip blacklisted landamrks
        if start in blacklist or end in blacklist: continue
        if start > len(x) or end > len(x): continue
        
        lines.append([
            [x[start], y[start], z[start]],
            [x[end], y[end], z[end]]
        ])
        
    # Filter landmarks
    x_filtered, y_filtered, z_filtered = [], [], []
    for i in range(len(x)):
        if i not in blacklist:
            x_filtered.append(x[i])
            y_filtered.append(y[i])
            z_filtered.append(z[i])
            
    return x_filtered, y_filtered, z_filtered, lines


import numpy as np

def get_bounding_cube(x, y, z):
    if len(x) == 0: return 0,0,0,1,1,1
    
    # Find the minimum and maximum values of x, y, z
    x_min, y_min, z_min = np.min(x), np.min(y), np.min(z)
    x_max, y_max, z_max = np.max(x), np.max(y), np.max(z)
    
    # Calculate the side length of the cube (largest range across any axis)
    cube_side = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Return the min and max values for each axis that define the cube
    return x_min, y_min, z_min, (x_min + cube_side), (y_min + cube_side), (z_min + cube_side)


def calculate_centered_bounding_cube(x, y, z):
    if len(x) == 0: return 0,0,0,1,1,1
    # Convert inputs to numpy arrays (in case they are lists)
    x, y, z = np.array(x), np.array(y), np.array(z)
    
    # Calculate the average (mean) of x, y, z
    mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
    
    # Find the minimum and maximum values of x, y, z
    x_min, y_min, z_min = np.min(x), np.min(y), np.min(z)
    x_max, y_max, z_max = np.max(x), np.max(y), np.max(z)
    
    # Calculate the side length of the cube (largest range across any axis)
    cube_side = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Adjust the cube to be centered around the average point
    center_offset = np.array([mean_x, mean_y, mean_z])
    
    # Calculate the new min and max values based on the center and cube side length
    min_point = center_offset - cube_side / 2
    max_point = center_offset + cube_side / 2
    
    return min_point[0], min_point[1], min_point[2], max_point[0], max_point[1], max_point[2]
    
    
def rot_x_90(x,y,z):
    if len(x) > 0:
        temp = z
        z = [-1*y_ for y_ in y]
        y = temp
    return x, y, z
    
    
def update_plot_callback(f):
    
    global rhl_x, rhl_y, rhl_z, lhl_x, lhl_y, lhl_z, pl_x, pl_y, pl_z
    
    # Set the new data in the pose plot
    px, py, pz, plines = update_scatter_and_lines(pl_x, pl_y, pl_z, pose_connection_list, pose_landmarks_blacklist)
    pl_scatter._offsets3d = (px, py, pz)
    pose_line_collection.set_segments(plines)
    # Set limits
    x_min, y_min, z_min, x_max, y_max, z_max = calculate_centered_bounding_cube(px, py, pz)
    pose_ax.set_xlim([x_min, x_max])
    pose_ax.set_ylim([y_min, y_max])
    pose_ax.set_zlim([z_min, z_max])
    
    
    # Set the new data in the hands plot
    rx, ry, rz, rlines = update_scatter_and_lines(rhl_x, rhl_y, rhl_z, hands_connection_list)
    rhl_scatter._offsets3d = (rx, ry, rz)
    right_line_collection.set_segments(rlines)    
    # Set limits
    x_min, y_min, z_min, x_max, y_max, z_max = calculate_centered_bounding_cube(rx, ry, rz)
    right_ax.set_xlim([x_min, x_max])
    right_ax.set_ylim([y_min, y_max])
    right_ax.set_zlim([z_min, z_max])
    
    lx, ly, lz, llines = update_scatter_and_lines(lhl_x, lhl_y, lhl_z, hands_connection_list)
    lhl_scatter._offsets3d = (lx, ly, lz)
    left_line_collection.set_segments(llines)   
    # Set limits
    x_min, y_min, z_min, x_max, y_max, z_max = calculate_centered_bounding_cube(lx, ly, lz)
    left_ax.set_xlim([x_min, x_max])
    left_ax.set_ylim([y_min, y_max])
    left_ax.set_zlim([z_min, z_max])
    
    
    return [
        pl_scatter,
        pose_line_collection,
        rhl_scatter,
        right_line_collection,
        lhl_scatter,
        left_line_collection
    ]
    
    

def listen_plot_topic():
    
    # Initialize node
    rospy.init_node("plotting")
    
    # Subscribe to the draw topic
    rospy.Subscriber('/plot_topic', plot, update_data_callback)
    
    # Define animation
    global fig
    animation = FuncAnimation(fig, update_plot_callback, interval=100, blit=False)
    
    # Show the plot
    plt.show()
    
    # Keep the listener spinning
    rospy.spin()
    
    
    

if __name__ == "__main__":
    try:
        listen_plot_topic()
    except rospy.ROSInterruptException:
        pass