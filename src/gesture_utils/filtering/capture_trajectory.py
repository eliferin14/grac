#!/usr/bin/env python3

import os
print("Current working directory:", os.getcwd())

import sys
sys.path.append('/home/iris/AR-ur10e/ros-ws/src/grac/src')

from gesture_utils.gesture_detector import GestureDetector

import rospy
rospy.init_node('camera_streamer')
rate = rospy.Rate(30)





from gesture_utils.gesture_detector import GestureDetector
from gesture_utils.drawing_utils import draw_hand, draw_pose, denormalize_landmarks
import cv2
import numpy as np
import roslib.packages
import os
import argparse
import time
import csv







cam = cv2.VideoCapture(2)

# Find the model directory absolute path
model_realtive_path = "src/gesture_utils/training/exported_model"
package_path = roslib.packages.get_pkg_dir('gesture_control')
model_absolute_path = file_path = os.path.join(package_path, model_realtive_path)

# Define the detector object
detector = GestureDetector(
    model_absolute_path,
    0.2
)







parser = argparse.ArgumentParser()
parser.add_argument('-t', '--time', type=float, default=10)
parser.add_argument('-f', '--filename', type=str, default="trajectory.csv")

args = parser.parse_args()
print(args)

# Initialise the array
trajectory = []





capturing = False

while not rospy.is_shutdown():
    
    ret, frame = cam.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    
    # Process frame (detect hands and pose)
    detector.process(frame)
    
    # Get gestures
    rh_gesture, lh_gesture = detector.get_hand_gestures()
    rhl = detector.right_hand_landmarks_matrix
    rhlw = detector.right_hand_world_landmarks_matrix

    # Start only after the signal
    if not capturing and rh_gesture != 'pick':
        continue

    if not capturing:
        start_time = time.time()
        capturing = True

    if len(rhl) == 0: continue
    
    # Get right hand wrist
    rhw_from_hand = rhlw[0]  

    # Get elapsed time
    elapsed_time = time.time() - start_time
    
    # Build the trajectory
    tuple = (elapsed_time, rhw_from_hand[0], rhw_from_hand[1], rhw_from_hand[2])
    rospy.loginfo(tuple)
    trajectory.append( tuple )
    
    if elapsed_time > args.time:
        break
    
    # De-normalize landmarks (for drawing)
    height, width = frame.shape[0], frame.shape[1]
    rhl_pixel = denormalize_landmarks(rhl, width, height)
    
    # Draw stuff on the frame
    draw_hand(frame, rhl_pixel, (255,0,0), (255,255,255))
    
    cv2.imshow("Live feed", frame)
    cv2.waitKey(1)     

cam.release()



# Save to .csv
file_path = '/home/iris/AR-ur10e/ros-ws/src/grac/src/gesture_utils/filtering/' + args.filename

# Write to the CSV file
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(trajectory)

print(f"Array saved to {file_path}")



























































    
    
    
    
""" print(trajectory_from_pose)


    

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_trajectory(ax, trajectory):
    # Plot the points
    ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], c='black')
    ax.scatter(trajectory[0, :], trajectory[1, :], trajectory[2, :], c='blue', marker='o')

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Enforce uniform scaling
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    all_limits = np.array([x_limits, y_limits, z_limits])
    center = np.mean(all_limits, axis=1)
    range_max = np.max(all_limits[:, 1] - all_limits[:, 0]) / 2

    ax.set_xlim3d([center[0] - range_max, center[0] + range_max])
    ax.set_ylim3d([center[1] - range_max, center[1] + range_max])
    ax.set_zlim3d([center[2] - range_max, center[2] + range_max])
    
    

# Create the figure and 3D axis
fig = plt.figure()
ax_pose = fig.add_subplot(121, projection='3d')
ax_hand = fig.add_subplot(122, projection='3d')

plot_trajectory(ax_pose, trajectory_from_pose)
plot_trajectory(ax_hand, trajectory_from_hand)

# Show the plot
plt.show() """