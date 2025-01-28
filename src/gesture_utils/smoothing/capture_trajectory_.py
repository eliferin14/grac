from gesture_utils.gesture_detector import GestureDetector
from gesture_utils.drawing_utils import draw_hand, draw_pose, denormalize_landmarks
import cv2
import numpy as np
import roslib.packages
import os
import argparse








cam = cv2.VideoCapture(3)

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
parser.add_argument('--trajectory_length', type=int, default=100)

args = parser.parse_args()
print(args)






# Empty array of points
trajectory_from_pose = np.zeros((3,args.trajectory_length))
trajectory_from_hand = np.zeros((3,args.trajectory_length))
i = 0







while True:
    
    ret, frame = cam.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    
    # Process frame (detect hands and pose)
    detector.process(frame)
    
    # Get gestures
    rh_gesture, lh_gesture = detector.get_hand_gestures()
    rhl = detector.right_hand_landmarks_matrix
    rhlw = detector.right_hand_world_landmarks_matrix
    
    # Get pose landmarks
    pl = detector.pose_landmarks_matrix
    if len(pl) == 0: continue
    if len(rhl) == 0: continue
    
    # Get right hand wrist
    rhw_from_pose = pl[15]  
    rhw_from_hand = rhlw[0]  
    
    # Build the trajectory
    if rh_gesture == 'palm' and i < args.trajectory_length:
        trajectory_from_pose[:,i] = rhw_from_pose
        trajectory_from_hand[:,i] = rhw_from_hand
        i += 1
        print(f"i = {i}")
    
    if i >= args.trajectory_length:
        break
    
    # De-normalize landmarks
    height, width = frame.shape[0], frame.shape[1]
    rhl_pixel = denormalize_landmarks(rhl, width, height)
    pl_pixel = denormalize_landmarks(pl, width, height)
    
    # Draw stuff on the frame
    draw_pose(frame, pl_pixel, (0,255,0), (255,255,255))
    draw_hand(frame, rhl_pixel, (255,0,0), (255,255,255))
    
    cv2.imshow("Live feed", frame)
    cv2.waitKey(1)     
       
    
    print(rhw_from_pose)
    
    
    
    
print(trajectory_from_pose)


    

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
plt.show()



