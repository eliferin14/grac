import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default="trajectories.npy")
args = parser.parse_args()
print(args)

trajectories_tensor = np.load(args.filename)
print(f"Loaded tensor with shape {trajectories_tensor.shape}")

# Define the parameter t
t = np.linspace(0, 4 * np.pi, 100)  # t goes from 0 to 4*pi

# Parametric equations for the spiral
x = np.cos(t)
y = np.sin(t)
z = t  # You can scale this if you want to change the height progression
test_trajectory = np.array([x,y,z])
print(test_trajectory.shape)


# Extract trajectories
hand_raw_trajectory = trajectories_tensor[:,0,:]
hand_filtered_trajectory = trajectories_tensor[:,1,:]
robot_raw_trajectory = trajectories_tensor[:,2,:]
robot_smoothed_trajectory = trajectories_tensor[:,3,:]
hand_delta_points = trajectories_tensor[:,4,:]
robot_delta_points = trajectories_tensor[:,5,:]

robot_measured_trajectory = trajectories_tensor[:,-1,:]

def plot_trajectories_with_bounding_cube(ax, trajectories, title, labels, colors, markers):
    """
    Plots multiple 3D trajectories on the given ax and calculates the smallest bounding cube.
    
    Parameters:
    - ax: Matplotlib 3D axis object
    - trajectories: list or array of shape (num_trajectories, num_points, 3)
    """
    # Plot all trajectories
    for traj, label, color, marker in zip(trajectories,labels,colors, markers):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=label, color=color, marker=marker)

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




# Plot the trajectory
fig, (ax_hand, ax_robot, ax_delta) = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})




# Plot the trajectory in 3D space
""" ax_hand.plot(*hand_raw_trajectory, label=f"Raw", marker='x')
ax_hand.plot(*hand_filtered_trajectory, label=f"Filtered")
#ax_hand.plot(*test_trajectory, label=f"Random")
ax_hand.set_xlabel('X')
ax_hand.set_ylabel('Y')
ax_hand.set_zlabel('Z')
ax_hand.set_title(f"Hand trajectory")
ax_hand.legend() 

# Plot the trajectory in 3D space
ax_robot.plot(*robot_raw_trajectory, label=f"Raw", marker='o')
ax_robot.plot(*robot_smoothed_trajectory, label=f"Smoothed")
ax_robot.plot(*robot_measured_trajectory, label=f"Measured", marker='x')
ax_robot.set_xlabel('X')
ax_robot.set_ylabel('Y')
ax_robot.set_zlabel('Z')
ax_robot.set_title(f"Robot trajectory")
ax_robot.legend()

# Plot the trajectory in 3D space
ax_delta.plot(*hand_delta_points, label=f"Hand", marker='o')
ax_delta.plot(*robot_delta_points, label=f"Robot", marker='x')
ax_delta.set_xlabel('X')
ax_delta.set_ylabel('Y')
ax_delta.set_zlabel('Z')
ax_delta.set_title(f"Delta vectors")
ax_delta.legend()"""



plot_trajectories_with_bounding_cube(ax_hand, 
                                     trajectories=[hand_raw_trajectory, hand_filtered_trajectory],
                                     title="Hand position",
                                     labels=['Raw', 'Filtered'],
                                     colors=['k','r'],
                                     markers=['x', '']
                                     )

plot_trajectories_with_bounding_cube(ax_robot, 
                                     trajectories=[robot_raw_trajectory, robot_smoothed_trajectory, robot_measured_trajectory],
                                     title="Robot position",
                                     labels=['Raw', 'Smoothed', 'Measured'],
                                     colors=['k','g', 'b'],
                                     markers=['x', 'o', '']
                                     )

plot_trajectories_with_bounding_cube(ax_delta, 
                                     trajectories=[hand_delta_points, robot_delta_points],
                                     title="Displacement vectors",
                                     labels=['Hand', 'Robot'],
                                     colors=['r','g'],
                                     markers=['x', '']
                                     )













# Show the plot
plt.tight_layout()
plt.show()