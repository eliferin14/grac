import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default="trajectories.npy")
args = parser.parse_args()
print(args)

loaded_data = np.load(args.filename)
trajectories_tensor = loaded_data['trajectory_tensor']
print(f"Loaded tensor with shape {trajectories_tensor.shape}")


# Extract trajectories
hand_raw_trajectory = trajectories_tensor[:,0,:]
hand_filtered_trajectory = trajectories_tensor[:,1,:]
robot_target_trajectory = trajectories_tensor[:,2,:]
robot_measured_trajectory = trajectories_tensor[:,3,:]

robot_target_trajectory -= robot_target_trajectory[0]
robot_measured_trajectory -= robot_measured_trajectory[0]

s1, s2, s3, s4 = 0.1, 0.5, 1, 3

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


# Plot the trajectory
fig, (ax_hand, ax_robot) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

plot_trajectories_with_bounding_cube(ax_hand, 
                                     trajectories=[hand_raw_trajectory, hand_filtered_trajectory],
                                     title="Hand position",
                                     labels=['Unfiltered', 'Filtered'],
                                     colors=['k','r'],
                                     markers=['', '']
                                     )

plot_trajectories_with_bounding_cube(ax_robot, 
                                     trajectories=[robot_target_trajectory*s1, robot_target_trajectory*s2, robot_target_trajectory*s3, robot_target_trajectory*s4],
                                     title="Robot target position",
                                     labels=['scaling = 0.1', 'scaling = 0.5', 'scaling = 1', 'scaling = 3'],
                                     colors=['skyblue', 'royalblue', 'blue', 'navy'],
                                     markers=['', '', '', '']
                                     )



# Show the plot
plt.tight_layout()
plt.show()