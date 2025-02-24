import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.testing import assert_array_equal, assert_array_almost_equal

parser = argparse.ArgumentParser()
parser.add_argument('-sf', '--sim_filename', type=str)
parser.add_argument('-lf', '--live_filename', type=str)
args = parser.parse_args()
print(args)

def plot_trajectories_with_bounding_cube(ax, trajectories, title, labels, colors, markers, linestyles, draw_arrows=True):
    """
    Plots multiple 3D trajectories on the given ax and calculates the smallest bounding cube.
    
    Parameters:
    - ax: Matplotlib 3D axis object
    - trajectories: list or array of shape (num_trajectories, num_points, 3)
    """
    # Plot all trajectories
    for traj, label, color, marker, linestyle in zip(trajectories,labels,colors, markers, linestyles):
        x, y, z = traj[:,0], traj[:,1], traj[:,2]
        ax.plot(x, y, z, label=label, color=color, marker=marker, linestyle=linestyle)

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

# Load data from files
sim_data = np.load(args.sim_filename)['trajectory_tensor']
live_data = np.load(args.live_filename)['trajectory_tensor']

assert sim_data.shape == live_data.shape
print(f"Trajectory has {sim_data.shape[0]} points")

# Extract trajectories from data
hand_raw_traj = sim_data[:,0,:]
hand_traj = sim_data[:,1,:]
hand_raw_traj_live = live_data[:,0,:]
hand_traj_live = live_data[:,1,:]

sim_robot_target_traj = sim_data[:,2,:]
sim_robot_measured_traj = sim_data[:,3,:]
live_robot_target_traj = live_data[:,2,:]
live_robot_measured_traj = live_data[:,3,:]

# "Normalize" robot trajectories (let them start from [0,0,0])
sim_robot_target_traj -= sim_robot_target_traj[0]
sim_robot_measured_traj -= sim_robot_measured_traj[0]
live_robot_target_traj -= live_robot_target_traj[0]
live_robot_measured_traj -= live_robot_measured_traj[0]

assert_array_equal(hand_raw_traj, hand_raw_traj_live)
assert_array_equal(hand_traj, hand_traj_live)
assert_array_almost_equal(sim_robot_target_traj, live_robot_target_traj)




# Plot the trajectory
fig, (ax_hand, ax_robot) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

plot_trajectories_with_bounding_cube(ax_hand, 
                                     trajectories=[hand_raw_traj, hand_traj],
                                     title="Hand position",
                                     labels=['Raw', 'Filtered'],
                                     colors=['k','r'],
                                     markers=['', ''],
                                     linestyles=['-', '-']
                                     )

plot_trajectories_with_bounding_cube(ax_robot, 
                                     trajectories=[sim_robot_target_traj, sim_robot_measured_traj, live_robot_measured_traj],
                                     title="Robot position",
                                     labels=['Target', 'Simulated', 'Real'],
                                     colors=['k', 'b', 'r'],
                                     markers=['', '', ''],
                                     linestyles=[':', '-', '-']
                                     )



# Show the plot
plt.tight_layout()
plt.show()