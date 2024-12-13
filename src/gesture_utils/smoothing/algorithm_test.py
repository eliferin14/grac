import numpy as np    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gesture_utils.smoothing.moving_average import MovingAverage
from gesture_utils.smoothing.savitzski_golay import SavitzkyGolaySmoothing
from gesture_utils.smoothing.ccma import CurvatureCorrectedMovingAverage



# Generate original trajectroy and add noise
trajectory = np.array([ np.array([np.sin(x), np.cos(x), x]) for x in np.linspace(0, 10, 1000) ])
n_points = 100  # Number of points in the trajectory
trajectory = np.cumsum(np.random.randn(n_points, 3), axis=0)
noisy_trajectory = trajectory + np.random.normal(0, 0.01, trajectory.shape)  

# Moving average
moving_average = MovingAverage()
moving_average.set_params(window_size=5)
ma_trajectory = moving_average.smooth(noisy_trajectory)

# Savitzski-Golay
sg = SavitzkyGolaySmoothing()
sg_trajectory = sg.smooth(noisy_trajectory)

# CCMA
ccma = CurvatureCorrectedMovingAverage()
ccma_trajectory = ccma.smooth(noisy_trajectory)






  

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot the trajectory
ax.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory', color='black')
ax.plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], label='Noisy trajectory', color='red')
ax.plot(ma_trajectory[:, 0], ma_trajectory[:, 1], label='MA', color='green')
ax.plot(sg_trajectory[:, 0], sg_trajectory[:, 1], label='SG', color='blue')
ax.plot(ccma_trajectory[:, 0], ccma_trajectory[:, 1], label='CCMA', color='cyan')

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# Show the plot
plt.show()