import csv
import matplotlib.pyplot as plt
import argparse








parser = argparse.ArgumentParser()
parser.add_argument('-t', '--time', type=float, default=10)
parser.add_argument('-f', '--filename', type=str, default="trajectory.csv")

args = parser.parse_args()
print(args)


# Load the trajectory
trajectory = []

with open(args.filename, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        # Convert each row of strings to floats
        trajectory.append([float(element) for element in row])

print(trajectory[0][2])

# Extract columns
t = [ row[0] for row in trajectory ]
x = [ row[1] for row in trajectory ]
y = [ row[2] for row in trajectory ]
z = [ row[3] for row in trajectory ]





# Filters
def decaying_recursive_average(values, decay_factor):
    """
    Computes the decaying recursive average of a given array.

    Args:
        values (list or array-like): The input array of values to filter.
        decay_factor (float): The decay factor (0 < decay_factor <= 1).

    Returns:
        list: The filtered array of values.
    """
    if not (0 < decay_factor <= 1):
        raise ValueError("Decay factor must be in the range (0, 1].")

    filtered_values = []
    current_average = 0

    for i, value in enumerate(values):
        if i == 0:
            # Initialize with the first value
            current_average = value
        else:
            # Update using the decay factor
            current_average = decay_factor * value + (1 - decay_factor) * current_average
        
        filtered_values.append(current_average)

    return filtered_values




def decaying_recursive_average_with_cumulative_time(values, cumulative_times, base_decay_factor):
    """
    Computes the decaying recursive average with cumulative sampling time.

    Args:
        values (list or array-like): The input array of values to filter.
        cumulative_times (list or array-like): The cumulative times from the start of the experiment.
        base_decay_factor (float): The base decay factor (0 < base_decay_factor <= 1).

    Returns:
        list: The filtered array of values.
    """
    if not (0 < base_decay_factor <= 1):
        raise ValueError("Base decay factor must be in the range (0, 1].")
    if len(values) != len(cumulative_times):
        raise ValueError("Values and cumulative_times must have the same length.")

    filtered_values = []
    current_average = 0

    for i in range(len(values)):
        if i == 0:
            # Initialize with the first value
            current_average = values[i]
            filtered_values.append(current_average)
        else:
            # Compute the time interval between consecutive samples
            dt = cumulative_times[i] - cumulative_times[i - 1]
            effective_decay = base_decay_factor ** dt
            current_average = effective_decay * values[i] + (1 - effective_decay) * current_average
            filtered_values.append(current_average)

    return filtered_values


import math

class TimeInvariantFilter:
    def __init__(self, tau):
        """
        Initializes the filter.
        
        Args:
            tau (float): Time constant controlling the filter's smoothness.
        """
        self.tau = tau
        self.prev_filtered = None
        self.prev_time = None

    def update(self, value, current_time):
        """
        Updates the filter with a new value and timestamp.

        Args:
            value (float): The new input value.
            current_time (float): The current timestamp.

        Returns:
            float: The filtered value.
        """
        if self.prev_time is None:
            # First call: Initialize the filter with the input value
            self.prev_filtered = value
            self.prev_time = current_time
            return self.prev_filtered

        # Calculate time difference
        delta_t = current_time - self.prev_time

        # Calculate time-normalized decay factor
        alpha = 1 - math.exp(-delta_t / self.tau)

        # Update the filtered value
        self.prev_filtered = alpha * value + (1 - alpha) * self.prev_filtered

        # Update previous time
        self.prev_time = current_time

        return self.prev_filtered

def apply_time_invariant_filter_to_trajectory(trajectory, timestamps, tau):
    """
    Applies the TimeInvariantFilter to a trajectory.

    Args:
        trajectory (list or array-like): List of trajectory values to filter.
        timestamps (list or array-like): Corresponding timestamps for each value.
        tau (float): Time constant controlling the filter's smoothness.

    Returns:
        list: The filtered trajectory.
    """
    if len(trajectory) != len(timestamps):
        raise ValueError("Trajectory and timestamps must have the same length.")

    # Initialize the TimeInvariantFilter
    filter = TimeInvariantFilter(tau)

    # Apply the filter to the trajectory
    filtered_trajectory = []
    for value, timestamp in zip(trajectory, timestamps):
        filtered_value = filter.update(value, timestamp)
        filtered_trajectory.append(filtered_value)

    return filtered_trajectory






z_filt_dec_avg = decaying_recursive_average(z, decay_factor=0.1)
z_filt_dec_avg_with_time = decaying_recursive_average_with_cumulative_time(z, t, 1e-15)
z_filt_time_inv = apply_time_invariant_filter_to_trajectory(z, t, 0.25)



plt.figure(figsize=(10,6))

plt.plot(t,z)
plt.plot(t,z_filt_dec_avg)
plt.plot(t,z_filt_dec_avg_with_time)
plt.plot(t,z_filt_time_inv)

plt.show()
