import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse

# python3 timestamps_analisys.py -f cartesian1.csv -g interpret -b 3 -n 2 -cm "Joint control" -lhg "one" "two" "three" "four" "fist" "palm" -rhg "one" "two"



parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default="processing_times.csv")
parser.add_argument('-g', '--gesture', type=str, default="landmarks")
parser.add_argument('-cm', '--control_mode', nargs='+', type=str, default="all")
parser.add_argument('-lhg', type=str, nargs='+',  default="all")
parser.add_argument('-rhg', type=str, nargs='+', default="all")
parser.add_argument('-b', '--bin_size', type=float, default=1)
parser.add_argument('-n', '--num_hands', type=int, default=-1)
parser.add_argument('-t', '--title', type=str, default="Default title")
parser.add_argument('-th', '--threshold', type=float, default=0.0)
parser.add_argument('-sp', '--skip_plot', action="store_true")

args = parser.parse_args()
print(args)

# Load the file
df = pd.read_csv(args.filename)
#print(df.head)

# Convert seconds to milliseconds
col_list = ["capture", "landmarks", "gestures", "interpret", "drawing", "ik", "jacobian"]
df[col_list] = df[col_list] * 1000

# Apply filters
if args.control_mode != 'all':
    df = df[ df['control_mode'].isin(args.control_mode) ]
    print(f"Rows after filtering control mode: {df.shape[0]}")
if args.lhg != 'all':
    df = df[ df['lhg'].isin(args.lhg) ]
    print(f"Rows after filtering lhg: {df.shape[0]}")
if args.rhg != 'all':
    df = df[ df['rhg'].isin(args.rhg) ]
    print(f"Rows after filtering rhg: {df.shape[0]}")
if args.num_hands > 0:
    df = df[ df['hands'] == args.num_hands]
    print(f"Rows after filtering num_hands: {df.shape[0]}")



# Calculate total time
total = df[col_list].sum(axis=1)
df['total'] = total
#print(df['total'].shape)

# Function to compute bins
def get_bin_edges(column, bin_range):
    bin_min = min(column) - (min(column) % bin_range)
    bin_max = max(column) - (max(column) % bin_range) + bin_range * 2
    return np.linspace(bin_min, bin_max, int((bin_max - bin_min) / bin_range) + 1)

def get_bin_edges_fixed_bins(column, num_bins):
    bin_min = min(column)
    bin_max = max(column)
    return np.linspace(bin_min, bin_max, num_bins + 1)


def get_bin_edges_constrained_bins(column, allowed_bin_sizes=[0.5, 1, 2, 5]):
    data_min = min(column)
    data_max = max(column)
    data_range = data_max - data_min

    best_bin_size = None
    best_num_bins = float('inf')  # Initialize with a very large number

    for bin_size in allowed_bin_sizes:
        num_bins = data_range / bin_size
        if num_bins >= 10:
            if abs(num_bins - 10) < abs(best_num_bins - 10):
                best_bin_size = bin_size
                best_num_bins = num_bins

    # Handle the case where no bin size results in at least 10 bins
    if best_bin_size is None:
        best_bin_size = max(allowed_bin_sizes)  # Choose the largest allowed bin size
        best_num_bins = data_range / best_bin_size
        print("Warning: No allowed bin size resulted in at least 10 bins. Using largest allowed bin size.")


    bin_min = data_min - (data_min % best_bin_size)  # Round down for consistent behavior
    bin_max = data_max - (data_max % best_bin_size) + best_bin_size  # Round up to include max
    num_bins = int((bin_max - bin_min) / best_bin_size)

    return np.linspace(bin_min, bin_max, num_bins + 1)

def plot_histogram(data, bins, color, mean_value, mean_color, edgecolor, threshold=0.0):
    # Create the histogram with density=False to get counts
    bin_count, bin_edges, _ = plt.hist(data, bins=bins, density=False)  # Get the counts, don't plot it

    # Calculate bin widths (assuming uniform bin widths)
    bin_width = bin_edges[1] - bin_edges[0]

    # Calculate probabilities
    probabilities = bin_count / len(data)
    print(abs(sum(probabilities)))
    assert abs(sum(probabilities)-1) < 1e-4

    # Apply the threshold mask:  Only plot bins where probability >= threshold
    mask = probabilities >= threshold #Create the filter

    # Apply the mask to bin centers and probabilities
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    masked_bin_centers = bin_centers[mask]
    masked_probabilities = probabilities[mask]
    masked_bin_edges = bin_edges[:-1][mask] #The start of the bin has to be filtered

    #Clear the plot and use plt.bar to display a probability chart
    plt.clf()
    plt.bar(masked_bin_centers, masked_probabilities, width=bin_width, color=color, edgecolor=edgecolor)


    # Plot the mean line (important: adjust label if units are different)
    plt.axvline(mean_value, color=mean_color, linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f} ms')

# Extract the target column
column = args.gesture
bin_size = args.bin_size

# Example of removing from multiple columns
def remove_outliers_iqr_multiple(df, column_names):
    """Removes outliers from multiple DataFrame columns using the IQR method.

    Args:
        df (pd.DataFrame): The DataFrame.
        column_names (list): A list of column names to remove outliers from.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.
    """
    filtered_df = df.copy()  # Create a copy to avoid modifying the original
    for column in column_names:
        Q1 = filtered_df[column].quantile(0.1)
        Q3 = filtered_df[column].quantile(0.9)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_df = filtered_df[(filtered_df[column] >= lower_bound) & (filtered_df[column] <= upper_bound)]
    return filtered_df

df = remove_outliers_iqr_multiple(df, [column])
print(f"Rows after removing outliers: {df.shape[0]}")
    
data = df[column].dropna()
mean_value = data.mean()

# Compute bin counts
bins = get_bin_edges(data, bin_size)
bins = get_bin_edges_fixed_bins(data, 10)
bins = get_bin_edges_constrained_bins(data, allowed_bin_sizes=[0.1, 0.2, 0.5, 1, 2, 5])
print(f"Bins: {bins}")

# Create plot object
plt.figure(figsize=(8, 5))

# Plotting parameters
hist_colors = ['gold', 'coral', 'skyblue']
line_colors = ['orange', 'red', 'blue']
edgecolor = '0.25'

plt.grid(True, linestyle='--', alpha=0.7)

# Plot the histogram
plot_histogram(data, bins, hist_colors[2], mean_value, line_colors[2], edgecolor, threshold=0)

# Formatting
plt.xlabel(f'Elapsed time (ms)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title(args.title, fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.title}.png",
            dpi=300,         # High resolution for printing
            bbox_inches="tight", # Remove extra whitespace
            transparent=False,  # No transparency (background will be white)
            facecolor='white') # Ensure the background is white

if not args.skip_plot: plt.show()
