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
parser.add_argument('-b', '--bin_size', type=float, default=0.001)
parser.add_argument('-n', '--num_hands', type=int, default=-1)
parser.add_argument('-t', '--title', type=str, default="Default title")
parser.add_argument('-th', '--threshold', type=float, default=0.0)

args = parser.parse_args()
print(args)

# Load the file
df = pd.read_csv(args.filename)
print(df.head)

# Convert seconds to milliseconds
col_list = ["capture", "landmarks", "gestures", "interpret", "drawing", "ik", "jacobian"]
df[col_list] = df[col_list] * 1000

# Function to compute bins
def get_bin_edges(column, bin_range):
    bin_min = min(column) - (min(column) % bin_range)
    bin_max = max(column) - (max(column) % bin_range) + bin_range * 2
    return np.linspace(bin_min, bin_max, int((bin_max - bin_min) / bin_range) + 1)

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
    plt.axvline(mean_value, color=mean_color, linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')

# Extract the target column
column = args.gesture
bin_size = args.bin_size

# Apply filters
if args.control_mode != 'all':
    df = df[ df['control_mode'].isin(args.control_mode) ]
if args.lhg != 'all':
    df = df[ df['lhg'].isin(args.lhg) ]
if args.rhg != 'all':
    df = df[ df['rhg'].isin(args.rhg) ]
    
data = df[column].dropna()
mean_value = data.mean()

# Compute bin counts
bins = get_bin_edges(data, bin_size)
print(f"Bins: {bins}")

# Create plot object
plt.figure(figsize=(8, 5))

# Plotting parameters
hist_colors = ['gold', 'coral', 'teal']
line_colors = ['orange', 'red', 'blue']
edgecolor = '0.25'

# Plot the histogram
plot_histogram(data, bins, hist_colors[2], mean_value, line_colors[2], edgecolor, threshold=args.threshold)

# Formatting
plt.xlabel(f'Elapsed time (ms)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title(args.title, fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
