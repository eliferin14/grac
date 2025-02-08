import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default="processing_times.csv")
parser.add_argument('-g', '--gesture', type=str, default="landmarks")
parser.add_argument('-b', '--bin_size', type=float, default=0.001)
parser.add_argument('-n', '--num-hands', type=int, default=-1)
args = parser.parse_args()

# Load the file
df = pd.read_csv(args.filename)

# Convert seconds to milliseconds
col_list = ["capture", "landmarks", "gestures", "interpret", "drawing"]
df[col_list] = df[col_list] * 1000

# Function to compute bins
def get_bin_counts(column, bin_range):
    bin_min = min(column) - (min(column) % bin_range)
    bin_max = max(column) - (max(column) % bin_range) + bin_range * 2
    return np.linspace(bin_min, bin_max, int((bin_max - bin_min) / bin_range) + 1)

# Extract the target column
column = args.gesture
bin_size = args.bin_size

# Handle different num_hands cases
if args.num_hands < 0:
    data = df[column].dropna()
    mean_value = data.mean()
elif args.num_hands in [0, 1, 2]:
    data = df[df['hands'] == args.num_hands][column].dropna()
    mean_value = data.mean()
elif args.num_hands == 12:
    data_1 = df[df['hands'] == 1][column].dropna()
    data_2 = df[df['hands'] == 2][column].dropna()
    data = pd.concat([data_1, data_2])  # For binning consistency
    mean_value_1 = data_1.mean()
    mean_value_2 = data_2.mean()

# Compute bin counts
bins = get_bin_counts(data, bin_size)

# Fit distributions (only if not comparing two histograms)
if args.num_hands != 12:
    dist_names = ['lognorm', 'gamma', 'weibull_min', 'expon']
    best_fit = None
    best_sse = float('inf')
    best_params = None

    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        params = dist.fit(data)

        # Compute goodness-of-fit
        pdf_fitted = dist.pdf(np.sort(data), *params)
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        sse = np.sum((hist - np.interp(bin_centers, np.sort(data), pdf_fitted)) ** 2)

        if sse < best_sse:
            best_sse = sse
            best_fit = dist_name
            best_params = params

    print(f"Best fit: {best_fit}")

# Plot
plt.figure(figsize=(8, 5))

if args.num_hands == 12:
    # Plot two histograms
    plt.hist(data_1, bins=bins, density=True, alpha=0.6, color='blue', edgecolor='black', label='Hands = 1')
    plt.hist(data_2, bins=bins, density=True, alpha=0.6, color='green', edgecolor='black', label='Hands = 2')

    # Vertical lines for means
    plt.axvline(mean_value_1, color='blue', linestyle=':', linewidth=3, label=f'Mean (Hands = 1): {mean_value_1:.2f} ms')
    plt.axvline(mean_value_2, color='green', linestyle=':', linewidth=2, label=f'Mean (Hands = 2): {mean_value_2:.2f} ms')

else:
    # Plot single histogram
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='gray', label='Histogram')

    # Plot best-fitting distribution
    dist = getattr(stats, best_fit)
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = dist.pdf(x, *best_params)
    plt.plot(x, pdf_fitted, 'r:', linewidth=2, label=f'Best Fit: {best_fit}')

    # Vertical line for mean
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f} ms')

# Formatting
plt.xlabel(f'{column} ranges (ms)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title(f'Histogram and Best Fit Distribution for {column}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
