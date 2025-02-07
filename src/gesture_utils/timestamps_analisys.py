import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default="processing_times.csv")
parser.add_argument('-g', '--gesture', type=str, default="landmarks")
parser.add_argument('-b', '--bin_size', type=float, default=0.001)
parser.add_argument('-n', '--num-hands', type=int, default=-1)
args = parser.parse_args()
print(args)

# Load the file
df = pd.read_csv(args.filename)
print(df)

# Filter by number of hands
num_hands = args.num_hands
if num_hands > 0:
    df = df[df['hands'] == num_hands]
    print(df)
    
    
# Convert seconds to milliseconds
col_list = ["capture", "landmarks", "gestures", "interpret", "drawing"]
df[col_list] = df[col_list]*1000
    
    
    
def get_bin_counts(column, bin_range):
    
    bin_min = min(column) - (min(column) % bin_range)
    bin_max = max(column) - (max(column) % bin_range) + bin_range*2
    
    # Define the bins using the range
    #bins = list(range(bin_min, bin_max, bin_range))
    bins = np.linspace(bin_min, bin_max, int((bin_max-bin_min)/bin_range)+1)
    print(f"Bins: {bins}")

    # Cut the data into bins and count the values in each bin
    binned_data = pd.cut(column, bins=bins)

    # Count the number of occurrences in each bin
    bin_counts = binned_data.value_counts().sort_index()
    
    return bin_counts

column = args.gesture
bin_size = args.bin_size

bin_counts = get_bin_counts(df[column], bin_size)
print(bin_counts)

# Plot the histogram
ax = bin_counts.plot(kind='bar', width=0.8, align='center', color='skyblue', edgecolor='gray', zorder=2)
ax.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=1)
plt.xlabel(f'{column} ranges (ms)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Histogram of {column}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()