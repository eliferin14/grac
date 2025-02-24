import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-vx', type=float)
parser.add_argument('-vy', type=float)
parser.add_argument('-vz', type=float)
parser.add_argument('-dx', type=float)
parser.add_argument('-dy', type=float)
parser.add_argument('-dz', type=float)

args = parser.parse_args()
print(args)

v_max = np.array([args.vx, args.vy])#, args.vz])
d = np.array([args.dx, args.dy])#, args.dz])

# Normalize vectors
n = v_max / np.linalg.norm(v_max)
d_n = d / np.linalg.norm(d)

# Get comparison coefficients
alphas = np.abs( d_n / n )
print(f"alphas: {alphas}")

i = np.argmax(alphas)
print(f"i: {i}")
    
beta = np.abs(v_max[i] / d_n[i])
print(f"beta: {beta}")

d_max = beta * d_n
print(f"d_max: {d_max}")

d_max_2 = np.array([np.abs(v_max[0]*d_n[0]), np.abs(v_max[1]*d_n[1])])

for c, v in zip(d_max, v_max):
    assert c <= v
    
def plot_2d_vectors_arrows(vectors, labels=None):
    """
    Plots a list of 2D vectors using matplotlib, with arrows.

    Args:
        vectors: A list of 2D vectors, where each vector is a tuple or list of (x, y) coordinates.
        labels: An optional list of labels, one for each vector.
    """

    x_coords = [v[0] for v in vectors]
    y_coords = [v[1] for v in vectors]
    
    colors = ['b', 'g', 'r', 'y', 'k', 'gray']

    plt.figure()

    # Plot the vectors as arrows
    if labels:
        for i, (x, y, c) in enumerate(zip(x_coords, y_coords, colors)):
            plt.arrow(0, 0, x, y, head_width=0.1, head_length=0.2, fc=c, ec=c, label=labels[i], length_includes_head=True)
    else:
        for x, y in zip(x_coords, y_coords):
            plt.arrow(0, 0, x, y, head_width=0.1, head_length=0.2, fc=c, ec=c,length_includes_head=True)

    plt.axhline(0, color='black', linewidth=0.5)  # Add x-axis
    plt.axvline(0, color='black', linewidth=0.5)  # Add y-axis

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Vectors as Arrows')
    plt.grid(True)
    if labels:
        plt.legend()
    plt.axis('equal')  # Ensure equal scaling for x and y axes
    plt.show()

#exit()

# Example usage:
vectors = [v_max, d, n, d_n, d_max, d_max_2]
labels = ['Max velocity', 'direction', 'direction of max velocity', 'normalized direction', 'result', 'result2']
plot_2d_vectors_arrows(vectors, labels)
