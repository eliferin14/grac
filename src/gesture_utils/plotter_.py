import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the piecewise functions
a = 0.2
b = 0.9

# Define the linear function
def linear_mapping(d):
    return d

# Define the piecewise linear function
def piecewise_linear_mapping(d, a, b):
    if d < a:
        return 0
    elif d < b:
        return (d - a) / (b - a)
    else:
        return 1

# Define the piecewise exponential function
def piecewise_exponential_mapping(d, a, b, base):
    if d < a:
        return 0
    elif d < b:
        return (base**(d)-base**(a))/(base**(b)-base**(a))  # Exponential growth
    else:
        return 1# Define the piecewise quadratic function
def piecewise_quadratic_mapping(d, a, b):
    if d < a:
        return 0
    elif d < b:
        return (d - a) ** 2 / (b - a) ** 2  # Quadratic between a and b
    else:
        return 1

# Generate the input range
d_values = np.linspace(0, 1, 500)

# Apply the functions to the input range
linear_values = linear_mapping(d_values)
piecewise_linear_values = np.array([piecewise_linear_mapping(d, a, b) for d in d_values])
piecewise_exponential_values = np.array([piecewise_exponential_mapping(d, a, b, 10) for d in d_values])
piecewise_quadratic_values = np.array([piecewise_quadratic_mapping(d, a, b) for d in d_values])

# Plot the functions
plt.figure(figsize=(8, 5))
plt.plot(d_values, linear_values, label='Linear Mapping', color='#7EC8E3', linewidth=2)
plt.plot(d_values, piecewise_linear_values, label='Piecewise Linear Mapping', color='#4F79A1', linewidth=2)
plt.plot(d_values, piecewise_exponential_values, label='Piecewise Exponential Mapping', color='#000080', linewidth=2)
#plt.plot(d_values, piecewise_quadratic_values, label='Piecewise Quadratic Mapping', linewidth=2)

# Plot a and b
plt.vlines([a,b], ymin=0, ymax=1, colors='k', linestyles=':', linewidth=2)

# Add x and y axes explicitly
plt.axhline(0, color='black',linewidth=1)  # Horizontal axis
plt.axvline(0, color='black',linewidth=1)  # Vertical axis


# Add labels and title
plt.xlabel(r'Hands distance $d$ (fraction of image width)', fontsize=12)
plt.ylabel(r'Scaling factor $\alpha$', fontsize=12)
plt.title(r'Comparison of mapping functions', fontsize=14)
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
