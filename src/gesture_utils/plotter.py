import numpy as np
import matplotlib.pyplot as plt

# Define the function
def linear_mapping(x):
    return x
    

def logarithmic_mapping(x, a, b, c, d):
    y = np.zeros(x.shape[0])
    for i, val in enumerate(x):
        if val < a: y[i] = c
        elif val > b: y[i] = d
        else: y[i] = c + (d - c) * (np.log(val) - np.log(a)) / (np.log(b) - np.log(a))
    return y


def exponential_mapping(x, a, b, c, d):
    y = np.zeros(x.shape[0])
    for i, val in enumerate(x):
        if val < a: y[i] = c
        elif val > b: y[i] = d
        else: y[i] = c + (d - c) * (np.exp(val) - np.exp(a)) / (np.exp(b) - np.exp(a))
    return y

# Parameters
a, b = 0.2, 0.9
c, d = 0.01, 1
x = np.linspace(0,1, 500)

# Compute y values
y_lin = linear_mapping(x)
y_log = logarithmic_mapping(x, a, b, c, d)
y_exp = exponential_mapping(x, a, b, c, d)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y_lin, label="Linear Mapping")
plt.plot(x, y_log, label="Logarithmic Mapping")
plt.plot(x, y_exp, label="Exponential Mapping")
plt.title("Mapping Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True)
plt.legend()
plt.show()