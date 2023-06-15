import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras



def generate_sinusoidal_data(num_points, amplitude, frequency, noise_std_dev):
    x = np.linspace(0, 10, num_points)
    noise = np.random.normal(0, noise_std_dev, num_points)
    y = sinus_eq(amplitdue, frequency, x) + noise
    return x, y


def sinus_eq(amplitdue, frequency, x):
    phase_shift = np.random.uniform(0, 2 * np.pi)  # Random phase shift
    return amplitude * np.sin(frequency * x + phase_shift)


#make x y data
def generate_linear_data(num_points, gradient, intercept, noise_std_dev):
    rand_nums = np.random.rand(num_points)
    x = 10 * rand_nums
    noise = np.random.normal(0, noise_std_dev, num_points)
    y = linear_eq(x, gradient, intercept) + noise
    return x, y

def linear_eq(X, m, c):
    return m*X + c


# Example usage
num_points = 100  # Number of data points to generate
gradient = 5  # User-defined gradient
intercept = 5  # User-defined intercept
noise_std_dev = 1  # Standard deviation of the noise

X, Y = generate_linear_data(num_points, gradient, intercept, noise_std_dev)





loss_list = []


# Building the model
m = 0.1 #Initial guess for gradient
c = 0 #Initial guess for intercept
L = 0.001  # The learning Rate
epochs = 2000  # The number of iterations to perform gradient descent
n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = linear_eq(X, m, c)  # The current predicted value of Y
    loss = (-1/n) * sum(Y - Y_pred) #MSE loss
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    loss_list.append(loss)


# Plot the data and the fitted line
plt.scatter(X, Y)
plt.plot(X, m * X + c, color='red')
plt.xlabel('x')
plt.ylabel('y')
# Display fitted m, c, and loss
text = f'm = {m:.3f}\nc = {c:.3f}\nloss = {loss:.3f}'
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment='top')
# Show the plot
plt.show()


plt.plot(abs(np.array(loss_list)))
plt.xlabel('epoch')
plt.ylabel('lost')
plt.show()




plt.clf()
# Define the range of 'm' and 'c' values
m_values = np.linspace(0,10, 1000)  # Range of 'm' values
c_values = np.linspace(0,10, 1000)  # Range of 'c' values

# Initialize a matrix to store the loss values
loss_values = np.zeros((len(m_values), len(c_values)))

# Compute the loss for each combination of 'm' and 'c' values
for i, m in enumerate(m_values):
    for j, c in enumerate(c_values):
        Y_pred = linear_eq(X, m, c)  # The current predicted value of Y
        loss = abs((-1/n) * np.sum(Y - Y_pred))  # MSE loss)
        loss_values[i, j] = loss


# Plot the heatmap
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
im = ax.imshow(loss_values, cmap='rainbow', extent=[min(c_values), max(c_values), min(m_values), max(m_values)])

# Colorbar
cbar = fig.colorbar(im, ax=ax, label='Loss')

# Plot the circle
circle = plt.Circle((gradient, intercept), radius=0.5, color='blue', fill=False)
ax.add_artist(circle)

# Axis labels and title
ax.set_xlabel('c')
ax.set_ylabel('m')
ax.set_title('Heatmap of Loss')

# Adjust figure size for a square plot
fig.tight_layout(rect=[0, 0, 1, 1])

# Set aspect ratio to make it square
ax.set_aspect('equal')

plt.show()




# Create the grid of 'm' and 'c' values
M, C = np.meshgrid(m_values, c_values)

# Create the figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(C, M, np.log(loss_values), cmap='rainbow')

# Set labels and title
ax.set_xlabel('c')
ax.set_ylabel('m')
ax.set_zlabel('Log Loss')
ax.set_title('3D Plot of Loss')

plt.show()




