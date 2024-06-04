import matplotlib.pyplot as plt

# Define the two points (x, y)
point1 = (0, 0)
point2 = (1, 1)

# Create a figure
plt.figure()

# Plot the two points
plt.plot(*point1, 'ro')  # Red point
plt.plot(*point2, 'ro')  # Red point

# Draw the arrow
plt.arrow(point1[0], point1[1], point2[0] - point1[0], point2[1] - point1[1], 
          head_width=0.05, head_length=0.1, fc='black', ec='black')

# Set plot limits
plt.xlim(-1, 2)
plt.ylim(-1, 2)

# Show the plot
plt.show()
