import matplotlib.pyplot as plt
import numpy as np

# Define the range of minutes within the interval [2, 4]
minutes_range = np.arange(120, 241)  # [120, 121, 122, ..., 239, 240]

# Calculate the probability for each minute using the given PDF expression
probabilities = [0.31 * np.exp(-3 * (m/60)) + 0.68 * np.exp(-2 * (m/60)) for m in minutes_range]

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(minutes_range, probabilities)
plt.title("Probability Density Function (PDF)")
plt.xlabel("Minutes")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(minutes_range, bins=minutes_range, weights=probabilities, edgecolor='black')
plt.title("Histogram")
plt.xlabel("Minutes")
plt.ylabel("Probability")
plt.grid(True)
plt.show()