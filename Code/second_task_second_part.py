import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def pdf(y):
    return 0.31 * np.exp(-3 * y) + 0.68 * np.exp(-2 * y)

# Calculate the probability of waiting between 2 and 4 hours
probability, _ = quad(pdf, 2, 4)

# Create an array of minutes from 0 to 240 (4 hours)
minutes = np.arange(0, 241)

# Calculate the PDF values for each minute
pdf_values = pdf(minutes / 60)

# Calculate the mean, variance, and quartiles
mean = quad(lambda y: y * pdf(y), 0, np.inf)[0]
variance = quad(lambda y: (y - mean) ** 2 * pdf(y), 0, np.inf)[0]
q1 = quad(pdf, 0, np.inf)[0] * 0.25
q2 = quad(pdf, 0, np.inf)[0] * 0.5
q3 = quad(pdf, 0, np.inf)[0] * 0.75

# Plot the PDF graph
plt.figure(figsize=(10, 6))
plt.plot(minutes, pdf_values)
plt.xlabel('Time (minutes)')
plt.ylabel('Probability Density')
plt.title('Probability Density Function')
plt.grid(True)
plt.show()

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(minutes, bins=60, weights=pdf_values * (minutes[1] - minutes[0]))
plt.xlabel('Time (minutes)')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)
plt.show()

# Display the mean, variance, and quartiles
print(f"Mean: {mean} hours")
print(f"Variance: {variance} hours^2")
print(f"Quartile 1: {q1} hours")
print(f"Median (Quartile 2): {q2} hours")
print(f"Quartile 3: {q3} hours")
