import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the negative binomial distribution
k = 40  # Number of successes
p = 0.26  # Probability of success

# Calculate the probabilities for different numbers of meteorites falling using the negative binomial distribution
n = np.arange(1, 100)  # Numbers of meteorites falling
probabilities = np.array([np.sum(np.power(1 - p, np.arange(0, i)) * p ** k) for i in n])

# Find the index at which the probability drops below 0.5%
threshold = 0.005
idx = np.where(probabilities < threshold)[0][-1]

# Calculate the expectation and median
expectation = k / p
median = np.floor((k - 1) / p)

# Plot the probability distribution
plt.figure(figsize=(10, 6))
plt.plot(n, probabilities, 'bo-', markersize=5, label='Probability')
plt.axvline(x=idx + 1, color='r', linestyle='--', label='Probability < 0.5%')
plt.axvline(x=expectation, color='g', linestyle='--', label='Expectation')
plt.axvline(x=median, color='m', linestyle='--', label='Median')
plt.xlabel('Number of Meteorites')
plt.ylabel('Probability')
plt.title('Probability Distribution of Meteorites Falling on an Ocean')
plt.legend()
plt.grid(True)
plt.show()