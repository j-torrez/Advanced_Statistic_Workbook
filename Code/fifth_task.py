import numpy as np
import matplotlib.pyplot as plt

# Given sample points
sample_points = [(10, 62200794443.21), (-10, 63084618692.71), (20, 58048228436923.3), (-12, 364419781502.02),
                 (-11, 158286531919.14), (15, 3429259119784.41), (-6, 367243631.12), (13, 855519370157.35),
                 (1, 20.14), (19, 36590429741774.51), (4, 6570128.74), (-9, 20960251189.5), (-4, 7187338.56),
                 (8, 6424279460.05), (16, 6542019625414.05), (7, 1664406863.9), (-15, 3596862038231.14),
                 (17, 12505645177768.95), (-18, 22006743059430.94), (-7, 1791542092.71), (-16, 6615740114842.96),
                 (0, 6.83)]

# Extract x and y values from the sample points
x_values = [point[0] for point in sample_points]
y_values = [point[1] for point in sample_points]

# Set up the design matrix X
X = np.column_stack([np.power(x_values, i) for i in range(11)])

# Set up the vector of observed values Y
Y = np.array(y_values)

# Step 4: Calculate the OLS estimate for the parameters
alpha_ols = np.linalg.inv(X.T @ X) @ X.T @ Y

# Step 5: Calculate the OLS ridge-regularized estimates for the parameters
lambda_ridge = 1  # Ridge regularization parameter
alpha_ridge = np.linalg.inv(X.T @ X + lambda_ridge * np.identity(11)) @ X.T @ Y

# Generate values for x to plot the approximating function
x_plot = np.linspace(-20, 20, 100)
X_plot = np.column_stack([np.power(x_plot, i) for i in range(11)])

# Calculate the values of the polynomial model function for the OLS estimate
y_ols = X_plot @ alpha_ols

# Calculate the values of the polynomial model function for the OLS ridge-regularized estimate
y_ridge = X_plot @ alpha_ridge

# Plot the graph of the approximating functions and the data points
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='red', label='Data Points')
plt.plot(x_plot, y_ols, label='OLS Estimate')
plt.plot(x_plot, y_ridge, label='OLS Ridge-Regularized Estimate (λ = 1)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approximating Functions')
plt.legend()
plt.grid(True)
plt.show()