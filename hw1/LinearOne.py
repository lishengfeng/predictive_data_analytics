from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

# Evaluate the linear regression
def compute_cost(X, y, theta):
    # compute cost for linear regreesion

    # Number of training samples
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn thetas by num_iters steps
    with learning rate alpha
    """
    m = y.size
    """
    J_history is an array that allows you to remember the values of the cost function for every update.
    Look at the value of J(θ) and check that it is decreasing with each step. 
    """
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        # numpy.dot For 2-D arrays it is equivalent to matrix multiplication
        # flatten Return a copy of the array collapsed into one dimension.
        predictions = X.dot(theta).flatten()

        error_x1 = (predictions - y) * X[:, 0]
        error_x2 = (predictions - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * error_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * error_x2.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

cols = ['size', 'bedrooms']
# Load the dataset
x = loadtxt('hw1_linear_x.dat')
y = loadtxt('hw1_linear_y.dat')

# Save a copy of the unscaled features for later
x_unscaled = x

x = (x - x.mean()) / x.std()

# plot the data
scatter(x, y, marker='o', c='b')
title('house price distribution')
xlabel('size')
ylabel('price')
# show()

# number of training samples
m = y.size

# Add a column of ones to x (interception data)
it = ones(shape=(m, 2))
it[:, 1] = x

# Initialize theta parameters
theta = zeros(shape=(2, 1))

# Some gradient descent settings
iterations = 100
alpha = 0.1

# computer and display initial cost
print(compute_cost(it, y, theta))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print(theta)

# plot the results
result = it.dot(theta).flatten()
plot(x, result)
show()

