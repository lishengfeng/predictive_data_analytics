from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def feature_normalize(X):
    """
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """

    mean_r = []
    std_r = []

    X_norm = X

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r

# Evaluate the linear regression
def compute_cost(X, y, theta):
    # compute cost for linear regreesion

    # Number of training samples
    m = y.size

    predictions = X.dot(theta)

    sqErrors = (predictions - y)

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

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
        predictions = X.dot(theta)

        theta_size = theta.size

        for it in range(theta_size):
            temp = X[:, it]
            temp.shape = (m, 1)

            error_x1 = (predictions - y) * temp
            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * error_x1.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

cols = ['size', 'bedrooms']
# Load the dataset
X = loadtxt('hw1_linearM_x.dat')
y = loadtxt('hw1_linearM_y.dat')

# Save a copy of the unscaled features for later
X_unscaled = X

# plot the data
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m in [('r', 'o')]:
    xs = X[:, 0]
    ys = X[:, 1]
    zs = y
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Size of the House')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price of the House')
plt.show()
"""

# number of training samples
m = y.size

y.shape = (m, 1)

# Scale features and set them to zero mean
x, mean_r, std_r = feature_normalize(X)

# Add a column of ones to x (interception data)
it = ones(shape=(m, 3))
it[:, 1:3] = x


# Initialize theta parameters
theta = zeros(shape=(3, 1))

# Some gradient descent settings
iterations = 100
alpha = 0.05

# computer and display initial cost
# print(compute_cost(it, y, theta))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

# print(theta)
# print(J_history)

# plot the results
plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function J')
show()

# Predict price of a 1750 sq-ft 3 br house
price = array([1.0, ((1750 - mean_r[0]) / std_r[0]), ((3 - mean_r[1]) / std_r[1])]).dot(theta)
print("Predicted price of a 1750 sq-ft, 3 bedroom house: %f" % price)