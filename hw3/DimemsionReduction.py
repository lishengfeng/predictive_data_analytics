import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.decomposition import PCA

# Load the dataset
X = loadtxt('hw3_pen_data.txt', delimiter=',', usecols=range(0, 16))
Y = loadtxt('hw3_pen_data.txt', delimiter=',', usecols=16)

pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
#
plt.figure()

# plt.scatter(X_r[:, 0], X_r[:, 1], 10, Y, label=Y)
for i in range(0, 10):
    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], label=i, alpha=.8, lw=1)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of MNIST Digit dataset')

plt.show()
