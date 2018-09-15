import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import *

# wineData = np.array([map(float, line.split(',')) for line in wineFile])
# wineData = np.genfromtxt('wine.data',  delimiter=',', dtype=None)
np.set_printoptions(suppress=True, linewidth=75)
all_data = np.loadtxt('wine.data', dtype=np.float64, delimiter=',', ndmin=2)

# A = wineData[:, 0]
# B = wineData[:, 1:14]
# print(A)
# print(B)
# print(wineData)

# load class labels from column 1
y_wine = all_data[:, 0]

# conversion of the class labels to integer-type array
y_wine = y_wine.astype(np.int64, copy=False)

# load the 14 features
X_wine = all_data[:, 1:]

# printing some general information about the data
print('\ntotal number of samples (rows):', X_wine.shape[0])
print('total number of feature (columns):', X_wine.shape[1])

# printing the 1st wine sameple
float_formatter = lambda x: '{:.2f}'.format(x)
np.set_printoptions(formatter={'float_kind': float_formatter})
print('\n1st sample (i.e., 1st row):\nClass label: {:d}\n{:}\n'.format(int(y_wine[0]), X_wine[0]))

# printing the rel.frequency of the class labels
print('Class label frequencies')
print('Class 1 samples: {:.2%}'.format(list(y_wine).count(1) / y_wine.shape[0]))
print('Class 2 samples: {:.2%}'.format(list(y_wine).count(2) / y_wine.shape[0]))
print('Class 3 samples: {:.2%}'.format(list(y_wine).count(3) / y_wine.shape[0]))

plt.figure(figsize=(10, 8))

# bin width of the histogram in steps of 0.15
bins = np.arange(floor(min(X_wine[:, 0])), ceil(max(X_wine[:, 0])), 0.15)

# get the max count for a particular bin for all classes combined
max_bin = max(np.histogram(X_wine[:, 0], bins=bins)[0])

# the order of the color for each histogram
colors = ('blue', 'red', 'green')

for label, color in zip(
        range(1, 4), colors):
    mean = np.mean(X_wine[:, 0][y_wine == label])  # class sample mean
    stdev = np.std(X_wine[:, 0][y_wine == label])  # class standard deviation
    plt.hist(X_wine[:, 0][y_wine == label],
             bins=bins,
             alpha=0.3,
             label='class {} ($\mu$={:.2f}, $\sigma$={:.2f})'.format(label, mean, stdev),
             color=color)


plt.ylim([0, max_bin*1.3])
plt.title('Wine data set - Distribution of alocohol contents')
plt.xlabel('alcohol by volume', fontsize=14)
plt.ylabel('count', fontsize=14)
plt.legend(loc='upper right')

plt.show()