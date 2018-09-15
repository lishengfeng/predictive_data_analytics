import csv
import math
import sys
from random import uniform

with open('hw2_kmeans_data1.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open('hw2_kmeans_data2.csv', 'r') as f:
    reader = csv.reader(f)
    data2 = list(reader)


def find_col_min_max(items):
    n = len(items[0])
    min_item = [sys.maxint for i in range(n)]
    max_item = [-sys.maxint - 1 for i in range(n)]

    for item in items:
        for f in range(len(item)):
            if item[f] < min_item[f]:
                min_item[f] = item[f]

            if item[f] > max_item[f]:
                max_item[f] = item[f]

    return min_item, max_item


def initialize_means(items, k, c_min, c_max):
    # Initialize means to random numbers between
    # the min and max of each column/feature
    f_num = len(items[0])  # number of features
    means = [[0 for i in range(f_num)] for j in range(k)]

    for mean in means:
        for i in range(len(mean)):
            # Set value to a random float
            # (adding +-1 to avoid a wide placement of a mean)
            mean[i] = uniform(c_min[i] + 1, c_max[i] - 1)

    return means


def euclidean_distance(x, y):
    S = 0  # The sum of the squared differences of the elements
    for i in range(len(x)):
        S += math.pow(x[i] - y[i], 2)

    return math.sqrt(S)  # The square root of the sum


def update_mean(n, mean, item):
    for i in range(len(mean)):
        m = mean[i]
        m = (m * (n - 1) + item[i]) / float(n)
        mean[i] = round(m, 3)

    return mean


def classify(means, item):
    # Classify item to the mean with minimum distance
    minimum = sys.maxint
    index = -1

    for i in range(len(means)):

        # Find distance from item to mean
        dis = euclidean_distance(item, means[i])

        if dis < minimum:
            minimum = dis
            index = i

    return index


def calculate_means(k, items, max_iterations=300):
    # Find the minima and maxima for columns
    c_min, c_max = find_col_min_max(items)

    # Initialize means at random points
    means = initialize_means(items, k, c_min, c_max)

    # Initialize clusters, the array to hold
    # the number of items in a class
    cluster_sizes = [0 for i in range(len(means))]

    # An array to hold the cluster an item is in
    belongs_to = [0 for i in range(len(items))]

    # Calculate means
    for e in range(max_iterations):

        # If no change of cluster occurs, halt
        no_change = True
        for i in range(len(items)):

            item = items[i]

            # Classify item into a cluster and update the
            # corresponding means.
            index = classify(means, item)

            cluster_sizes[index] += 1
            c_size = cluster_sizes[index]
            means[index] = update_mean(c_size, means[index], item)

            # Item changed cluster
            if index != belongs_to[i]:
                no_change = False

            belongs_to[i] = index

        # Nothing changed, return
        if no_change:
            break

    return means


def find_clusters(means, items):
    clusters = [[] for i in range(len(means))]  # Init clusters

    for item in items:
        # Classify item into a cluster
        index = classify(means, item)

        # Add item to cluster
        clusters[index].append(item)

    return clusters
