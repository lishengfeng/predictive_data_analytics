from sklearn.cluster import KMeans
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

with open('hw2_kmeans_data1.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open('hw2_kmeans_data2.csv', 'r') as f:
    reader = csv.reader(f)
    data2 = list(reader)

# max n_cluster = 3
kmeans_data1 = KMeans(n_clusters=3, max_iter=300)
kmeans_data1.fit(data1)

# max n_cluster = 3
kmeans_data2 = KMeans(n_clusters=3, max_iter=300)
kmeans_data2.fit(data2)

centroid_data1 = kmeans_data1.cluster_centers_
labels_data1 = kmeans_data1.labels_

centroid_data2 = kmeans_data2.cluster_centers_
labels_data2 = kmeans_data2.labels_

colors = ["g.", "r.", "c."]

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for i in range(len(data1)):
    ax1.plot(data1[i][0], data1[i][1], colors[labels_data1[i]], markersize=10)

for i in range(len(data2)):
    ax2.plot(data2[i][0], data2[i][1], colors[labels_data2[i]], markersize=10)

ax1.scatter(centroid_data1[:, 0], centroid_data1[:, 1], marker="x", s=150, linewidths=5, zorder=10)
ax2.scatter(centroid_data2[:, 0], centroid_data2[:, 1], marker="x", s=150, linewidths=5, zorder=10)

plt.show()
