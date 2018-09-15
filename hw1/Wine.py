import numpy as np
import matplotlib.pyplot as plt
from math import *
import pandas as pd

cols = ['Class', 'Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols',
        'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity',
        'Hue', 'OD280/OD315', 'Proline']

data = pd.read_csv('wine.data', names=cols)
# y = data['Class']
# x = data.ix[:, 'Alcohol':]

# histogram plot
# data.ix[:, 'Alcohol':].hist(bins=10)
# density plot
# data.ix[:, 'Alcohol':].plot(kind='density', subplots=True, layout=(4, 4), sharex=False)
# boxplot
fig, axes = plt.subplots(ncols=3, sharey=False)
fig.subplots_adjust(wspace=5)

for label, ax in zip(cols[1:], axes):
    ax.boxplot([data[1][1] ])

plt.show()
# data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False)
# df = pd.DataFrame(data=data)
# df['class'] = pd.Series(['Class1', 'Class2', 'Class3'])
# box = df.boxplot(column=cols[13:14], by='Class')
# axes = df.boxplot(column=cols[1:], by='Class', layout=(4, 4), figsize=(100, 50), return_type='axes')
# for ax in axes:
#     ax.set_ylim(10, 30)
# plt.show()

# # correlation plot
# correlations = data.ix[:, 'Alcohol':].corr()
# # plot correlation matrix
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0, 13, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(cols[1:])
# ax.set_yticklabels(cols[1:])
#
plt.show()
