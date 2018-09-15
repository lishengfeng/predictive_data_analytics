import numpy as np
import pylab as pl

data = np.loadtxt("Chapter_7.data")

pl.plot(data[:, 0], data[:, 1], 'r', label='tas')
pl.plot(data[:, 0], data[:, 2], 'b', label='ttas')
pl.title("Mystery Explained")
pl.xlabel("threads")
pl.ylabel("time")
# pl.legend([plot_tas, plot_ttas], ["TAS", "TTAS"], "best")
pl.legend(loc='upper right')
pl.show()


