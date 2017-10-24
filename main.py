import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
# test data
Xs = np.array( [1, 2, 3, 4, 5, 6], dtype=np.float64)
Ys = np.array( [5, 4, 6, 5, 6, 7], dtype=np.float64)
#plt.scatter(Xs, Ys)
#plt.show()

def find_best_fit_slope(Xs, Ys):

    m = ( ( (mean(Xs) * mean(Ys)) - mean(Xs * Ys) )/
    ((mean(Xs)*mean(Xs)) - mean(Xs * Xs) ))

    return m



m = find_best_fit_slope(Xs,Ys)
print(m)

