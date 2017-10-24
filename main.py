"""
file: main.py
author: Petri Lamminaho


"""

import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
# test data
Xs = np.array( [1, 2, 3, 4, 5, 6], dtype=np.float64)
Ys = np.array( [5, 4, 6, 5, 6, 7], dtype=np.float64)
#Xs = np.array( [1, 4, 7, 8, 9, 10], dtype=np.float64)
#Ys = np.array( [3, 2, 3, 5, 7, 8], dtype=np.float64)


#plt.scatter(Xs, Ys)
#plt.show()

def find_best_fit_slope_and_intercept(Xs, Ys):
    """
    :param Xs:
    :param Ys:
    :return: m, b
    m is best fit slope
    b is intercept
    """
    m = ( ( (mean(Xs) * mean(Ys)) - mean(Xs * Ys) )/
    ((mean(Xs)*mean(Xs)) - mean(Xs * Xs) ))
    b = mean(Ys) - m * mean(Xs)

    return m, b



m, b  = find_best_fit_slope_and_intercept(Xs,Ys)
reg_line =[(m * x) + b for x in Xs]
pre_x = 9 # test predict value x = 9
pre_y =(m*pre_x) + b # calculates y when x = 9
print(pre_y)# prints about 7.857

plt.scatter(Xs, Ys)
plt.scatter(pre_x, pre_y) # draws predict point to the graph
plt.plot(Xs, reg_line)
plt.show()

