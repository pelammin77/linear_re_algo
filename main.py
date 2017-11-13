
"""
file: main.py
author: Petri Lamminaho
My own linear reg algorith
Makes random dataset
Find best b and m values
Minimize loss/error and find best fit line for data and  calculate model confidence
Draw the graph
"""

import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')
# test data
#Xs = np.array( [1, 2, 3, 4, 5, 6], dtype=np.float64)
#Ys = np.array( [5, 4, 6, 5, 6, 7], dtype=np.float64)
#Xs = np.array( [1, 4, 7, 8, 9, 10], dtype=np.float64)
#Ys = np.array( [3, 2, 3, 5, 7, 8], dtype=np.float64)

def create_dataset(how_many, variance, step=2, correlation=False):
    """
    create random dataset and return it
params:
    :param how_many:
    :param variance:
    :param step:
    :param correlation:
    :return: np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
    """
    val = 1
    ys = []
    for i in range(how_many):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation =='pos':
            val +=step
        elif correlation and correlation =='neg':
            val -=step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def find_best_fit_slope_and_intercept(Xs, Ys):
    """
    Find best b and m value for slope for data
:params:
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

def squared_error(ys_org, ys_line):
    """
Minimize squared error/loss and returns it
    :param ys_org:
    :param ys_line:
    :return: squared_error
    """

    return sum((ys_line - ys_org) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
    """
return confidende value for the model calls 'squared_error' function
    :param ys_orig:
    :param ys_line:
    :return: confidence percent
    """
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

# Main function
if __name__ == '__main__':
    Xs, Ys = create_dataset(40, 10, 2, correlation='pos')
    m, b  = find_best_fit_slope_and_intercept(Xs,Ys)
    reg_line =[(m * x) + b for x in Xs]
    #regression_line = [(m*x)+b for x in Xs]
    pre_x = 9 # test predict value x = 9
    pre_y =(m*pre_x) + b # calculates y when x = 9
    r_sq = coefficient_of_determination(Ys, reg_line)
    print('reg', r_sq)
    plt.scatter(Xs, Ys)
    plt.scatter(pre_x, pre_y) # draws predict point to the graph
    plt.plot(Xs, reg_line)
    plt.show()