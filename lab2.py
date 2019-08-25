import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import rc, cm
from scipy import optimize

import matplotlib.pylab as plt

with open('ex2data1.txt') as f1, open('ex2data2.txt') as f2:
	dataset1 = np.loadtxt(f1, delimiter = ',',
		dtype = 'float', usecols = None)
	dataset2 = np.loadtxt(f2, delimiter = ',',
		dtype = 'float', usecols = None)

X = dataset1[:, :-1]
Y = dataset1[:, 2]
KO = np.where(Y == 0)[0]
OK = np.where(Y)[0]
plt.scatter(X[KO, 0], X[KO, 1])
plt.scatter(X[OK, 0], X[OK, 1], c="r")
#plt.show()


def sigmoid (z):
	
	return 1.0 / (1.0 + np.exp(-z))

def costFunction (theta, x, y):
	m = len(x)
	h = sigmoid(np.inner(x, theta))
	return np.sum(y*np.log(h) + (1-y) * np.log(1-h)  ) / -m


newX = np.concatenate((np.ones((100, 1)), X), axis=1)
theta = (0.1, 0.1, 0.1)
print(theta)

theta = optimize.minimize(costFunction, theta, args=(newX, Y))

print(theta)

x = [a for a in range(30, 100)]
y = [-(theta.x[1]/theta.x[2])*b - theta.x[0]/theta.x[2] for b in range(30, 100)]
plt.plot(x,y)

plt.show()

inputx = [1.0, 45.0, 85.0]
inputy = [1.0]


#print(costFunction(theta.x, inputx, inputy))
print(sigmoid(np.inner(inputx, theta.x)))
