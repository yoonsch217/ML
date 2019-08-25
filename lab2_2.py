import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import rc, cm
from scipy import optimize

import matplotlib.pylab as plt

from sklearn.preprocessing import PolynomialFeatures


with open('ex2data1.txt') as f1, open('ex2data2.txt') as f2:
	dataset1 = np.loadtxt(f1, delimiter = ',',
		dtype = 'float', usecols = None)
	dataset2 = np.loadtxt(f2, delimiter = ',',
		dtype = 'float', usecols = None)

X = dataset2[:, :-1]
Y = dataset2[:, 2]
KO = np.where(Y == 0)[0]
OK = np.where(Y)[0]
plt.scatter(X[KO, 0], X[KO, 1])
plt.scatter(X[OK, 0], X[OK, 1], c="r")
#plt.show()

# Making feature map
poly = PolynomialFeatures(degree = 6)
X_poly = poly.fit_transform(X)
#print(X_poly[0])


def sigmoid (z):
	return 1.0 / (1.0 + np.exp(-z))

def costFunction (theta, x, y, l):
	m = len(x)
	h = sigmoid(np.inner(x, theta))
	reg = l * np.sum(theta * theta) / (2 * m)
	return (np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / -m) + reg
def costFunction2 (theta, x, y, l):
	m = len(x)
	h = sigmoid(np.inner(x, theta))
	reg = l * np.sum(np.square(theta)) / (2 * m)
	return (np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / -m) + reg

theta = [0.1]*28
print(theta)


#newX = np.concatenate((np.ones((100, 1)), X), axis=1)


theta_result = optimize.minimize(costFunction, theta, args=(X_poly, Y, 1))
theta_result2 = optimize.minimize(costFunction2, theta, args=(X_poly, Y, 1))
print(theta_result.x[2])
print(theta_result2.x[2])
theta = theta_result


x = [a for a in range(30, 100)]
y = [-(theta.x[1]/theta.x[2])*b - theta.x[0]/theta.x[2] for b in range(30, 100)]
#plt.plot(x,y)

#plt.show()

###################3
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))
def mapFeatureForPlotting(X1, X2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
    return out

for i in range(len(u)):
    for j in range(len(v)):
        z[i,j] = np.dot(mapFeatureForPlotting(u[i], v[j]), theta.x)

plt.contour(u,v,z,0)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
#plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()







