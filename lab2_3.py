import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import rc, cm
from scipy import optimize

import matplotlib.pylab as plt
import itertools
from sklearn.preprocessing import PolynomialFeatures


with open('tmp.txt') as f1:
	dataset1 = np.loadtxt(f1, delimiter = ',',
		dtype = 'float', usecols = None)

#X = dataset2[:, :-1]
#Y = dataset2[:, 2]
#X = dataset1[:, 0:3]
#Y = dataset1[:, 3:6]

X = dataset1[:12, 0:1]
Y = dataset1[:12, 3]

Xval = dataset1[:,1:2]
Yval = dataset1[:,4:5]

Xtest = dataset1[:,2:3]
Ytest = dataset1[:,5:6]

#plt.scatter(X[:,0],Y[:,0])
'''
plt.figure()
plt.scatter(X,Y)
plt.show()
'''
# Making feature map
#poly = PolynomialFeatures(degree = 6)
#X_poly = poly.fit_transform(X)
#print(X_poly[0])


def sigmoid (z):
	return 1.0 / (1.0 + np.exp(-z))


def costFunction (theta, x, y, lamda):
	m = len(x)
	h = np.inner(theta,x)
	regularization = lamda * np.sum(np.square(theta)) / (2 * m)
	return np.sum(np.square(h - y)) / (2 * m) + regularization


'''
theta = [0.1]*28
print(theta)
'''
theta = (0, 0)


newX = np.concatenate((np.ones((len(X), 1)), X), axis=1)

plt.figure()

err_sum = [0] * 12
for i in range (1, 12):
	
	theta = optimize.minimize(costFunction, theta, args=(newX[:i, :],Y[:i], 2)).x

	err2 = costFunction(theta, newX[:i,], Y[:i], 0)
	err_sum[i] = err2
	print err2

x_arr = [1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(x_arr, err_sum, '-o')
plt.show()
#theta = optimize.minimize(costFunction, theta, args=(newX,Y, 2))
#theta = optimize.minimize(costFunction, theta, args=(X_poly, Y, 1))

#print(theta)


x = np.array(range(-50,50))
plt.plot(x, x*theta[1] + theta[0])

#plt.show()
'''
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

'''
