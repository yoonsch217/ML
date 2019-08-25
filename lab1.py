import numpy as np
import matplotlib as mpl
import pandas
import matplotlib.pyplot as plt

from matplotlib import rc


def costFunction(points, theta):
     x = points[0]
     y = points[1]
     
     h = theta[0] + theta[1] * x - y
     
     return np.sum(h * h) / (2.0 * len(x))

def gardientDescent(points, theta, alpha, epsilon):
    x = points[0]
    y = points[1]
    m = len(x)
    diff = epsilon + 1.0
    oldtheta = theta[:]
    oldcost = 0
    
    progress = []
    
    while diff > epsilon:
        tmpsum = np.sum(oldtheta[0] + oldtheta[1] * x - y)
        theta[0] = oldtheta[0] - alpha / m * tmpsum
        
        tmpsum = np.sum((oldtheta[0] + oldtheta[1] * x - y) * x)
        theta[1] = oldtheta[1] - alpha / m * tmpsum

        cost = costFunction(points,theta)
        diff = abs(oldcost - cost)

        progress.append(cost)
        oldtheta = theta[:]
        oldcost = cost
        
    plt.figure()
    plt.plot(progress, "rx")   
    plt.title("Evolution of theta during training")

    return theta


citytruck = np.loadtxt("/home/julien/Bureau/Nextcloud/cours/SU/ML/lab1/data1", delimiter=",", unpack=True)

theta = gardientDescent(citytruck, [0,0], 0.01, 0.000005)


plt.figure()
plt.plot(*citytruck, "rx")
plt.title("Food truck profit according to city size)")
plt.xlabel("Population is 10,000s")
plt.ylabel("Profit of City in 10,000s")

print(theta)

x = np.array(range(0,25))
plt.plot(x, x * theta[1] + theta[0])
