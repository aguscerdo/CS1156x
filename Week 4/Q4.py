import numpy as np


def predictFit(iterations):
    X = np.random.uniform(-1,1,size=[2,iterations])
    Y = np.sin(np.pi * X)

    # Least Square Error
    slope = np.divide(np.multiply(X[0], Y[0]) + np.multiply(X[1], Y[1]), (np.square(X[0]) + np.square(X[1])))
    averageSlope = np.average(slope)


    linSample = (np.linspace(-1, 1, 1000))
    avgLin = averageSlope * linSample
    sinLin = np.sin(np.pi * linSample)
    gLin = np.outer(slope, linSample)


    # Bias Average Error -- Bias = Ex((g_avg(x) - f(x))^2)
    bias = np.mean(np.square(avgLin - sinLin))

    # Variance -- Var = Ex((g(x) - g_avg(x))^2)
    variance = np.mean(np.square(gLin - avgLin))


    return [averageSlope, bias, variance]


# predictFit(iterations):
# f(x) = sin(pi*x)
results = predictFit(5000)


# Q4 Average prediction in form y = mx for 2 samples
print("Average g(x): {}".format(results[0]))


# Q5 Bias  --  Bias = Ex((g_avg(x) - f(x))^2)
print("Bias: {}".format(results[1]))


# Q6 Variance  --  Variance = Ex((g(D)(x) - g_avg(x))^2)
print("Variance: {}".format(results[2]))