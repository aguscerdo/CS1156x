import numpy as np


def testFit(iterations, linspacing = 1000):
    X = np.random.uniform(-1,1,size=[2,iterations])
    Y = np.sin(np.pi * X)

    linSample = (np.linspace(-1, 1, linspacing))
    SqLinSpace = np.square(linSample)
    sinLin = np.sin(np.pi * linSample)


    # Constant y = b
    height = (Y[0] + Y[1])/2
    avgHeight = np.average(height)

    biasConstant = np.mean(np.square(avgHeight - sinLin))

    varianceConstant = np.mean(np.square(np.outer(height, linSample) - sinLin))

    results_constant = [biasConstant, varianceConstant, biasConstant + varianceConstant]
    #---------------------------#
    # Linear at origin -- y = mx
    # Slope = (x0*y0 + x1*y1)/(x0 + x1)
    slopeB0 = np.divide(np.multiply(X[0], Y[0]) + np.multiply(X[1], Y[1]), (np.square(X[0]) + np.square(X[1])))
    averageSlopeB0 = np.average(slopeB0)

    avgLinB0 = averageSlopeB0 * linSample

    biasLinB0 = np.mean(np.square(avgLinB0 - sinLin))

    varianceLinB0 = np.mean(np.square(np.outer(slopeB0, linSample) - avgLinB0))

    results_Linear_B0 = [biasLinB0, varianceLinB0, biasLinB0 + varianceLinB0]


    #---------------------------#
    # Linear -- y = mx + b
    # Line cuts through both points
    slopeLinear = np.divide(Y[1] - Y[0], X[1] - X[0])
    interceptLinear = Y[1] - np.multiply(slopeLinear, X[1])

    averageSlopeLinear = np.average(slopeLinear)
    averageInterceptLinear = np.average(interceptLinear)

    avgLinearLin = np.multiply(averageSlopeLinear, linSample) + averageInterceptLinear

    biasLinear = np.mean(np.square(avgLinearLin - sinLin))

    varianceLinear = np.mean(np.square(np.outer(slopeLinear - averageSlopeLinear, linSample) + np.expand_dims(interceptLinear - averageInterceptLinear, 1)))

    results_Linear = [biasLinear, varianceLinear, biasLinear + varianceLinear]


    #---------------------------#
    # Quadratic at origin -- y = mx^2
    # Coeff = (x0^2*y0 + x1^2*y1)/(x0^4 + x1^4)
    quadtraticB0 = np.divide(np.multiply(np.square(X[0]), Y[0]) + np.multiply(np.square(X[1]), Y[1]), np.power(X[0], 4) + np.power(X[1], 4))
    averageQuadraticB0 = np.average(quadtraticB0)

    avgQuadraticB0 = averageQuadraticB0 * SqLinSpace

    biasQuadraticB0 = np.mean(np.square(avgQuadraticB0 - sinLin))

    varianceQuadraticB0 = np.mean(np.square(np.outer(quadtraticB0, SqLinSpace) - avgQuadraticB0))

    results_Quadratic_B0 = [biasQuadraticB0, varianceQuadraticB0, biasQuadraticB0 + varianceQuadraticB0]


    #---------------------------#
    # Quadratic -- y = mx^2 + b
    # Coeff = (x0^2*y0 + x1^2*y1)/(x0^4 + x1^4)

    #Helpers
    x2 = np.square(X[0]) + np.square(X[1])
    ySums = Y[0] + Y[1]
    x4 = np.power(X[0], 4) + np.power(X[1], 4)
    x2y = np.square(X[0]) * Y[0] + np.square(X[1]) * Y[1]
    bottom = 2 * x4 - np.square(x2)


    quadraticCoeff = np.divide(2 * x2y - np.multiply(x2, ySums), bottom)
    quadraticIntercept = np.divide(np.multiply(ySums, x4) - np.multiply(x2, x2y), bottom)

    averageQuadraticCoeff = np.average(quadraticCoeff)
    averageQuadraticIntercept = np.average(quadraticIntercept)


    avgQuadratic = averageQuadraticCoeff * SqLinSpace + averageQuadraticIntercept


    biasQuadratic = np.mean(np.square(avgQuadratic - sinLin))

    varianceQuadratic = np.mean(np.square(np.outer(quadraticCoeff, SqLinSpace) + np.expand_dims(quadraticIntercept, 1) - avgQuadratic))

    results_Quadratic = [biasQuadratic, varianceQuadratic, biasQuadratic + varianceQuadratic]
    
    
    return [results_constant, results_Linear_B0, results_Linear, results_Quadratic_B0, results_Quadratic]


# Q7
results = testFit(10000)

print("y = b: Bias %f, Var %f, Error %f" % tuple(results[0]))
print("y = ax: Bias %f, Var %f, Error %f" % tuple(results[1]))
print("y = ax + b: Bias %f, Var %f, Error %f" % tuple(results[2]))
print("y = ax^2: Bias %f, Var %f, Error %f" % tuple(results[3]))
print("y = ax^2 + b: Bias %f, Var %f, Error %f" % tuple(results[4]))



