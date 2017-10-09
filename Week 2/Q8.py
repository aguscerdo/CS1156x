from numpy import linalg, sign, dot, transpose, ones, multiply, square, random as nprand, array, divide
import random


def labelTransform(pointMatrix, noise=0):    #Return sign of (x^2 + y^2 - 0.6)
    N = len(pointMatrix)
    label = []
    for i in range(N):
        label.append(sign(pointMatrix[i][1]**2 + pointMatrix[i][2]**2 - 0.6))
        label[label==0] = 1

    noisy = random.sample(range(N), int(N*noise))    #Choose 'noise' fraction of changed labels
    for i in noisy:
        label[i] *= -1

    return label


def nonLinearTransform(Samples, noise=0, linear=False):
    # Generate points and label
    x = nprand.uniform(-1, 1, Samples)
    y = nprand.uniform(-1, 1, Samples)
    coordinateMatrix = transpose([ones(Samples), x, y])
    label = labelTransform(coordinateMatrix, noise=0.1)

    if linear:  # Use normal linear regression
        solution = linalg.lstsq(coordinateMatrix, label)[0]
        testSolution = sign(dot(coordinateMatrix, solution))
        miss = 0
        for i in range(Samples):
            if label[i] != testSolution[i]:
                miss += 1
        miss /= Samples
        return [miss, solution]

    else:   # Use non-linear transform
        nonLinearMatrix = transpose([ones(Samples), x, y, multiply(x, y), square(x), square(y)])
        solution = linalg.lstsq(nonLinearMatrix, label)[0]
        testSolution = sign(dot(nonLinearMatrix, solution))
        testSolution[testSolution==0] = 1

        miss = 0
        for i in range(Samples):
            if label[i] != testSolution[i]:
                miss += 1
        miss /= Samples
        return [miss, solution]


def testNonLinearTransform(Iterations, Samples, noise=0, linear=False):
    avgMiss = 0
    if linear:
        avgWeight = [0, 0, 0]
    else:
        avgWeight = [0, 0, 0, 0, 0, 0]
    for i in range(Iterations):
        result = nonLinearTransform(Samples, noise, linear)
        avgMiss += result[0]
        avgWeight += result[1]
    avgMiss /= Iterations
    avgWeight /= Iterations

    print("Average Ein: %f" % avgMiss)

    if linear:
        print("Weights: %f, %f, %f" %
              (avgWeight[0], avgWeight[1], avgWeight[2]))
    else:
        print("Weights: %f, %f, %f, %f, %f, %f" %
              (avgWeight[0], avgWeight[1],avgWeight[2],avgWeight[3],avgWeight[4],avgWeight[5]))

    return [avgMiss, avgWeight]


def outOfSample(NewSamples, TrainingSamples, noise=0, linear=False):
    # In Sample Training
    # Generate training points and label
    x = nprand.uniform(-1, 1, TrainingSamples)
    y = nprand.uniform(-1, 1, TrainingSamples)
    coordinateMatrix = transpose([ones(TrainingSamples), x, y])
    label = labelTransform(coordinateMatrix, noise=0.1)

    if linear:  # Use normal linear regression
        solution = linalg.lstsq(coordinateMatrix, label)[0]
        testSolution = sign(dot(coordinateMatrix, solution))
        testSolution[testSolution == 0] = 1

        Ein = 0
        for i in range(TrainingSamples):
            if label[i] != testSolution[i]:
                Ein += 1
        Ein /= TrainingSamples

    else:  # Use non-linear transform
        nonLinearMatrix = transpose([ones(TrainingSamples), x, y, multiply(x, y), square(x), square(y)])
        solution = linalg.lstsq(nonLinearMatrix, label)[0]
        testSolution = sign(dot(nonLinearMatrix, solution))
        testSolution[testSolution == 0] = 1

        Ein = 0
        for i in range(TrainingSamples):
            if label[i] != testSolution[i]:
                Ein += 1
        Ein /= TrainingSamples

    # Out of Sample Testing
    # Generate new Points  and label
    xNew = nprand.uniform(-1, 1, TrainingSamples)
    yNew = nprand.uniform(-1, 1, TrainingSamples)
    newCoordinateMatrix = transpose([ones(TrainingSamples), xNew, yNew])
    newLabel = labelTransform(newCoordinateMatrix, noise=0.1)

    if linear:
        newTestSolution = sign(dot(coordinateMatrix, solution))
        newTestSolution[newTestSolution == 0] = 1

        Eout = 0
        for i in range(NewSamples):
            if newLabel[i] != newTestSolution[i]:
                Eout += 1
        Eout /= NewSamples
    else:
        newNonLinearMatrix = transpose([ones(NewSamples), xNew, yNew, multiply(xNew, yNew), square(xNew), square(yNew)])
        newSolution = linalg.lstsq(newNonLinearMatrix, newLabel)[0]
        newTestSolution = sign(dot(newNonLinearMatrix, newSolution))
        newTestSolution[newTestSolution == 0] = 1

        Eout = 0
        for i in range(NewSamples):
            if newLabel[i] != newTestSolution[i]:
                Eout += 1
        Eout /= NewSamples

    return [Ein, Eout]


def testOutOfSample(Iterations, NewSamples, TrainingSamples, noise=0, linear=False):
    avgEin = 0
    avgEout = 0
    for i in range(Iterations):
        result = outOfSample(NewSamples, TrainingSamples, noise, linear)
        avgEin += result[0]
        avgEout += result[1]
    avgEout /= Iterations
    avgEin /= Iterations

    print("Average Ein: %f\n"
          "Average Eout: %f" % (avgEin, avgEout))

    return (avgEin, avgEout)


# Q8
# testNonLinearTransform(Iterations, Samples, noise=0, linear=False):
print("\nTesting linear regression")
testNonLinearTransform(1000, 1000, 0.1, True)


# Q9
# testNonLinearTransform(Iterations, Samples, noise=0, linear=False):
print("\nTesting linear transform")
testNonLinearTransform(1000, 1000, 0.1, False)


# Q10
# testOutOfSample(Iterations, NewSamples, TrainingSamples, noise=0, linear=False)
print ("\nTesting out of sample performance")
testOutOfSample(1000, 1000, 1000, 0.1, False)

