import numpy
import matplotlib.pyplot as plt
import random

# Helper
def PlotLine(line, xMin=-1, xMax=+1, color='k', alpha=1):
    plt.plot([xMin, xMax],
             [-(line[0] + xMin*line[1])/line[2], -(line[0] + xMax*line[1])/line[2]],
             c=color, alpha=alpha)


# Q5
def leastSquare(Samples, noise=0, display=False, retPoints=False):
    #Generate Line
    p1 = numpy.random.uniform(-1, 1, 2)
    p2 = numpy.random.uniform(-1, 1, 2)
    line = [p2[0]*p1[1] - p2[1]*p1[0], p2[1]-p1[1], p1[0]-p2[0]]

    #Generate Samples
    x = numpy.random.uniform(-1, 1, Samples)
    y = numpy.random.uniform(-1, 1, Samples)
    coordinateMatrix = numpy.transpose([numpy.ones(Samples), x, y])

    label = numpy.sign(numpy.dot(coordinateMatrix, line))
    label[label == 0] = 1  # sign(0) = 1

    # Add Noise
    noisy = random.sample(range(len(x)), int(len(x)*noise))    #Choose 'noise' fraction of changed labels
    for i in noisy:
        label[i] *= -1

    solution = numpy.linalg.lstsq(coordinateMatrix, label)[0]
    testSolution = numpy.sign(numpy.dot(coordinateMatrix, solution))
    miss = 0
    for i in range(Samples):
        if label[i] != testSolution[i]:
            miss += 1
    miss /= Samples
    # print("Ein: %f" % miss)

    #Plot
    if display:
        PlotLine(solution)
        PlotLine(line, color='darkgreen',alpha=0.8)

        cmap = plt.get_cmap('bwr')
        colors = cmap(label)
        plt.scatter(x, y, c=colors)

        plt.axis([-1, 1, -1, 1])
        plt.axvline(0, c='k', alpha=0.2)
        plt.axhline(0, c='k', alpha=0.2)
        plt.show()

    if retPoints:   #Used for Q7 PLA
        return [line, solution, coordinateMatrix, label]
    else:
        return [line, solution, miss]


def testLeastSquares(Iterations, Samples, noise=0):
    avgEin = 0
    for i in range(Iterations):
        avgEin += leastSquare(Samples, noise=noise)[2]
    avgEin /= Iterations
    print('Average Ein: %f\n' % avgEin)
    return avgEin


# Q6
def trainedLS(NewSamples, trainSamples, noise=0):
    line, solution, missEin = leastSquare(trainSamples, noise)

    # Generate new samples
    x = numpy.random.uniform(-1, 1, NewSamples)
    y = numpy.random.uniform(-1, 1, NewSamples)
    coordinateMatrix = numpy.transpose([numpy.ones(NewSamples), x, y])
    label = numpy.sign(numpy.dot(coordinateMatrix, line))
    label[label == 0] = 1  # sign(0) = 1

    testSolution =  numpy.sign(numpy.dot(coordinateMatrix, solution))
    miss = 0
    for i in range(NewSamples):
        if label[i] != testSolution[i]:
            miss += 1
    miss /= NewSamples
    return miss


def testTrainedLS(Iterations, NewSamples, trainSamples, noise=0):
    avgEout = 0
    for k in range(Iterations):
        avgEout += trainedLS(NewSamples, trainSamples, noise)
    avgEout /= Iterations
    print('Average Eout: %f\n' % avgEout)
    return avgEout


# Q7
def perceptron(Samples, noise=0):
    line, solution, pointsMatrix, label = leastSquare(Samples, noise=noise, retPoints=True)
    iterations = 0
    while 1:
        iterations += 1

        testL = numpy.sign(numpy.dot(pointsMatrix, solution))
        testL[testL==0] = 1

        badLabel = []
        for i in range(len(label)):
            if not label[i] == testL[i]:  #Test all bad points
                badLabel.append(i)

        if not len(badLabel):   #No bad points, break
            break
        k = numpy.random.choice(badLabel)   #Choose a random bad point

        #Adjust weights
        solution += label[k]*pointsMatrix[k]

    return iterations


def testPerceptron(Iterations, trainSamples, noise=0):
    avgIterations = 0
    for i in range(Iterations):
        avgIterations += perceptron(trainSamples, noise=noise)
    avgIterations /= Iterations
    print("Average perceptron iterations: %f" % avgIterations)
    return avgIterations


#--------------------------------------------------------------#

# Q5
# testLeastSquares(Iterations,Samples, noise=0)
print("Testing Ein")

testLeastSquares(1000, 100)


# Q6
# testTrainedLS(Iterations, NewSamples, trainSamples, noise=0):
print("Testing Eout")

testTrainedLS(1000, 1000, 100)

# Q7
# testPerceptron(Iterations, trainSamples, noise=0):
print('Testing perceptron')

testPerceptron(1000, 10)