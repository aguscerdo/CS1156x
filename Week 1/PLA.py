import numpy

def runPLA(Samples):
    #Generate Line
    p1 = numpy.random.uniform(-1, 1, 2)
    p2 = numpy.random.uniform(-1, 1, 2)
    line = [p2[0]*p1[1] - p2[1]*p1[0], p2[1]-p1[1], p1[0]-p2[0]]

    #Generate Samples
    x = numpy.random.uniform(-1, 1, Samples)
    y = numpy.random.uniform(-1, 1, Samples)
    label = numpy.sign(numpy.dot(line, [numpy.ones(Samples), x, y]))
    label[label==0] = 1 #sign(0) = 1

    w = [0, 0, 0]   #Set starting weights to 0
    iterations = 0
    ones = numpy.ones(Samples)  #Vector of only 1s
    while 1:
        iterations += 1
        testL = numpy.sign(numpy.dot(w, [ones, x, y]))
        if iterations:  #On first iteration, all are misclassified
            testL[testL==0] = 1

        badLabel = []
        for i in range(Samples):
            if not label[i] == testL[i]:  #Test all bad points
                badLabel.append(i)

        if not len(badLabel):   #No bad points, break
            break

        k = numpy.random.choice(badLabel)   #Choose a random bad point

        #Adjust weights
        w[0] += label[k]
        w[1] += label[k]*x[k]
        w[2] += label[k]*y[k]

    return [iterations, line, w]


def testPLA(Samples, Runs):
    print("PLA classifier test for %d samples and %d runs" % (Samples, Runs))

    iterations = 0
    for i in range(Runs):
        result = runPLA(Samples)
        iterations += result[0]
        # print("%f, %f, %f  --  %f, %f, %f"
        #       % (result[1][0],result[1][1],result[1][2],result[2][0],result[2][1],result[2][2]))

    avgIter = iterations / Runs
    avgMissP = 1/avgIter

    print("Average Iterations: %f  --  Average miss percentage: %f" % (avgIter, avgMissP))


def main():
    print("Testing PLA.")
    testPLA(100, 1000)


main()