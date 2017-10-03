import matplotlib.pyplot as plt
import numpy

SAMPLES = 100
numpy.random.seed()

p1 = [numpy.random.uniform(-1, 1), numpy.random.uniform(-1, 1)]
p2 = [numpy.random.uniform(-1, 1), numpy.random.uniform(-1, 1)]

slope = (p2[1]-p1[1])/(p2[0]-p1[0])
intercept = p2[1] - slope*p2[0]
line = [p2[0] * p1[1] - p2[1] * p1[0], p2[1] - p1[1], p1[0] - p2[0]]


xc = numpy.random.uniform(-1, 1, SAMPLES)
yc = numpy.random.uniform(-1, 1, SAMPLES)

dot = numpy.sign(numpy.dot(line, [numpy.ones(SAMPLES), xc, yc]))
cmap = plt.get_cmap('bwr')
colors = cmap(dot)

plt.scatter(xc, yc, c=colors)
plt.plot([-1,1], [intercept - slope,intercept + slope], c='k')
plt.axis([-1, 1, -1, 1])
plt.axvline(0, c='k', alpha=0.5)
plt.axhline(0, c='k', alpha=0.5)
plt.show()