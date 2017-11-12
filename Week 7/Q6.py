from numpy.random import uniform
from numpy import minimum, average


def expectation(n):
    e1 = uniform(0,1,n)
    e2 = uniform(0,1,n)
    e = minimum(e1, e2)

    return [average(e1), average(e2), average(e)]


# Q6
# Expected value for e1, e2, e
result = expectation(1000000)
print("e1: {}, e2: {}, e: {}".format(result[0], result[1], result[2]))