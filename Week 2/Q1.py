import numpy as np
from time import time

def flipCoins(coins, flips):
    allThrows = [None] * (coins)
    for i in range(coins):
        allThrows[i] = np.sum((np.random.randint(0, 2, size=flips)))

    return [allThrows[0], np.random.choice(allThrows), np.min(allThrows)]


def test(Runs, coins, flips):
    cum1 = 0
    cumRand = 0
    cumMin = 0
    for i in range(Runs):
        # print(i)
        curr = flipCoins(coins, flips)
        cum1 += curr[0]
        cumRand += curr[1]
        cumMin += curr[2]

    return [cum1, cumRand, cumMin]


def Run(runs, coins, flips):
    np.random.seed()
    timeStart = time()

    print("Starting Test")
    results = test(int(runs), int(coins), int(flips))
    avg1 = results[0]/(runs*flips)
    avgRand = results[1]/(runs*flips)
    print(results[2])
    avgMin = results[2]/(runs*flips)

    timeEnd = time()
    print("Avg first: %f; Avg rand: %f; Avg min: %f" % (avg1, avgRand, avgMin))
    print("Time Elapsed: %d" % (timeEnd - timeStart))


# Q1
# Run(runs, coins, flips)
Run(1500, 1000, 10)

