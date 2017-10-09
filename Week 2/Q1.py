from numpy import random, sum, min
from time import time

def flipCoins(coins, flips):
    allThrows = [None] * (coins)
    for i in range(coins):
        allThrows[i] = sum((random.randint(0, 2, size=flips)))

    return [allThrows[0], random.choice(allThrows), min(allThrows)]


def testFlipCoins(Runs, coins, flips):
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


def testFlipCoins(Runs, Coins, Flips):
    random.seed()
    timeStart = time()
    cum1 = 0
    cumRand = 0
    cumMin = 0

    print("Starting Test")
    for i in range(Runs):
        # print(i)
        curr = flipCoins(Coins, Flips)
        cum1 += curr[0]
        cumRand += curr[1]
        cumMin += curr[2]

    avg1 = cum1/(Runs*Flips)
    avgRand = cumRand/(Runs*Flips)
    avgMin = cumMin/(Runs*Flips)

    timeEnd = time()
    print("Avg first: %f; Avg rand: %f; Avg min: %f" % (avg1, avgRand, avgMin))
    print("Time Elapsed: %d" % (timeEnd - timeStart))


# Q1
# testFlipCoins(Runs, Coins, Flips)
print("Testing coin flips")
testFlipCoins(1500, 1000, 10)

