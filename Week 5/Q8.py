import numpy as np

def generate_line():
    p1 = np.random.uniform(-1, 1, 2)
    p2 = np.random.uniform(-1, 1, 2)
    return [p2[0] * p1[1] - p2[1] * p1[0], p2[1] - p1[1], p1[0] - p2[0]]


def SGD(w, point, label):
    numerator = np.multiply(point, label)
    denominator = (1 + np.exp(label*np.dot(w, point)))
    error = -np.divide(numerator, denominator)
    return error


def cross_entropy_test(weights, target, test_size=1000):
    # Generate test data
    test_points = np.array([np.ones(test_size), np.random.uniform(-1, 1, test_size), np.random.uniform(-1, 1, test_size)]).reshape(3, test_size)
    test_label = np.sign(np.dot(target, test_points)).reshape(1, test_size)

    err = np.log(1 + np.exp(-1 * np.multiply(test_label, np.dot(weights, test_points))))
    return np.average(err)


def logistic_regression(samples, threshold = 0.01, lr=0.1, test_size=1000):
    line = generate_line()  # Create target function
    points = [np.ones(samples), np.random.uniform(-1, 1, samples), np.random.uniform(-1, 1, samples)]   # Creat sample data
    label = np.sign(np.dot(line, points)).reshape(1, samples)   # Label sample data

    w = np.array([0., 0., 0.])
    epochs = 0
    indices = np.arange(samples)
    while 1:
        epochs += 1
        previous_w = np.copy(w)

        np.random.shuffle(indices)
        for d in indices:   # Randomized stochastic descent
            descent = SGD(w, [1, points[1][d], points[2][d]], label[0][d])
            w -= np.multiply(descent, lr) # Adjust weights

        dw = np.linalg.norm(w - previous_w) # Obtain magnitude of descent
        if np.any(dw < threshold): break

    # Test with Cross-Entropy-Error
    error = cross_entropy_test(w, line, test_size)

    return [error, epochs]


def test_logistic_regression(samples, runs, threshold=0.01, lr=0.1, test_size=1000):
    error = 0
    epochs = 0
    i = runs
    while i:
        print("Run %d" % i)
        i -= 1
        result = logistic_regression(samples, threshold, lr, test_size)
        error += result[0]
        epochs += result[1]

    error /= runs
    epochs /= runs

    return [error, epochs]


results = test_logistic_regression(samples=100, runs=100, threshold=0.01, lr=0.01, test_size=1000)

# Q8
# Out of sample error
print("E_out: {}".format(results[0]))


# Q9
# Average epochs or iterations
print("Epochs: {}".format(results[1]))
