import numpy as np


def generate_points(file_name):
    file = open(file_name, 'r')
    x1 = []
    x2 = []
    label = []
    for line in file:
        reading = line.split()
        x1.append(float(reading[0]))
        x2.append(float(reading[1]))
        label.append(float(reading[2]))
    file.close()

    return [x1, x2, label]


def transformation(datapoints):
    t = []
    # Transformation = (1, x1, x2, x1^2, x2^2, x1*x2, |x1-x2|, |x1+x2|)
    t.append(np.ones(len(datapoints[0])))
    t.append(datapoints[0])
    t.append(datapoints[1])
    t.append(np.square(datapoints[0]))
    t.append(np.square(datapoints[1]))
    t.append(np.multiply(datapoints[0], datapoints[1]))
    t.append(np.abs(np.subtract(datapoints[0], datapoints[1])))
    t.append(np.abs(np.add(datapoints[0], datapoints[1])))

    return np.transpose(t)


def evaluate_solution(points, label, solution):
    new_label = np.sign(np.dot(points, solution))
    new_label[new_label==0] = 1
    misses = np.not_equal(label, new_label)

    return np.average(misses)


def linear_regression(training_file, testing_file):
    training = generate_points(training_file)
    t_training = transformation(training)

    w = np.linalg.lstsq(t_training, training[2])[0]
    w.reshape(len(w), 1)
    E_in = evaluate_solution(t_training, training[2], w)

    testing = generate_points(testing_file)
    t_testing = transformation(testing)
    E_out = evaluate_solution(t_testing, testing[2], w)

    return [E_in, E_out, w]


def decay_weight(points, label, lambda_power):
    N = len(points[0])

    lamb = 10**lambda_power
    # Z^t
    transpose = np.transpose(points)
    # (Z^t * Z + lambda * I)^-1
    inverse = np.linalg.inv(np.matmul(transpose, points) + lamb*np.eye(N))
    prel = np.matmul(inverse, transpose)

    w = np.matmul(prel, label)

    return w


def regression_decay(training_file, testing_file, lambda_power):
    training = generate_points(training_file)
    t_training = transformation(training)

    w = decay_weight(t_training, training[2], lambda_power)
    w.reshape(len(w), 1)
    E_in = evaluate_solution(t_training, training[2], w)

    testing = generate_points(testing_file)
    t_testing = transformation(testing)
    E_out = evaluate_solution(t_testing, testing[2], w)

    return [E_in, E_out, w]


def test_k(training_file, testing_file, range_min=-2, range_max=2):
    E_out = []
    for i in range(range_min, range_max+1):
        err = regression_decay(training_file, testing_file, i)[1]
        E_out.append(err)
        print("E_out k = {}: {}".format(i, err))

    return E_out


# Q2
result_2 = linear_regression(training_file="in.txt", testing_file="out.txt")
print("Q2:\nE_in: {}  ||  E_out: {}\n".format(result_2[0], result_2[1]))


# Q3
result_3 = regression_decay(training_file="in.txt", testing_file="out.txt", lambda_power=-3)
print("Q3:\nE_in: {}  ||  E_out: {}\n".format(result_3[0], result_3[1]))


# Q4
result_4 = regression_decay(training_file="in.txt", testing_file="out.txt", lambda_power=3)
print("Q4:\nE_in: {}  ||  E_out: {}\n".format(result_4[0], result_4[1]))


# Q5
print("Q5:")
result_5 = test_k(training_file="in.txt", testing_file="out.txt", range_min=-2, range_max=2)

