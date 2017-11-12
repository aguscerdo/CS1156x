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

    return np.transpose([x1, x2, label])


def transformation(datapoints):
    t = []
    # Transformation = (1, x1, x2, x1^2, x2^2, x1*x2, |x1-x2|, |x1+x2|)
    N = len(datapoints)
    t.append(np.ones(len(datapoints)))
    t.append(datapoints[:N,0])
    t.append(datapoints[:N,1])
    t.append(np.square(datapoints[:N,0]))
    t.append(np.square(datapoints[:N,1]))
    t.append(np.multiply(datapoints[:N,0], datapoints[:N,1]))
    t.append(np.abs(np.subtract(datapoints[:N,0], datapoints[:N,1])))
    t.append(np.abs(np.add(datapoints[:N,0], datapoints[:N,1])))

    return np.transpose(t)


def validation_regression(train, t_label, val, v_label):
    error = []
    w_all = []
    t_label.reshape(len(t_label), 1)

    # Evaluate using different number of coefficents
    for k in range(len(train[0])):
        reduced_d = train[:, :k+1]
        reduced_d_transpose = np.transpose(reduced_d)

        w_0 = np.linalg.inv(np.matmul(reduced_d_transpose, reduced_d))
        w_1 = np.matmul(w_0, reduced_d_transpose)
        w = np.dot(w_1, t_label)

        eval_labels = np.sign(np.dot(val[:, :k+1], w))
        eval_labels[eval_labels==0] = 1
        err_in = np.average(np.not_equal(eval_labels, v_label))
        error.append(err_in)
        w_all.append(w)

    return [error, w_all]


def out_of_sample_test(w, test, t_label):
    E_out = []
    for k in range(len(w)):
        new_label = np.sign(np.dot(test[:, :k+1], w[k]))
        new_label[new_label==0] = 1
        err = np.average(np.not_equal(new_label, t_label))
        E_out.append(err)

    return E_out


def linear_regression(training_file, testing_file, N, reverse_order=False):
    training = generate_points(training_file)

    if N >= len(training):
        return [["ERROR: N needs to be smaller than the dataset length"]]

    if reverse_order:
        training = np.flip(training, axis=0)

    train_t = transformation(training)

    D_train = train_t[:N]
    D_train_label = training[:N, 2]
    D_val = train_t[N:]
    D_val_label = training[N:, 2]

    validation = validation_regression(D_train, D_train_label, D_val, D_val_label)
    error = validation[0]
    w = validation[1]

    test = generate_points(testing_file)
    test_t = transformation(test)
    test_label = test[:,2]

    E_out = out_of_sample_test(w, test_t, test_label)

    return [error, E_out]


# -------------------------------------------------------------------#
# Answers:


result_1_2 = linear_regression("in.txt", "out.txt", 25)


# Q1
# Minimum k for E_val
result_q1 = result_1_2[0]
min_1 = [1, 1]
for i in range(len(result_q1)):
    if result_q1[i] < min_1[1]:
        min_1[0] = i
        min_1[1] = result_q1[i]
    print("k={}, E_val = {}".format(i, result_q1[i]))
print('Q1: Minimum E_val at k={}, E_val = {}\n'.format(min_1[0], min_1[1]))


# Q2
# Minimum k for E_out
result_q2 = result_1_2[1]
min_2 = [1, 1]
for i in range(len(result_q2)):
    if result_q2[i] < min_2[1]:
        min_2[0] = i
        min_2[1] = result_q2[i]
    print("k={}, E_out = {}".format(i, result_q2[i]))
print('Q2: Minimum E_out at k={}, E_val = {}\n'.format(min_2[0], min_2[1]))



result_3_4 = linear_regression("in.txt", "out.txt", 10, reverse_order=True)


# Q3
# Minimum k for E_val, with 10 train points
result_q3 = result_3_4[0]
min_3 = [1, 1]
for i in range(len(result_q3)):
    if result_q3[i] < min_3[1]:
        min_3[0] = i
        min_3[1] = result_q3[i]
    print("k={}, E_val = {}".format(i, result_q3[i]))
print('Q3: Minimum E_val at k={}, E_val = {}\n'.format(min_3[0], min_3[1]))


# Q4
# Minimum k for E_out, with 10 train points
result_q4 = result_3_4[1]
min_4 = [1, 1]
for i in range(len(result_q4)):
    if result_q4[i] < min_4[1]:
        min_4[0] = i
        min_4[1] = result_q4[i]
    print("k={}, E_out = {}".format(i, result_q4[i]))
print('Q4: Minimum E_out at k={}, E_val = {}\n'.format(min_4[0], min_4[1]))


# Q5
# E_out for best model on 2 and 4
print("Q5: E_out_1: {}; E_out_2: {}\n".format(min_2[1], min_4[1]))