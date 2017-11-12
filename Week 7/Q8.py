import numpy as np
from sklearn import svm
from math import inf


def generate_line():
    p1 = np.random.uniform(-1, 1, 2)
    p2 = np.random.uniform(-1, 1, 2)
    return [p2[0] * p1[1] - p2[1] * p1[0], p2[1] - p1[1], p1[0] - p2[0]]


def generate_points(N, line):
    p1 = np.random.uniform(-1, 1, N)
    p2 = np.random.uniform(-1,1, N)
    points = [np.ones(N), p1, p2]
    labels = np.sign(np.dot(np.transpose(points), line))
    labels[labels==0] = 1
    points.append(labels)

    return np.transpose(points)


def test_svg(svg, line, N=1000):
    p1 = np.random.uniform(-1, 1, N)
    p2 = np.random.uniform(-1,1, N)

    points = np.transpose([np.ones(N), p1, p2])
    labels = np.sign(np.dot(points, line))
    labels[labels==0] = 1

    svg_labels = svg.predict(points)

    return np.average(np.not_equal(svg_labels, labels))


def test_pla(w, line, N=1000):
    p1 = np.random.uniform(-1, 1, N)
    p2 = np.random.uniform(-1,1, N)
    points = np.transpose([np.ones(N), p1, p2])
    labels = np.sign(np.dot(points, line))
    labels[labels==0] = 1

    w_labels = np.sign(np.dot(points, w))
    w_labels[w_labels==0] = 1

    return np.average(np.not_equal(w_labels, labels))


def PLA(data, line):
    w = [0, 0, 0]
    points = data[:, :3]
    labels = data[:, 3]
    N = len(labels)
    epoch = 0
    while 1:
        test_labels = np.sign(np.dot(points, w))
        test_labels[test_labels==0] = 1

        bad_labels = []
        for i in range(N):
            if not test_labels[i] == labels[i]:
                bad_labels.append(i)

        if not len(bad_labels): # No bad labels, stop
            break

        k = np.random.choice(bad_labels)

        # Adjust Weights
        w += points[k]*labels[k]

    return w


def SVG(data, line):
    points = data[:, :3]
    labels = data[:, 3]
    try:
        clf = svm.SVC(kernel='linear', C = 1000)
        clf.fit(points, labels)
        # w = clf.coef_[0]
        return [clf, len(clf.support_vectors_)]
    except Exception as e:
        return e


def test(N, iterations, N_test=5000):
    err_pla = []
    err_svg = []
    support_vectors = []

    errors = 0
    while(iterations):
        line = generate_line()
        data = generate_points(N, line)

        w_pla = PLA(data, line)
        svg = SVG(data, line)
        try:
            w_svg = svg[0]  # Returns clf itself, not vector
        except:
            errors += 1
            continue

        support_v = svg[1]

        err_pla.append(test_pla(w_pla, line, N=N_test))
        err_svg.append(test_svg(w_svg, line, N=N_test))
        support_vectors.append(support_v)
        iterations -= 1

    diff = np.less(err_svg, err_pla)
    return [np.average(diff), np.average(support_vectors)]



# Q8
# For N = 10, 1000 trials, percentage of SVG better than PLA
result_8 = test(10, 1000, N_test=15000)
print("Q8: E_svg < E_pla = %.4f\n" % result_8[0])


# Q9
# For N = 100, 1000 trials, percentage of SVG better than PLA
result_9_10 = test(100, 1000, N_test=15000)
print("Q9: E_svg < E_pla = %.4f\n" % result_9_10[0])


# Q10
# Average number of support vectors for N = 100
print("Q10: Avg support vectors = %.5f\n" % result_9_10[1])


