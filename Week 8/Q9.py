import requests
import numpy as np
from sklearn import svm
from collections import Counter
import time
time_init = time.time()

def request_points(url):
    text = requests.get(url).text
    rows = text.splitlines()
    result = []
    for i in rows:
        x = []
        for j in i.split():
            x.append(float(j))
        result.append(x)
    return result


def one_v_one(points, d1, d2, label_index=0):
    # Label is first entry
    result = []
    for v in points:
        if v[label_index] == d1:
            result.append([1, v[1], v[2]])
        elif v[label_index] == d2:
            result.append([-1, v[1], v[2]])

    return result


def one_v_all(points, d1, label_index=0):
    # Label is first entry
    result = []
    for v in points:
        if v[label_index] == d1:
            result.append([1, v[1], v[2]])
        else:
            result.append([-1, v[1], v[2]])

    return result


def run_svm_validation(train_points, test_points, c, d1=0, d2=1, oneVone=True):
    if oneVone:
        t = one_v_one(train_points, d1, d2, label_index=0)
        tt = one_v_one(test_points, d1, d2, label_index=0)
    else:
        t = one_v_all(train_points, d1, label_index=0)
        tt = one_v_all(test_points, d1, label_index=0)

    labels = np.array(t)[:, 0]
    points = np.array(t)[:, 1:]

    clf = svm.SVC(kernel='rbf', C=c)#, gamma=1, coef0=1)
    clf.fit(points, labels)
    prediction = clf.predict(points)
    Ein = np.average(np.not_equal(labels, prediction))

    labels_test = np.array(tt)[:, 0]
    points_test = np.array(tt)[:, 1:]
    prediction_test = clf.predict(points_test)
    Ecv = np.average(np.not_equal(labels_test, prediction_test))

    return [Ein, len(clf.support_vectors_), Ecv]


def cross_validation(points, ratio):
    result_p = []
    k = int(len(points) * ratio)
    np.random.shuffle(points)

    for i in range(int(1/ratio)):
        k1 = i*k
        k2 = k1+k
        subset = points[k1:k2]
        result_p.append(subset)

    return result_p


def test_one_v_one_exp(url_train, url_test, c, d1, d2):
    train_points = request_points(url_train)
    test_points = request_points(url_test)
    result = []
    for C in c:
        r = run_svm_validation(train_points, test_points, C, d1=d1, d2=d2, oneVone=True)
        result.append(r)
    return result



def Q9_10(url_train, url_test):
    # Q9
    # Most selected C
    np.random.seed()
    C = [1e6, 1e4, 1e2, 1e0, 1e-2]
    results = test_one_v_one_exp(url_train, url_test, c=C, d1=1, d2=5)

    print("\nQ9: C with the smallest Ein\n")
    for i in range(len(results)):
        print("C={} -> Ein={}".format(C[i], results[i][0]))
    i = np.argmin(np.array(results)[:, 0])
    print("Minimum at C={}, Ein={}".format(C[i], results[i][0]))

    print("\n10: C with the smallest Eout\n")
    for i in range(len(results)):
        print("C={} -> Eout={}".format(C[i], results[i][2]))
    j = np.argmin(np.array(results)[:, 2])
    print("Minimum at C={}, Eout={}".format(C[j], results[j][0]))

Q9_10("http://www.amlbook.com/data/zip/features.train", "http://www.amlbook.com/data/zip/features.test")

print("\n\nRuntime of %.3f seconds" % (time.time()-time_init))