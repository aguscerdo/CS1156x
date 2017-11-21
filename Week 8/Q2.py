import requests
import numpy as np
from sklearn import svm
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


def run_svm(train_points, test_points, c, Q ,kernel='poly', d1=0, d2=1, oneVone=True):
    if oneVone:
        t = one_v_one(train_points, d1, d2, label_index=0)
        tt = one_v_one(test_points, d1, d2, label_index=0)
    else:
        t = one_v_all(train_points, d1, label_index=0)
        tt = one_v_all(test_points, d1, label_index=0)

    labels = np.array(t)[:, 0]
    points = np.array(t)[:, 1:]

    clf = svm.SVC(kernel=kernel, C=c, degree=Q, gamma=1, coef0=1)
    clf.fit(points, labels)
    prediction = clf.predict(points)
    Ein = np.average(np.not_equal(labels, prediction))

    labels_test = np.array(tt)[:, 0]
    points_test = np.array(tt)[:, 1:]
    prediction_test = clf.predict(points_test)
    Eout = np.average(np.not_equal(labels_test, prediction_test))

    return [Ein, len(clf.support_vectors_), Eout]


def best_one_v_all(url_train, url_test, c, Q, kernel='poly'):
    test_points = request_points(url_test)
    train_points = request_points(url_train)

    result = []
    for i in range(10):
        r = run_svm(train_points, test_points, c, Q, kernel=kernel, d1=i, oneVone=False)
        result.append(r)

    return result


def test_one_v_one(url_train, url_test, c, Q, d1, d2, kernel='poly'):
    test_points = request_points(url_test)
    train_points = request_points(url_train)

    result = []
    for i in c:
        k = []
        for q in Q:
            r = run_svm(train_points, test_points, i, q, kernel=kernel, d1=d1, d2=d2, oneVone=True)
            k.append(r)
        result.append(k)

    return result

# Questions -------------------------------------------------------------------------------

def Q234(url_test, url_train):
    result_234 = best_one_v_all(url_test, url_train, 0.01, 2, kernel='poly')
    # Q2
    # Worst one versus all
    print("Q2: highest Ein")
    max = [0, 0]
    min = [1, 0]

    for i in range(len(result_234)):
        if result_234[i][0] > max[0]:
            max[0] = result_234[i][0]
            max[1] = i
        if result_234[i][0] < min[0]:
            min[0] = result_234[i][0]
            min[1] = i
        print("{} versus all: Ein = {}, Eout = {}".format(i, result_234[i][0], result_234[i][2]))

    print('Maximum Ein on {} versus all: Ein = {}\n'.format(max[1], max[0]))

    # Q3
    # Best one versus all
    print("Q3: lowest Ein")
    print('Minimum Ein on {} versus all: Ein = {}\n'.format(min[1], min[0]))

    # Q4
    # Difference in support vector number
    print("Q4: difference in support vectors")
    diff = abs(result_234[max[1]][1] - result_234[min[1]][1])
    print("Difference in SV = {}\n".format(diff))


def Q56(url_train, url_test):
    c_56 = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    Q_56 = [2, 5]
    result_56 = test_one_v_one(url_test, url_train, c=c_56, Q=Q_56, d1=1, d2=2)
    # Q5
    # 1 versus 5, C response
    print("\nQ5: C response")
    for i in range(len(result_56)):
        print("C={} -> Ein={}, Eout={}, SV={}".format(c_56[i], result_56[i][0][0], result_56[i][0][2], result_56[i][0][1]))

    # Q6
    # 1 versus 5, Q and C response
    print("\nQ6: C and Q response")
    for i in range(len(result_56)):
        for j in range(len(result_56[0])):
            print("C={}, Q={} -> Ein={}, Eout={}, SV={}".format(c_56[i], Q_56[j], result_56[i][j][0], result_56[i][j][2], result_56[i][j][1]))


#--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--#
url_test = "http://www.amlbook.com/data/zip/features.train"
url_train = "http://www.amlbook.com/data/zip/features.test"


Q234(url_test, url_train)
Q56(url_test, url_train)

print("\n\nRuntime of %.3f seconds" % (time.time()-time_init))