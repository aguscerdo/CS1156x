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


def run_svm_validation(train_points, validation_points, c, Q, d1=0, d2=1, oneVone=True):
    if oneVone:
        t = one_v_one(train_points, d1, d2, label_index=0)
        tt = one_v_one(validation_points, d1, d2, label_index=0)
    else:
        t = one_v_all(train_points, d1, label_index=0)
        tt = one_v_all(validation_points, d1, label_index=0)

    labels = np.array(t)[:, 0]
    points = np.array(t)[:, 1:]

    clf = svm.SVC(kernel='poly', C=c, degree=Q, gamma=1, coef0=1)
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


def test_one_v_one_cross(url_train, ratio, c, Q, d1, d2):
    train_points = request_points(url_train)
    points = cross_validation(train_points, ratio)  # Return 1/ratio groups. Validate with all of them

    result = []
    for C in c:
        placeholder = []
        for i in range(len(points)):
            validation = points[i]
            t = points[:i] + points[(i+1):]
            training = []
            for j in t:
               training += j
            r = run_svm_validation(training, validation, C, Q, d1=d1, d2=d2, oneVone=True)
            placeholder.append(r)
        average1 = np.average(np.array(placeholder)[:, 0])
        average2 = np.average(np.array(placeholder)[:, 1])
        average3 = np.average(np.array(placeholder)[:, 2])
        result.append([average1, average2, average3])
    return result


def iterate_validation(url_train, iterations, ratio, c, Q, d1, d2):
    results = []
    minimum_c = []
    for i in range(iterations):
        print("Iteration {}".format(i))
        r = test_one_v_one_cross(url_train, ratio,c, Q, d1, d2)
        results.append(r)
        min_Ecv = [1, 0]
        for j in range(len(r)):
            if r[j][2] <= min_Ecv[0]:
                min_Ecv[0] = r[j][0]
                min_Ecv[1] = int(j)
        minimum_c.append(min_Ecv)
    print(results)
    return [results, minimum_c]


def Q78(url_train):
    # Q7
    # Most selected C
    np.random.seed()
    C = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    results = iterate_validation(url_train, 100, 0.1, c=C, Q=2, d1=1, d2=5)

    min_c = np.array(results[1])[:, 1]
    cmap = Counter(min_c)
    index, count = cmap.most_common(1)[0]
    print(cmap)
    frequent_c = C[int(index)]

    print("\nQ7: C with the smallest Ecv\n")
    print("Most frequent C = {}, count = {}".format(frequent_c, count))


# -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-! #
print("ANSWER IS NOT RIGHT. I DO NOT KNOW WHY")
# -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-! #

Q78("http://www.amlbook.com/data/zip/features.train")

print("\n\nRuntime of %.3f seconds" % (time.time()-time_init))