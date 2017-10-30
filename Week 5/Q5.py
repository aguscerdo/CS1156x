import numpy as np

def error(w): # E = (u*e^v - 2*v*e^(-u))^2
    u = w[0]
    v = w[1]
    err = u*np.exp(v) - 2 * v * np.exp(-u)
    return np.square(err)


def gradient(w):
    u = w[0]
    v = w[1]
    du = 2*(np.exp(v) + 2*v*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))
    dv = 2*(u*np.exp(v) - 2*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))
    return [du, dv]


def one_step(u0, v0, threshold, lr = 0.1):
    w = [u0, v0, 0] # Element 2 is iterations

    while w[2] < 1000:
        err = error(w)
        # print("{}: {}".format(w[2], err))
        if err < threshold:
            break

        w[2] += 1
        descent = gradient(w)
        w[0] -= lr*descent[0]
        w[1] -= lr*descent[1]

    return w


def two_step(u0, v0, iterations, lr = 0.1):
    w = [u0, v0] # Element 2 is iterations
    i = 0
    while i < iterations:
        i += 1
        # Step 1: u descent
        w[0] -= lr*gradient(w)[0]
        # Step 2: v descent
        w[1] -= lr*gradient(w)[1]

    return error(w)



# def one_step(u0, v0, threshold, lr = 0.1)
results_1 = one_step(1, 1, 10**(-14), 0.1)

# Q5
# Iterations needed for the error to drop below 10^-14
print("Iterations: {}".format(results_1[2]))


# Q6
# u and v values from Q5
print("u, v: {}, {}".format(results_1[0], results_1[1]))


# def two_step(u0, v0, iterations, lr = 0.1)
results_2 = two_step(1, 1, 15, 0.1)

# Q7
# Error of two step descent after 15 iterations
print("Error: {}".format(results_2))