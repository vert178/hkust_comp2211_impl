import numpy as np


# I don't really see this relabelling function useful now but i really like it so I am saving it
def Relabel(data):
    # Do some relabeling for non-number type object
    for i in range(data.shape[1] - 1):
        if not isNum(data[i][0]):
            v = np.vectorize(lambda x: np.where(
                np.unique(data[:, i]) == x)[0][0])
            data[:, i] = v(data[:, i])
    return data


def GenerateNormalDistribution(mean, sd, count=1):
    # An algorithm to generate normal distribution from uniform distribution
    # Using the neat little identity cdf_normal(x) = 1/2 + 1/2erf(x/sqrt(2),
    # We can turn a random number from -1 to 1, to a number with mean 0 sd 1

    # The messy lambda is just sqrt(2) * inverse of error function, up to x ** 7
    # which should be enough since if x < 1, x ** 9 <= 1e-9
    result = 2 * np.random.rand(count) - 1
    erf_inverse = np.vectorize(lambda x: (
        PI / 2) ** 0.5 * (x + PI/12 * x**3 + 7*PI**2/480 * x**5 + 127*PI**3/40320 * x**7))
    return mean + sd * erf_inverse(result)