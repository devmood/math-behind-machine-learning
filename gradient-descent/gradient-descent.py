import numpy as np
import matplotlib.pyplot as plt


def step_gradient(a_current, b_current, etha, data):
    a_gradient = 0
    b_gradient = 0
    n = float(len(data))
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        a_gradient += -(2/n) * (y - ((b_current * x) + a_current))
        b_gradient += -(2/n) * x * (y - ((b_current * x) + a_current))
    new_a = a_current - (etha * a_gradient)
    new_b = b_current - (etha * b_gradient)
    return new_a, new_b


def change_coefficients(starting_a, starting_b, iterations):
    a = starting_a
    b = starting_b
    for _ in range(iterations):
        a, b = step_gradient(a, b, 0.0001, np.array(data))
    return a, b


def measure_error(a, b, data):
    error = 0
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        error += (y - (a*x + b)) ** 2
        # or error += (y - predict()) ** 2
    return error / float(len(data))


if __name__ == '__main__':
    data = [[1, 2], [3, 4], [6, 7]] # find it out later how to write it in here
    initial_a = 0
    initial_b = 0
    iterations = 1000
    print('Before running the gradient descent algorithm...\na: ', initial_a, ', b: ', initial_b, ',\
    error: ', measure_error(initial_a, initial_b, data))
    a, b = change_coefficients(initial_a, initial_b, iterations)
    print('After running the gradient descent algorithm...\na: ', initial_a, ', b: ', initial_b, ',\
    error: ', measure_error(a, b, data))
    