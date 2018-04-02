import numpy as np
import matplotlib.pyplot as plt


def svm_stochastic_gradient_descent(X, y):
    weights = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000
    errors = []

    for epoch in range(1, epochs):
        error = 0
        # x is nothing else. but X[i]
        for i, x in enumerate(X):
            # case of misclassification
            if (y[i] * np.dot(x, weights)) < 1:
                weights = weights + eta * ((x * y[i]) + (-2 * (1/epoch) * weights))
                error = 1
            # case of correct classification:
            else:
                weights = weights + eta * (-2 * (1/epoch) * weights)
        errors.append(error)
        print(len(errors))
    return weights, errors


def plot_errors(errors):
    plt.plot(errors, '|')
    plt.ylim(0.5, 1.5)
    plt.yticks([])
    plt.xlabel('Epochs')
    plt.ylabel('Misclassified')
    plt.title('Change of errors in regards to epoch')
    plt.show()


def plot_svm(X, weights):
    # plot the train data
    for d, sample in enumerate(X):
        if d < 2:
            plt.scatter(sample[0], sample[1], marker='_', color='black')
        else:
            plt.scatter(sample[0], sample[1], marker='+', color='black')
    # plot the test data
    plt.scatter(2.0, 2.0, marker='_', color='red')
    plt.scatter(4.0, 3.0, marker='+', color='blue')
    #plot the hyperplane
    x2 = [weights[0], weights[1], -weights[1], weights[0]]
    x3 = [weights[0], weights[1], weights[1], -weights[0]]
    hyperplane = np.array([x2, x3])
    X, y, U, V = zip(*hyperplane)
    ax = plt.gca()
    ax.quiver(X, y, U, V, scale=1, color='k')
    plt.show()


def prepare_data():
    X = np.array([[-2, 4, -1], [4, 1, -1], [1, 6, -1], [2, 4, -1], [6, 2, -1]])
    # output labels (either -1 or 1)
    y = np.array([-1, -1, 1, 1, 1])
    for d, sample in enumerate(X):
        if d < 2:
            plt.scatter(sample[0], sample[1], marker='_')
            print(sample[0], sample[1])
        else:
            print(sample[0], sample[1])
            plt.scatter(sample[0], sample[1], marker='+')
    # plotting a dummy hyperplane, just for sake of visualization
    plt.plot([-2, 6], [6, 0.5])
    plt.show()
    return X, y


if __name__ == '__main__':
    X, y = prepare_data()
    weights, errors = svm_stochastic_gradient_descent(X, y)
    plot_errors(errors)
    plot_svm(X, weights)
