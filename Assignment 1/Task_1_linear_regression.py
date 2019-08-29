import numpy as np
from matplotlib import pyplot as plt

# Task 1.1
def ordinary_least_squares(input_data, expected_values):
    return np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(input_data), input_data)), np.transpose(input_data)), expected_values)

def mean_squared_error(weights, input_values, expected_values):
    mse_1 = np.dot(np.dot(np.dot(np.transpose(weights), np.transpose(input_values)), input_values), weights)
    mse_2 = np.dot(np.multiply(2, np.transpose(np.dot(input_values, weights))), expected_values)
    mse_3 = np.dot(np.transpose(expected_values), expected_values)

    return mse_1 - mse_2 + mse_3

def load_2d_data(path):
    file = open(path, "r")
    content = file.readlines()[1::]
    file.close()

    x = []
    y = []
    for line in content:
        data = line.split(',')
        x.append((1, float(data[0]), float(data[1])))
        y.append(float(data[2]))

    return (np.array(x), np.array(y))

def load_1d_data(path):
    file = open(path, "r")
    content = file.readlines()[1::]
    file.close()

    x = []
    y = []
    for line in content:
        data = line.split(',')
        x.append((1, float(data[0])))
        y.append(float(data[1]))

    return (np.array(x), np.array(y))

def plot_1d(x, y, weights):
    x0 = np.linspace(0, 1, 100)
    x1 = []
    for xt in x0:
        x1.append((1, xt))

    graph = np.dot(weights, np.transpose(x1))


    plt.plot(x, y, "ro")
    plt.plot(x0, graph)

    plt.legend(["Given Values", "Regressed function"])

    plt.show()


#Task 1.2 - 2D Regression
def task1_2():
    training_data = load_2d_data("dataset/regression/train_2d_reg_data.csv")
    test_data = load_2d_data("dataset/regression/test_2d_reg_data.csv")

    weights = ordinary_least_squares(training_data[0], training_data[1])
    print(weights)

    mse_training = mean_squared_error(weights, training_data[0], training_data[1])
    mse_test = mean_squared_error(weights, test_data[0], test_data[1])
    print("Mean Squared Error for training set:\t" + str(mse_training))
    print("Mean Squared Error for test set:\t" + str(mse_test))

#Task 1.3 - 1D Regression
def task1_3():
    training_data = load_1d_data("dataset/regression/train_1d_reg_data.csv")
    test_data = load_1d_data("dataset/regression/test_1d_reg_data.csv")

    weights = ordinary_least_squares(training_data[0], training_data[1])
    mse_training = mean_squared_error(weights, training_data[0], training_data[1])
    mse_test = mean_squared_error(weights, test_data[0], test_data[1])

    print("Mean Squared Error for training set:\t" + str(mse_training))
    print("Mean Squared Error for test set:\t" + str(mse_test))

    x = []
    for x1 in training_data[0]:
        x.append(x1[1])

    #Plot training data points
    plot_1d(x, training_data[1], weights)

    x_tst = []
    for x1 in test_data[0]:
        x_tst.append(x1[1])

    #Plot test data points
    plot_1d(x_tst, test_data[1], weights)

# Uncomment task you want to run
#task1_2()
task1_3()