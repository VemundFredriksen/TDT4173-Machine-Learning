import numpy as np
from matplotlib import pyplot as plt
import math
import random

epsillon = 1e-15
learning_rate = 0.005

def load_data(path):
    file = open(path, "r")
    content = file.readlines()
    file.close()

    x = [] #features
    y = [] #class
    for line in content:
        data = line.split(',')
        x.append((1, float(data[0]), float(data[1])))
        y.append(float(data[2]))

    return (np.array(x), np.array(y))

def generate_weights(input_number):
    return [random.random() for i in range(input_number + 1)]

def cross_entropy(pred, expected):
    return math.fabs(expected * math.log(pred + epsillon, 2)) + math.fabs((1 - expected)* math.log(1 - pred + epsillon, 2))

def sigmoid(x):
    return 1/(1 + (math.pow(math.e ,x*-1)))

def predict(weights, input_values):
    sums = np.dot(weights, np.transpose(input_values))
    preds = []
    for s in sums:
        preds.append(sigmoid(s))
    return preds

def validate(weights, input_values, labels):
    predictions = predict(weights, input_values)
    err = 0
    for i in range(len(predictions)):
        err += cross_entropy(predictions[i], labels[i])
    err = err / len(predictions)
    return err

def update_weights(weights, input_values, class_def):
    predictions = predict(weights, input_values)
    err = validate(weights, input_values, class_def)
    delta = np.multiply(np.dot(np.subtract(predictions, class_def), input_values), learning_rate)

    weights = np.subtract(weights, delta)

    return weights, err

def train(weights, training_data, training_labels, test_data, test_labels, epochs):
    train_err = []
    test_err = []
    for i in range(epochs):
        res = update_weights(weights, training_data, training_labels)
        val = validate(weights, test_data, test_labels)
        weights = res[0]
        train_err.append(res[1])
        test_err.append(val)

    x = [i + 1 for i in range(len(train_err))]
    plt.plot(x, train_err)
    plt.plot(x, test_err)
    plt.legend(["Training Loss", "Test Loss"])
    plt.show()

    print("Final Train Error: " + str(train_err[len(train_err) - 1]))
    print("Final Test Error: " + str(test_err[len(test_err) - 1]))
    return weights

def BoundryCalc(weights, x1):
    return (-1 * weights[0] - weights[1] * x1)/weights[2]

def visualize_class(input_values, class_def, weights):
    c1_x = []
    c1_y = []
    c2_x = []
    c2_y = []

    for i in range(len(class_def)):
        if(class_def[i] == 0):
            c1_x.append(input_values[i][1])
            c1_y.append(input_values[i][2])
            continue
        c2_x.append(input_values[i][1])
        c2_y.append(input_values[i][2])

    # Boundry Math
    x1 = np.linspace(0, 1, 100)
    x2 = []
    for x in x1:
        x2.append(BoundryCalc(weights, x))

    plt.plot(c1_x, c1_y, 'bo')
    plt.plot(c2_x, c2_y, 'ro')
    plt.plot(x1, x2)
    plt.legend(["Class 0", "Class 1", "Decision Boundary"])
    plt.show()


#task 2.1 and 2.2 (change the path to the correct data)
def task2():
    training_data = load_data("dataset/classification/cl_train_1.csv")
    test_data = load_data("dataset/classification/cl_test_1.csv")

    weights = generate_weights(2)
    weights = train(weights, training_data[0], training_data[1], test_data[0], test_data[1], 1000)
    visualize_class(test_data[0], test_data[1], weights)



task2()