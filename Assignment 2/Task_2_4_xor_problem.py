import numpy as np
import random

# ======================== Forward Pass ======================== #
#Macros for sigmoid and vectorization of sigmoid
sigmoid = lambda x: 1/(1 + np.exp(-x))
vector_sigmoid = np.vectorize(sigmoid)

#The aggregate, expects the bias to be included in the weights (Bias Trick)
def z(w, x):
    return np.dot(np.transpose(w), x)

#Activation function, expects nparray
def a(z):
    return vector_sigmoid(z)

# ======================== Loss ======================== #

#Loss function
'''
y : predicted value
t : truth
'''
def loss(y, t):
    return 0.5*(t - y)**2

# ======================== Training ======================== #
'''
network : the network to be trained
data : the data to train on
truths : the truths for the data given, in same order as the data
learning_rate : learning rate used during training
epochs : maximum number of epochs
stopWhenPerfect: stops the training if it manages a whole epoch with precision = recall = 1.0
'''
def train(network, data, truths, learning_rate = 0.05, epochs = 10, stopWhenPerfect = False):
    if(len(data) != len(truths)):
        print("Training data is not aligned with truth-data")
        return None

    losses = []

    for e in epochs:
        epochLoss = 0
        for i in range(len(data)):
            prediction = predict(network, data[i])
            epochLoss += loss(prediction, truths[i])
        losses.append(epochLoss/len(data))

    return losses

# ======================== Predict ======================== #
'''
network : a list of the weights, length of the list defines the depth of the network
value : the value to run prediction on, must have same size as inputnodes
'''
def predict(network, value):
    if(len(value) + 1 != len(network[0])):
        print("Input Data does not fit network!")
        return None

    x = np.copy(value)
    for layer in network:
        x = np.append(x, [1])
        s = z(layer, x)
        x = a(s)

    return x

'''
l1 : number of nodes in first layer
l2 : number of nodes in next layer
'''
def random_weights(l1, l2):
    return np.array([[random.uniform(-1, 1) for j in range(l2)] for i in range(l1 + 1)])

def init():
    w1 = random_weights(2, 2)
    w2 = random_weights(2, 1)
    net = [w1, w2]

    return net

#net = init()
#predict(net, np.array([2, 3]))

w1 = np.array([[4, -3], [4, -3], [-2, 5]])
w2 = np.array([5, 5, -5])
net = [w1, w2]

print(predict(net, np.array([1,1])))