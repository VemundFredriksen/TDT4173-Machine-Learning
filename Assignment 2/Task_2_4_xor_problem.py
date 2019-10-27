import numpy as np
import random
import matplotlib.pyplot as plt

# ======================== Forward Pass ======================== #
#Macros for sigmoid and vectorization of sigmoid
sigmoid = lambda x: 1/(1 + np.exp(-x))
vector_sigmoid = np.vectorize(sigmoid)

sigmoid_derivative = lambda x: sigmoid(x)*(1 - sigmoid(x))
vector_sigmoid_derivative = np.vectorize(sigmoid_derivative)

#The aggregate, expects the bias to be included in the weights (Bias Trick)
def z(w, x):
    return np.dot(w, x)

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
    return 0.5*((t - y[0,0])**2)[0]

def loss_derivative(y, t):
    return y - t

# ======================== Training ======================== #
def calculate_deltas(network, x, t):
    deltas = []
    activations = []
    zums = []

    x = np.array(x)
    activations.append(x)

    x = np.copy(activations[0])
    for layer in network:
        x = np.append(x, [1])
        x.shape = (len(x), 1)
        s = z(layer, x)
        x = a(s)
        zums.append(s)
        activations.append(x)

    #Calculate Delta for last layer
    last_delta = loss_derivative(activations[-1], t) * (vector_sigmoid_derivative(zums[-1]))
    ac = np.append(activations[-2], [1]).T #Adds the 1 to align with bias-trick
    ac = np.vstack([np.copy(ac) for i in range(last_delta.shape[0])])
    deltas.append(ac * last_delta)

    #Calculate Delta for the rest of the layers backwards
    for i in range(len(network) - 2, -1, -1):
        last_delta = network[i + 1][::,:-1].T.dot(last_delta) * sigmoid_derivative(zums[i])
        ac = np.append(activations[i], [1]).T #Adds the 1 to align with bias-trick
        ac = np.vstack([np.copy(ac) for i in range(last_delta.shape[0])])
        deltas.append(ac * last_delta)

    deltas.reverse() #Reverses the deltas so because it was generated backwards
    return deltas

def update_weights(network, deltas, learning_rate):
    if(len(network) != len(deltas)):
        print("The deltas given does not match the number of layers in the network!")
        exit()

    for i in range(len(network)):
        network[i] = network[i] - (learning_rate * deltas[i])

'''
network : the network to be trained
data : the data to train on
truths : the truths for the data given, in same order as the data
learning_rate : learning rate used during training
epochs : maximum number of epochs
early_stop: stops the training if it manages a whole epoch with precision = recall = 1.0
TODO : implement early stop
'''
def fit(network, data, truths, learning_rate = 0.05, epochs = 100, early_stop = False):
    if(len(data) != len(truths)):
        print("Training data is not aligned with truth-data")
        exit()

    losses = []

    for e in range(epochs):
        epochLoss = 0
        for i in range(len(data)):
            prediction = predict(network, data[i])
            epochLoss += loss(prediction, truths[i])
            deltas = calculate_deltas(network, data[i], truths[i])
            update_weights(network, deltas, learning_rate)
        losses.append(epochLoss/len(data))

    return losses

# ======================== Predict ======================== #
'''
network : a list of the weights, length of the list defines the depth of the network
value : the value to run prediction on, must have same size as inputnodes
'''
def predict(network, value):
    if(len(value) + 1 != network[0].shape[1]):
        print("Input Data does not fit network!")
        exit()

    x = np.copy(value)
    for layer in network:
        x = np.append(x, [1])
        x.shape = (len(x), 1)
        s = z(layer, x)
        x = a(s)

    x.shape = (len(x), 1)
    return x

# ======================== Plot ======================== #

def plot_loss(losses):
    x = [ (i + 1) for i in range(len(losses))]
    plt.plot(x, losses)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()

'''
l1 : number of nodes in first layer
l2 : number of nodes in next layer
'''
def random_weights(l1, l2):
    w = np.array([[np.random.normal(-1, 1) for j in range(l1 + 1)] for i in range(l2)])
    w.shape = (l2, l1+1)

    return w

# ======================== Assignment Tasks ======================== #

def task_2_4():

    #Initializes random weights
    w1 = random_weights(2, 2)
    w2 = random_weights(2, 1)
    net = [w1, w2]

    #Prints Results before training
    print("Networks prediction before training:\n")
    print("0,0 : {} ".format(predict(net, [0,0])))
    print("0,1 : {} ".format(predict(net, [0,1])))
    print("1,0 : {} ".format(predict(net, [1,0])))
    print("1,1 : {} ".format(predict(net, [1,1])))
    print("\n")

    #Declaration of training data and truth values
    data = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
    truths = [[0], [1], [1], [0]]

    #Train the network
    learning_rate = 0.8
    epochs = 1500

    print("Training the network\nlearning rate: {}\nepochs: {}\nTraining...\n".format(learning_rate, epochs))
    losses = fit(net, data, truths, learning_rate, epochs)
    print("Training finished!\n")

    print("Networks prediction after training:\n")
    print("0,0 : {} ".format(predict(net, [0,0])))
    print("0,1 : {} ".format(predict(net, [0,1])))
    print("1,0 : {} ".format(predict(net, [1,0])))
    print("1,1 : {} ".format(predict(net, [1,1])))

    print("\nPlotting loss graph")
    plot_loss(losses)

task_2_4()
