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
    x.shape = (2,1)
    activations.append(np.array(x))

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
    ac = np.append(activations[-2], [1]) #Adds one so the last element is the bias-delta
    ac.shape = (network[-1].shape[0], len(ac))
    deltas.append(last_delta * ac)

    #Calculate Delta for the rest of the layers backwards
    for i in range(len(network) - 2, -1, -1):
       last_delta = network[i + 1][:,:-1].T.dot(last_delta)  * vector_sigmoid_derivative(zums[i])
       deltas.append(last_delta * np.append(activations[i], [1]))

    deltas.reverse()
    return deltas

def update_weights(network, deltas, learning_rate):
    if(len(network) != len(deltas)):
        print("The deltas given does not match the number of layers in the network!")
        return None

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
        return None

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
        return None

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
    return np.array([[random.uniform(-1, 1) for j in range(l1 + 1)] for i in range(l2)])

def init():
    w1 = random_weights(2, 2)
    w2 = random_weights(2, 1)

    net = [w1, w2]
    
    return net



net = init()

#w1 = np.array([[4, 4, -2], [-3, -3, 5]])
#w2 = np.array([[5,5,-5]])
#net = [w1, w2]

data = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
truths = [[-1], [1], [1], [-1]]

print("\n\n{}\n\n".format(net))

print("0,0 : {} ".format(predict(net, [0,0])))
print("0,1 : {} ".format(predict(net, [0,1])))
print("1,0 : {} ".format(predict(net, [1,0])))
print("1,1 : {} ".format(predict(net, [1,1])))

lss = fit(net, data, truths, learning_rate=0.01, epochs=900)

print("0,0 : {} ".format(predict(net, [0,0])))
print("0,1 : {} ".format(predict(net, [0,1])))
print("1,0 : {} ".format(predict(net, [1,0])))
print("1,1 : {} ".format(predict(net, [1,1])))

print("\n\n{}\n\n".format(net))

plot_loss(lss)
