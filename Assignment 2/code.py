import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt

class Layer:
    weights = []
    bias = []
    activation_function = None
    activation_derivative = None

# ======================== Forward Pass ======================== #
#Macros for sigmoid and vectorization of sigmoid
sigmoid = lambda x: 1/(1 + np.exp(-x))
vector_sigmoid = np.vectorize(sigmoid)

sigmoid_derivative = lambda x: x * (1 - x)
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
def cross_entropy(y, t):
    return 0.5*((t - y[0,0])**2)[0]

def cross_entropy_derivative(y, t):
    return y - t

# ======================== Training ======================== #
def calculate_deltas(network, x, t, loss_derivative):
    deltas = []
    activations = []
    activations.append(x)

    for layer in network:
        s = z(x, layer.weights) + layer.bias
        x = layer.activation_function(s)
        activations.append(x)

    #Calculate Delta for last layer
    last_delta = loss_derivative(activations[-1], t) * (network[-1].activation_derivative(activations[-1]))
    deltas.append(last_delta)

    #Calculate Delta for the rest of the layers backwards
    for i in range(len(network) - 2, -1, -1):
        hidden_error = last_delta.dot(network[i+1].weights.T)
        last_delta = hidden_error * network[i].activation_derivative(activations[i+1])
        deltas.append(last_delta)

    deltas.reverse() #Reverses the deltas so because it was generated backwards
    return activations, deltas

def update_weights(network, activations, deltas, learning_rate):
    for i in range(len(network)):
        network[i].weights = network[i].weights - (learning_rate * activations[i].T.dot(deltas[i]))
        network[i].bias = network[i].bias - (learning_rate * np.sum(deltas[i], axis = 0, keepdims = True))

'''
network : the network to be trained
data : the data to train on
truths : the truths for the data given, in same order as the data
learning_rate : learning rate used during training
epochs : maximum number of epochs
early_stop: stops the training if it manages a whole epoch with precision = recall = 1.0
TODO : implement early stop
'''
def fit(network, data, truths, loss_function, loss_derivative, learning_rate = 0.05, epochs = 100, early_stop = False):
    losses = []

    data_batch = []
    truth_batch = []
    temp = []
    temp1 = []
    for i in range(len(data)):
        if (i == len(data) - 1):
            temp.append(data[i])
            temp1.append(truths[i])
            data_batch.append(np.array(temp))
            truth_batch.append(np.array(temp1))
        elif (i % 128 == 127):
            temp.append(data[i])
            temp1.append(truths[i])
            data_batch.append(np.array(temp))
            truth_batch.append(np.array(temp1))
            temp = []
            temp1 = []
        else:
            temp.append(data[i])
            temp1.append(truths[i])

    data = np.array(data_batch)
    truths = np.array(truth_batch)

    for e in tqdm.tqdm(range(epochs)):
        loss = 0
        for i in range(len(data)):
            prediction = predict(network, data[i])
            loss += np.sum(loss_function(prediction, truths[i]))
            activations, deltas = calculate_deltas(network, data[i], truths[i], loss_derivative)
            update_weights(network, activations, deltas, learning_rate)
        losses.append(loss)

    return losses

# ======================== Predict ======================== #
'''
network : a list of the weights, length of the list defines the depth of the network
value : the value to run prediction on, must have same size as inputnodes
'''
def predict(network, value):
    x = np.copy(value)
    for layer in network:
        s = z(x, layer.weights) + layer.bias
        x = layer.activation_function(s)
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
    return np.random.uniform(1,-1,size = (l1, l2))

def random_bias(layer):
    return np.random.uniform(1,-1,size = (1, layer))

# ======================= Init network ============================= #

def init(layers, act_func, der_act_func):
    net = []
    for i in range(len(layers) - 1):
        l = Layer()
        l.weights = random_weights(layers[i], layers[i+1])
        l.bias = random_bias(layers[i+1])
        l.activation_function = act_func[i]
        l.activation_derivative = der_act_func[i]
        net.append(l)
    return net

def encode(target, classes):
    onehot = np.zeros((target.shape[0], classes))
    onehot[np.arange(0, target.shape[0]), target] = 1
    return onehot

# ======================== Assignment Tasks ======================== #

def task_2_4():

    arr = [2,2,1]
    func = [vector_sigmoid, vector_sigmoid]
    der_func = [vector_sigmoid_derivative, vector_sigmoid_derivative]

    net = init(arr, func, der_func)

    #Prints Results before training
    print("Networks prediction before training:\n")
    print("0,0 : {} ".format(predict(net, [0,0])))
    print("0,1 : {} ".format(predict(net, [0,1])))
    print("1,0 : {} ".format(predict(net, [1,0])))
    print("1,1 : {} ".format(predict(net, [1,1])))
    print("\n")

    #Declaration of training data and truth values
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    truths = np.array([[0], [1], [1], [0]])

    #Train the network
    learning_rate = 0.8
    epochs = 1500

    print("Training the network\nlearning rate: {}\nepochs: {}\nTraining...\n".format(learning_rate, epochs))
    losses = fit(net, data, truths, cross_entropy, cross_entropy_derivative, learning_rate, epochs)

    print("Training finished!\n")

    print("Networks prediction after training:\n")
    print("0,0 : {} ".format(predict(net, [0,0])))
    print("0,1 : {} ".format(predict(net, [0,1])))
    print("1,0 : {} ".format(predict(net, [1,0])))
    print("1,1 : {} ".format(predict(net, [1,1])))

    print("\nPlotting loss graph")
    plot_loss(losses)

from sklearn.datasets import load_digits

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def softmax_der(x):
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def task_2_5():
    #Loads the images and labels
    data = load_digits()

    val = 0

    tr_data = (data["data"]/8) - 1
    target = encode(data["target"], 10)

    arr = [64,32,10]
    func = [vector_sigmoid, vector_sigmoid]
    der_func = [vector_sigmoid_derivative, vector_sigmoid_derivative]

    net = init(arr, func, der_func)

    #for i in net:
        #print(i.weights[0][:10])
    #print(predict(net, tr_data[0]))

    losses = fit(net, tr_data, target, cross_entropy, cross_entropy_derivative, learning_rate=0.001, epochs=4000)

    #print(predict(net, tr_data[0]))
    #for i in net:
        #print(i.weights[0][:10])

    successes = 0
    total = 0

    for i in range(len(tr_data)):
        pred = predict(net, tr_data[i])
        if (target[i][np.array(pred[0]).argmax()] == 1.0):
            successes += 1
        total += 1
    
    print(successes)
    print(total)

    plot_loss(losses)
    

task_2_5()
