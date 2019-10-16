import numpy as np

# ======================== Forward Pass ======================== #
#Macros for sigmoid and vectorization of sigmoid
sigmoid = lambda x: 1/(1 + np.exp(-x))
vector_sigmoud = np.vectorize(sigmoid)

#The aggregate, expects the bias to be included in the weights (Bias Trick)
def z(w, x):
    return np.dot(np.transpose(w), x)

#Activation function, expects nparray
def a(z):
    return vector_sigmoud(z)

w = np.array([[0.6], [0.6], [-0.5]])
print(w)
x = np.array([1, 1, 1])

print(a(z(w, x)))

