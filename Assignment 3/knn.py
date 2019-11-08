import pandas as pd
import math

'''
element = [x1, x2, x3, x4, y]
'''
def load_data(path):
    return pd.read_csv(path) 

def euclidian_distance(e1, e2):
    return math.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2 + (e1[2] - e2[2])**2)

def knn(neighbours, value, k = 10):

    #Initializes the K nearest storage
    classes = [0 for i in range(k)]
    distances = [float('inf') for i in range(k)]

    #Finds the K nearest
    for n in neighbours.iloc:
        distance = euclidian_distance(n, value)
        if(distance < max(distances)):
            index = index(max(distances))
            classes[index] = n['y']
            distances[index] = distance
    
    #Finds the most common among k nearest
    types = {}
    for c in classes:
        if(not types[c]):
            types[c] = 1
        else:
            types[c] = types[c] + 1
    
    type = types[types.keys()[0]]
    for k in types.keys():
        if(types[k] > types[type]):
            type = types[k]

    #Returns the most common
    return type


def main():
    data = load_data("./data/knn_classification.csv")
    d1 = data[0]
    knn(data, d1)
    

main()