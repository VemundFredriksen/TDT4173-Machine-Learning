import pandas as pd
import math

def load_data(path):
    return pd.read_csv(path) 

def euclidian_distance(e1, e2):
    return math.sqrt((e1['x1'] - e2['x1'])**2 + (e1['x2'] - e2['x2'])**2 + (e1['x3'] - e2['x3'])**2)

def knn(neighbours, value, k = 10):
    classes = [0 for i in range(k)]
    distances = [float('inf') for i in range(k)]

    for n in neighbours:
        distance = euclidian_distance(n, value)
        if(distance < max(distances)):
            index = index(max(distances))
            classes[index] = n['y']
            distances[index] = distance
    
    types = {}
    for c in classes:
        if(not types[c]):
            types[c] = 1
        else:
            types[c] = types[c] + 1
    
    type = types[types.keys()[0]]
    for k in types.keys():
        print(k)


def main():
    #Load data
    data = load_data("./data/knn_classification.csv")
    d1 = data.iloc[1]
    d2 = data.iloc[2]
    print(data.iloc[124])
    knn(data, d1)
    

main()