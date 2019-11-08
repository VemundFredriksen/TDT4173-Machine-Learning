import pandas as pd
import math

'''
element = [x1, x2, x3, x4, y]
'''
def load_data(filename):
    with open(filename, "r") as f:
        s = f.read()
        s = s.split("\n")
        k = []
        for i in range(1, len(s)):
            t = []
            l = s[i].split(",")
            for z in l:
                t.append(float(z))
            k.append(t)
            
        return k
 

def euclidian_distance(e1, e2):
    return math.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2 + (e1[2] - e2[2])**2)

def knn(neighbours, value, k = 10):

    #Initializes the K nearest storage
    classes = [0 for i in range(k)]
    distances = [float('inf') for i in range(k)]

    #Finds the K nearest
    for n in neighbours:
        distance = euclidian_distance(n, value)
        if(distance < max(distances)):
            i = distances.index(max(distances))
            classes[i] = n[4]
            distances[i] = distance
    
    #Finds the most common among k nearest
    types = {}
    for c in classes:
        if(not c in types.keys()):
            types[c] = 1
        else:
            types[c] = types[c] + 1
    
    type = classes[0]
    for k in types.keys():
        if(types[k] > types[type]):
            type = k

    #Returns the most common
    return type


def main():
    data = load_data("./data/knn_classification.csv")
    d1 = data[124]
    data[124] = [20, 20, 20, 20, 20]
    print(knn(data, d1))
    

main()