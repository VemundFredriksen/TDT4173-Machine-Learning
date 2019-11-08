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

#Simple euclidian distnace function
def euclidian_distance(e1, e2):
    return math.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2 + (e1[2] - e2[2])**2 + (e1[3] - e2[3])**2)

#KNN With Voting
def knn(neighbours, value, k = 10):

    #Initializes the K nearest storage
    classes = [0 for i in range(k)]
    distances = [float('inf') for i in range(k)]
    _neigbours = [None for i in range(k)]

    #Finds the K nearest
    for n in neighbours:
        distance = euclidian_distance(n, value)
        if(distance < max(distances)):
            i = distances.index(max(distances))
            classes[i] = n[4]
            distances[i] = distance
            _neigbours[i] = n
    
    type = vote(classes, distance, _neigbours)
    
    #Returns the most common
    return type, _neigbours

def vote(classes, distances, k_nearest):
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
            
    return type

def main():
    data = load_data("./data/knn_classification.csv")

    #Task 1
    print("\n\nFetching element 124 and performing classification...")
    d1 = data[123]
    (type, neighbours) = knn(data, d1)
    print("Element {} is predicted to be of class {}".format(124, type))
    print("10 nearest neighbours in random order:")
    for n in neighbours:
        print(n)
    
main()