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
    distance = 0
    for i in range(len(e1) - 1):
        distance += (e1[i] - e2[i])**2
    return math.sqrt(distance)

def average(nearest):
    aggregate = 0
    for n in nearest:
        aggregate += n[3]

    return aggregate/len(nearest)

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
            classes[i] = n[len(n) - 1]
            distances[i] = distance
            _neigbours[i] = n
    
    return _neigbours

def vote(k_nearest):
    #Finds the most common among k nearest
    types = {}

    classes = []
    for k in k_nearest:
        classes.append(k[len(k) - 1])

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

def regression(nearest):
    return average(nearest)


def main():
    classification_data = load_data("./data/knn_classification.csv")
    regression_data = load_data("./data/knn_regression.csv")

    # #Task 1
    print("\n\nFetching element 124 and performing classification...")
    d1 = classification_data[123]
    neighbours = knn(classification_data, d1)
    print("10 nearest neighbours in random order:")
    for n in neighbours:
        print(n)
    
    print("Classification by vote shows {}".format(vote(neighbours)))

    #Task 2
    print("\n\nFetching element 124 and performing regression...")
    d1 = regression_data[123]
    neighbours = knn(regression_data, d1)
    print("10 nearest neighbours in random order:")
    for n in neighbours:
        print(n)
    print("Regression by average shows: {}".format(regression(neighbours)))
    
main()