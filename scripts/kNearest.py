import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn(data, query, k):
    distances = []
    for index, row in data.iterrows():
        distance = euclidean_distance(row[1:-1].values, query)
        distances.append((distance, row['label']))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return neighbors

def predict(data, query, k):
    neighbors = knn(data, query, k)
    classes = [neighbor[1] for neighbor in neighbors]
    return max(set(classes), key=classes.count) 