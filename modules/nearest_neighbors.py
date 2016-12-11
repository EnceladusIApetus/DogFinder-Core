from sklearn.neighbors import NearestNeighbors

nn = None
neighbor = None
radius = None

def set(neighbor, radius):
    global nn
    globals()['neighbor'] = neighbor
    globals()['radius'] = radius
    nn = NearestNeighbors(neighbor, radius)

def fit(data):
    global nn, neighbor, radius
    if len(data) < neighbor:
        nn = nn = NearestNeighbors(len(data), radius)
    nn.fit(data)

def neighbors(data):
    global nn
    hello = nn
    return hello.kneighbors(data, return_distance=False)