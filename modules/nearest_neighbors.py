from sklearn.neighbors import NearestNeighbors

nn = None
neighbor_num = None
radius = None

def set(neighbor_num, radius):
    global nn
    globals()['neighbor_num'] = neighbor_num
    globals()['radius'] = radius
    nn = NearestNeighbors(neighbor_num, radius)

def fit(data):
    global nn, neighbor_num, radius
    if len(data) < neighbor_num:
        nn = NearestNeighbors(len(data), radius) # if number of neighbor is less than defined
    nn.fit(data)

def neighbors(data):
    global nn
    hello = nn
    return hello.kneighbors(data, return_distance=False)