from __future__ import division
import shutil
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, MiniBatchKMeans
import numpy as np
import os
import nearest_neighbors
from collections import Counter
import shutil
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

divider = None

def group(instances, file_names, rootDir, destDir, prefer_cluster=None):
    global divider
    data = scale(instances)
    cluster = prefer_cluster if prefer_cluster is not None else 9
    pca = PCA(n_components=cluster).fit(data)
    #divider = KMeans(init=pca.components_, n_clusters=cluster, n_init=1)
    # divider = MiniBatchKMeans(cluster, random_state=0, batch_size=5)
    divider = AffinityPropagation()
    labels = divider.fit_predict(data)
    # labels = divider.fit_predict(instances)
    shutil.rmtree(destDir + '/')
    for index in range(0, len(instances)):
        if not os.path.exists(destDir + '/' + ''.join(str(labels[index]))):
            os.makedirs(destDir + '/' + ''.join(str(labels[index])))
        shutil.copy(rootDir + file_names[index], destDir + '/' + ''.join(str(labels[index])))
    return (max(labels) + 1, labels)

def predict(instances):
    labels = []
    global divider
    nearest_neighbors.set(1, 10)
    nearest_neighbors.fit(divider.cluster_centers_)
    for i, label in enumerate(nearest_neighbors.neighbors(instances)):
        labels.append(label[0])
    return labels

def evaluate(original_labels, training_set, testing_set):
    training_labels = predict(training_set['instances'])
    testing_labels = predict(testing_set['instances'])
    training_start = 0
    training_end = 0
    testing_start = 0
    testing_end = 0
    correct = 0
    incorrect = 0
    testing_labels_size = len(testing_set['original_labels'])
    training_labels_size = len(training_set['original_labels'])
    for index in range(1, max(original_labels) + 1):
        while training_end < training_labels_size and training_set['original_labels'][training_end] == index :
            training_end += 1
        while testing_end < testing_labels_size and testing_set['original_labels'][testing_end] == index:
            testing_end += 1
        mode = Counter(training_labels[training_start:training_end]).most_common(1)[0][0]
        training_start = training_end
        for i in range(testing_start, testing_end):
            if testing_labels[i] == mode:
                correct += 1
            else:
                incorrect += 1
        testing_start = testing_end

    return (correct/(correct + incorrect)) * 100

def purity(sorted_data):
    items = 0
    accumulated_cluster = 0
    for index in range(0, len(sorted_data)):
        items += len(sorted_data[index])
        accumulated_cluster += Counter(sorted_data[index]).most_common(1)[0][1] if len(sorted_data[index]) > 0 else 0

    return (1/items) * accumulated_cluster