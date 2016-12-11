import os

def read_data(rootDir):
    instances = []
    paths = []
    file_names = []
    original_labels = []
    index = -1
    dirs = None
    for root, dir, files in os.walk(rootDir):
        if index == -1: dirs = dir
        for file in files:
            original_labels.append(int(dirs[index]))
            paths.append(root + '/' + file)
            file_names.append(root[-1] + '/' + file[:-9])
            sample = open(root + '/' + file, 'r')
            raw = sample.read()[9:-2].split(' ')
            features = [float(feature) for feature in raw][0:4096]
            instances.append(features)
            sample.close()
        index += 1
    return (instances, paths, file_names, original_labels)

def sort_data(cluster, labels, original_labels, instances, file_names):
    ordered_data = []
    available_labels = labels if labels is not None else original_labels
    for i in range(0, cluster + 1):
        ordered_data.append([])

    for index, label in enumerate(available_labels):
        temp = {}
        temp['instance'] = instances[index]
        temp['file_name'] = file_names[index]
        temp['original_label'] = original_labels[index]
        if labels is not None: temp['label'] = label
        ordered_data[label].append(temp)
    return ordered_data

def seperate_data(ratio, ordered_data, file_name = True, original_label = True, label = True):
    testing_set = {'instances': [], 'labels': [], 'file_names': [], 'original_labels': []}
    training_set = {'instances': [], 'labels': [], 'file_names': [], 'original_labels': []}

    for i in range(0, len(ordered_data)):
        for j in range(0, len(ordered_data[i])):
            if (j < (len(ordered_data[i]) * ratio)):
                training_set['instances'].append(ordered_data[i][j]['instance'])
                if label == True: training_set['labels'].append(i)
                if file_name == True: training_set['file_names'].append(ordered_data[i][j]['file_name'])
                if original_label == True: training_set['original_labels'].append(ordered_data[i][j]['original_label'])
            else:
                testing_set['instances'].append(ordered_data[i][j]['instance'])
                if label == True: testing_set['labels'].append(i)
                if file_name == True: testing_set['file_names'].append(ordered_data[i][j]['file_name'])
                if original_label == True: testing_set['original_labels'].append(ordered_data[i][j]['original_label'])

    return (training_set, testing_set)