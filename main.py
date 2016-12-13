from __future__ import division
from random import randint
from modules import data_manager, cluster, nearest_neighbors, feature_selector, feature_extractor

feature_extractor.test()
# (instances, paths, file_names, original_labels) = data_manager.read_data(
#     '/home/icekung/Documents/OverFeat/overfeat/samples/output/dogsColor')
#
# rootDir = '/home/icekung/Documents/OverFeat/overfeat/samples/orderDogsIn/'
# destDir = '/home/icekung/Documents/OverFeat/overfeat/samples/orderDogs'
#
# ordered_data = data_manager.sort_data(max(original_labels), None, original_labels, instances, file_names)
# instances = feature_selector.reduce_features(ordered_data, instances)
# ordered_data = data_manager.sort_data(max(original_labels), None, original_labels, instances, file_names)
#
# (training_set, testing_set) = data_manager.seperate_data(0.5, ordered_data, label=False)
#
# (cluster_num, labels) = cluster.group(training_set['instances'], training_set['file_names'], rootDir, destDir,
#                                       10)
# testing_labels = cluster.predict(testing_set['instances'])
# training_labels = cluster.predict(training_set['instances'])
# print 'accuracy: ' + str(cluster.evaluate(original_labels, training_set, testing_set))
#
# (training_set, testing_set) = data_manager.seperate_data(1, ordered_data, label=False)
# (cluster_num, labels) = cluster.group(training_set['instances'], training_set['file_names'], rootDir, destDir,
#                                       10)
# ordered_data = data_manager.sort_data(cluster_num, labels, training_set['original_labels'],
#                                       training_set['instances'], training_set['file_names'])
#
# ordered_labels = []
# for index in range(0, len(ordered_data)):
#     ordered_labels.append({k: v['original_label'] for k, v in enumerate(ordered_data[index])}.values())
# print 'purity: ' + str(cluster.purity(ordered_labels))

# rand_cluster = randint(1, len(ordered_data) - 1)
# random = ordered_data[rand_cluster][randint(1, len(ordered_data[rand_cluster]) - 1)]
# predicted_label = cluster.predict(random['instance'])[0]
#
# target_group = {k: v['instance'] for k, v in enumerate(ordered_data[predicted_label])}.values()
# nearest_neighbors.set(5, 2)
# nearest_neighbors.fit(target_group)
# indices = nearest_neighbors.neighbors([random['instance']])
#
# print 'chosen pic: ' + random['file_name']
# print 'predicted_label: ' + str(predicted_label)
#
# for index, value in enumerate(indices[0]):
#     print str(index + 1) + ': ' + ordered_data[predicted_label][value]['file_name']
