from sklearn import linear_model
import math

def match_sample(ordered_data):
    internal_matching = []
    cross_matching = []

    groups_size = len(ordered_data)

    for i in range(1, groups_size):
        for j in range(1, groups_size):
            for k in range(0, len(ordered_data[i])):
                if i == j:
                    for l in range(0, len(ordered_data[j])):
                        internal_matching.append(ordered_data[i][k]['instance'] + ordered_data[j][l]['instance'])
                else:
                    for l in range(0, len(ordered_data[j])):
                        cross_matching.append(ordered_data[i][k]['instance'] + ordered_data[j][l]['instance'])

    return (internal_matching, cross_matching)

def get_coeff(data, labels):
    model = linear_model.LinearRegression()
    model.fit(data, labels)
    return model.coef_

def get_mask(coef, filters):
    for index, weight in enumerate(coef):
        if math.fabs(weight) < filters:
            coef[index] = 0
    return coef

def reduce_features(ordered_data, instances):
    (intern_match, cross_match) = match_sample(ordered_data)
    cross_match = cross_match[::2]
    labels = [1] * len(intern_match) + [0] * len(cross_match)
    coef = get_coeff(intern_match + cross_match, labels)[0:4096]
    filters = 13000000000
    mask = get_mask(coef, filters)

    for index in range(0, len(instances)):
        instances[index] = [weight for index2, weight in enumerate(instances[index]) if mask[index2] != 0]

    return instances