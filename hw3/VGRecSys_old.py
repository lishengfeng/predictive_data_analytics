import pandas as pd
from math import sqrt
import numpy as np
import csv
# from sklearn.metrics.pairwise import cosine_similarity

USER_ID = 'U868476845'

# Load the dataset
header = ['user_id', 'item_id', 'rating']
df = pd.read_csv('rating.txt', sep=' ', names=header)


# n_users = df.user_id.unique().shape[0]
# n_items = df.item_id.unique().shape[0]

print('\n')
mean = (df.groupby('user_id')['rating'].mean()['U868476845'])
print("The average rating of user U868476845 is " + str(mean))

# item = dataset.loc[dataset['user_id'] == 'U103689342', 'item_id']
# print(item)


# dataset = df.groupby('user_id')['rating'].apply(list)
# dataset = df.groupby('user_id')['item_id'].apply(lambda x: "%s" % ', '.join(x))
# print(dataset)
# item = dataset['U103689342']
# print(item)

dataset = {}
with open("rating.txt", 'r') as data_file:
    data = csv.DictReader(data_file, delimiter=" ", fieldnames=header)
    for row in data:
        item = dataset.get(row["user_id"], dict())
        item[row["item_id"]] = float(row["rating"])

        dataset[row["user_id"]] = item

# print(dataset)

def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)

def similarity_score(person1, person2):
    # Returns ratio Euclidean distance score of person1 and person2

    array_x = []
    array_y = []

    for item in dataset[person1]:
        if item in dataset[person2]:
            array_x.append(dataset[person1][item])
            array_y.append(dataset[person2][item])

    if len(array_x) <= 3:
        return 0

    return cosine_similarity(array_x, array_y)
    # return cosine_similarity(np.reshape(array_x, (-1, 1)), np.reshape(array_y, (-1, 1)))[0][0]

# def similarity_score(person1, person2):
#     # Returns ratio Euclidean distance score of person1 and person2
#
#     both_viewed = {}  # To get both rated items by person1 and person2
#
#     for item in dataset[person1]:
#         if item in dataset[person2]:
#             both_viewed[item] = 1
#
#         # Conditions to check they both have an common rating items
#         if len(both_viewed) == 0:
#             return 0
#
#         # Finding Euclidean distance
#         sum_of_eclidean_distance = []
#
#         for item in dataset[person1]:
#             if item in dataset[person2]:
#                 sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item], 2))
#         sum_of_eclidean_distance = sum(sum_of_eclidean_distance)
#
#         return 1 / (1 + sqrt(sum_of_eclidean_distance))


# def pearson_correlation(person1, person2):
#     # To get both rated items
#     both_rated = {}
#     for item in dataset[person1]:
#         if item in dataset[person2]:
#             both_rated[item] = 1
#
#     number_of_ratings = len(both_rated)
#
#     # Checking for number of ratings in common
#     if number_of_ratings == 0:
#         return 0
#
#     # Add up all the preferences of each user
#     person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
#     person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])
#
#     # Sum up the squares of preferences of each user
#     person1_square_preferences_sum = sum([pow(dataset[person1][item], 2) for item in both_rated])
#     person2_square_preferences_sum = sum([pow(dataset[person2][item], 2) for item in both_rated])
#
#     # Sum up the product value of both preferences for each item
#     product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])
#
#     # Calculate the pearson score
#     numerator_value = product_sum_of_both_users - (
#     person1_preferences_sum * person2_preferences_sum / number_of_ratings)
#     denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum, 2) / number_of_ratings) * (
#     person2_square_preferences_sum - pow(person2_preferences_sum, 2) / number_of_ratings))
#     if denominator_value == 0:
#         return 0
#     else:
#         r = numerator_value / denominator_value
#         return r

def most_similar_users(person, number_of_users):
    # returns the number_of_users (similar persons) for a given specific person.
    scores = [(similarity_score(person, other_person), other_person) for other_person in dataset.keys() if
              other_person != person]

    # Sort the similar persons so that highest scores person will appear at the first
    scores.sort()
    scores.reverse()
    return scores[0:number_of_users]

def user_reommendations(person):

    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list =[]
    for other in dataset.keys():
        # don't compare me to myself
        if other == person:
            continue
        sim = similarity_score(person,other)

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in dataset[other]:

            # only score movies i haven't seen yet
            if item not in dataset[person] or dataset[person][item] == 0:

            # Similrity * score
                totals.setdefault(item,0)
                totals[item] += dataset[other][item]* sim
                # sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+= sim

        # Create the normalized list

    rankings = [(total/simSums[item],item) for item,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendataions_list = [recommend_item for score,recommend_item in rankings]
    return recommendataions_list

print('\n')
print("Users most similar to U868476845 are (User with less than 2 common items have been ignored):")
print(most_similar_users(USER_ID, 10))

print('\n')
print("Ranking of recommendation to U868476845 are:")
print(user_reommendations(USER_ID))
