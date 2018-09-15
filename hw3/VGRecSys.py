import pandas as pd
from math import sqrt
import csv

USER_ID = 'U868476845'

# Load the dataset
header = ['user_id', 'item_id', 'rating']
df = pd.read_csv('rating.txt', sep=' ', names=header)

print('\n')
mean = (df.groupby('user_id')['rating'].mean()['U868476845'])
print("The average rating of user U868476845 is " + str(mean))

dataset = {}
with open("rating.txt", 'r') as data_file:
    data = csv.DictReader(data_file, delimiter=" ", fieldnames=header)
    for row in data:
        item = dataset.get(row["user_id"], dict())
        item[row["item_id"]] = float(row["rating"])

        dataset[row["user_id"]] = item

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

    if len(array_x) <= 1:
        return 0

    return cosine_similarity(array_x, array_y)

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
print("Users most similar to U868476845 are (User with less than 1 common items have been ignored):")
print(most_similar_users(USER_ID, 10))

print('\n')
print("Ranking of recommendation to U868476845 are:")
print(user_reommendations(USER_ID))
