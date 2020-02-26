import numpy as np
import random
# ------------- PROCESSING -----------
f = open('movies_dataset/dataset_302812_7.txt', 'r')
whole = f.read()
lines = whole.split(sep='\n')
imp = lines[0].split(sep=',')

# print('Important data ', *imp)

train, test = imp[0].split(sep=' ')
train = int(train)
test = int(test)

user_id, movie_id, score_id = [], [], []
for i in range(1, int(train)+1):
    user_id.append(int(lines[i].split(sep=' ')[0]))
    movie_id.append(int(lines[i].split(sep=' ')[1]))
    score_id.append(float(lines[i].split(sep=' ')[2]))

train_matrix = np.array([user_id, movie_id])

users = np.ones((int(max(user_id))+1, 15))
movies = np.ones((15, int(max(movie_id)) + 1))

# ------------- DONE PROCESSING -----------


def loss(index):
    return (np.dot(movies[:, movie_id[index]], users[user_id[index]]) - score_id[index]) ** 2


def prediction(test_users, test_movies):
    return np.dot(test_users.T, test_movies)


def sgd(train_matrix, score_id, latent_features=15, alpha=.1, tol=.001, maxiters=100):
    u, m = train_matrix.shape
    train_err = []
    validation_err = []
    for it in range(maxiters):
        for i, j in zip(train_matrix):
            error = train_matrix[i, j] - prediction(users[i], movies[:, j])
            users[i] += alpha * (error * movies[:, j] - tol * users[i])
            movies[:, j] += alpha * (error * users[i] - tol * movies[:, j])
        train_loss = loss
