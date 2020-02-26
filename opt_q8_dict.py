import math
import numpy as np
import random
from scipy import optimize as opt

# ------------- PROCESSING -----------
f = open('movies_dataset/dataset.txt', 'r')
n_cats = 20
whole = f.read()
lines = whole.splitlines()

train, test = lines[0].split()
train = int(train)
test = int(test)

dictionary_matix = {}
for i in range(1, int(train) + 1):
    dictionary_matix[
        (
            int(lines[i].split()[0]),
            int(lines[i].split()[1])
        )
    ] = float(lines[i].split()[2])


max_tuple = max(dictionary_matix.keys())
maximum = max_tuple[0]
user_x_categories = np.ones((maximum+1, n_cats))
movies_x_categories = np.ones((n_cats, maximum + 1))

# ------------- DONE PROCESSING -----------


def loss(x):
    return (f(x)) ** 2


def f(x):
    return x[0] @ x[1] - x[2]


def gradient(x):
    eps = 1e-5

    grad = [np.zeros((n_cats,)) for _ in range(3)]
    x0 = np.copy(x[0])
    x1 = np.copy(x[1])

    for i in range(len(x[0])):
        x0_eps = np.copy(x0)
        x0_eps[i] = x0[i] + eps
        new_x = [x0_eps, x1, x[2]]
        grad[0][i] = (loss(new_x) - loss(x)) / eps
    for i in range(len(x[1])):
        x1_eps = np.copy(x1)
        x1_eps[i] = x1[i] + eps
        new_x = [x0, x1_eps, x[2]]
        grad[1][i] = (loss(new_x) - loss(x)) / eps
    return np.array(grad)


def batchsgd(batchsize=512):
    batch = []
    start = random.randrange(maximum)
    for i in range(start, start + batchsize):
        for j in range(start, start + batchsize):
            new_tuple = (i, j)
            if new_tuple in dictionary_matix:
                batch.append(new_tuple)
    return batch


def sgd(alpha, indices=batchsgd()):
    for i in range(len(indices)):
        x = np.array((
            user_x_categories[indices[i][0]],
            movies_x_categories[:, indices[i][1]],
            dictionary_matix[indices[i]]
        ))

        grad = gradient(x)

        user_x_categories[indices[i][0]] -= alpha * grad[0]
        movies_x_categories[:, indices[i][1]] -= alpha * grad[1]


def train_(maxiter=9000):
    training = []
    alpha = .01
    constant = .001
    for i in range(maxiter):
        alpha *= (math.exp(-constant))
        sgd(alpha, batchsgd())
        total_loss = 0
        random_value = random.randrange(len(dictionary_matix))
        if i % random_value == 0:
            for k, v in dictionary_matix:
                x = np.array(
                    [
                        user_x_categories[k],
                        movies_x_categories[:, k],
                        v
                    ]
                )
                total_loss += loss(x)
            print('loss so far__', total_loss)
        # training.append((i, loss_))
    return training


predicted_scores = train_()
# print(predicted_scores)

def predict(arr):
    arr = np.array(arr)
    return arr[0] @ arr[1]


testing = []
for i in range(test):
    user_idx = int(lines[train+i].split()[0])
    movie_idx = int(lines[train + i].split()[0])
    testing.append(user_x_categories[user_idx]
                   @movies_x_categories[:, movie_idx])
print(testing)


with open('ans.txt', 'w') as f:
    for item in testing:
        f.write("%s\n" % item)
