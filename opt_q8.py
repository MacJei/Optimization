import numpy as np
import random
from scipy import optimize as opt

# ------------- PROCESSING -----------
f = open('movies_dataset/dataset_302812_8.txt', 'r')
n_cats = 20
whole = f.read()
lines = whole.splitlines()

train, test = lines[0].split()
train = int(train)
test = int(test)

known_users, movies_users, known_scores = [], [], []
for i in range(1, int(train)+1):
    known_users.append(int(lines[i].split()[0]))
    movies_users.append(int(lines[i].split()[1]))
    known_scores.append(float(lines[i].split()[2]))


user_x_categories = np.ones((max(known_users)+1, n_cats))
movies_x_categories = np.ones((n_cats, max(movies_users)+1))

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


def get_alpha(x):
    for _ in range(100):
        alpha, _, _, _, _, _ = opt.line_search(
            f(x),
            gradient(x),
            x,
            -gradient(x),
        )
    return alpha


def sgd(indices):
    # random.seed(2020)
    range_loss = []
    for i in range(len(indices)):
        x = np.array([user_x_categories[known_users[indices[i]]],
                      movies_x_categories[:, movies_users[indices[i]]],
                      known_scores[indices[i]]])
        grad = gradient(x)
        range_loss.append(loss(x))
        alpha = get_alpha(x)

        print('alpha of__', alpha)

        user_x_categories[known_users[indices[i]]] -= alpha * grad[0]
        movies_x_categories[:, movies_users[indices[i]]] -= alpha * grad[1]
        if i % len(indices) == 0:
            total_loss = 0
            for ind in range(len(known_scores)):
                x = np.array([user_x_categories[known_users[ind]],
                              movies_x_categories[:, movies_users[ind]],
                              known_scores[ind]])
                print(x)
                # Modify alpha inside here
                total_loss += loss(x)

            print('loss so far__', total_loss)  # Time consuming
    return alpha


def batchsgd(batchsize=512):
    batch = []
    for i in range(batchsize):
        # Choose a score value and all scores in that row and column
        random_score = random.randrange(max(known_users))
        batch.append(random_score)
    return batch


def train_(maxiter=50):
    training = []
    loss_ = 0
    for i in range(maxiter):
        sgd(batchsgd())
        training.append((i, loss_))
    return training


print(train_())
""" testing = []
for i in range(test):
    user_index = int(lines[train+i].split()[0])
    movie_index = int(lines[train + i].split()[0])
    testing.append(user_x_categories[user_index]
                   @ movies_x_categories[:, movie_index])
# print it to a file. If loss doesn't go low enough we can decrease alpha at each iteration
# print(testing)


with open('ans.txt', 'w') as f:
    for item in testing:
        f.write("%s\n" % item)
 """
