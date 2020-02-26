import numpy as np
import random
# ------------- PROCESSING -----------
f = open('movies_dataset/dataset_302812_7.txt', 'r')
n_cats = 15
whole = f.read()
lines = whole.splitlines()
# print('Important data ', *imp)

train, test = lines[0].split()
train = int(train)
test = int(test)

user_id, movie_id, scores = [], [], []
for i in range(1, int(train)+1):
    user_id.append(int(lines[i].split()[0]))
    movie_id.append(int(lines[i].split()[1]))
    scores.append(float(lines[i].split()[2]))


# complete_matrix = np.array([user_id, movie_id, score_id])

users = np.ones((max(user_id)+1, n_cats))
movies = np.ones((n_cats, max(movie_id)+1))

# ------------- DONE PROCESSING -----------


def loss(x):
    return (f(x)) ** 2


def f(x):
    return x[0] @ x[1] - x[2]


def gradient(x):
    eps = 1e-10

    grad = [np.zeros((n_cats,)) for _ in range(3)]
    x0 = np.copy(x[0])
    x1 = np.copy(x[1])

    for i in range(len(x[0])):
        x0_eps = np.copy(x0)
        # x1_eps = np.copy(x1)
        x0_eps[i] = x0[i] + eps
        new_x = [x0_eps, x1, x[2]]
        grad[0][i] = (loss(new_x) - loss(x)) / eps
    for i in range(len(x[1])):
        # x0_eps = np.copy(x0)
        x1_eps = np.copy(x1)
        x1_eps[i] = x1[i] + eps
        new_x = [x0, x1_eps, x[2]]
        grad[1][i] = (loss(new_x) - loss(x)) / eps
    return grad


def sgd(alpha=.01):
    # random.seed(2020)
    range_loss = []
    for i in range(len(scores)):
        random_ind = random.randrange(len(scores))
        u = user_id[random_ind]
        x = np.array([users[u],
                      movies[:, movie_id[random_ind]],
                      scores[random_ind]])
        grad = gradient(x)
        range_loss.append(loss(x))

        users[user_id[random_ind]] -= alpha * grad[0]
        movies[:, movie_id[random_ind]] -= alpha * grad[1]
        if i % 900 == 0:
            total_loss = 0
            for ind in range(len(scores)):
                u = user_id[ind]
                x = np.array([users[u],
                              movies[:, movie_id[ind]],
                              scores[ind]])
                total_loss += loss(x)

            print('loss so far__', total_loss)  # Time consuming


def train_(maxiter=10):
    training = []
    loss_ = 0
    for i in range(maxiter):
        sgd()
        training.append((i, loss_))
    # count = 0
    # for i in range(random.randint(0, n_cats)):
    #     while count < random.randint(0, np.mean(user_id)):
    #         random_choice = random.randint(0, max(user_id))
    #         if scores[random_choice] == np.dot(users[random_choice], movies[:, random_choice]):
    #             count += 1
    return training


# def predict()
predicted_scores = train_()
print(predicted_scores)


def predict(arr):
    arr = np.array(arr)
    return arr[0] @ arr[1]


testing = []
for i in range(test):
    user_idx = int(lines[train+i].split()[0])
    movie_idx = int(lines[train + i].split()[0])
    testing.append(users[user_idx]@movies[:, movie_idx])
# print it to a file. If loss doesn't go low enough we can decrease alpha at each iteration
print(testing)


with open('ans.txt', 'w') as f:
    for item in testing:
        f.write("%s\n" % item)
