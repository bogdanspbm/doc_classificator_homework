import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns


def task_1():
    a_vec = np.arange(1, 21)
    b_vec = 2 ** a_vec
    c_vec = b_vec / a_vec
    print(c_vec)


def task_2():
    a_vec = np.arange(0, 6)
    b_vec = 0.1 ** (3 * a_vec)
    c_vec = 0.2 ** (4 * a_vec)
    result = np.dot(b_vec, c_vec)

    print(result)


def task_3():
    a_mat = np.zeros((8, 8))
    a_mat[1::2, ::2] = 1
    a_mat[::2, 1::2] = 1

    print(a_mat)


def task_4():
    X = np.random.random((5, 2))
    x_dec = X[:, 0]
    y_dec = X[:, 1]

    fi = np.arctan(y_dec / x_dec)
    r = x_dec / np.cos(fi)

    X = np.reshape(np.concatenate((fi, r)), (5, 2), order='F')

    print(X)


def task_5():
    X = np.random.random((10, 10))
    # X = np.random.randint(0, 10, size=(10, 10))
    x_vec, y_vec = np.where(X == np.max(X))

    result = np.reshape(np.concatenate((x_vec, y_vec)), (len(x_vec), 2), order='F')

    print(result)


def task_6():
    X = np.random.random((10, 2))
    y = np.random.random((1, 2))

    x_deformed = X - y

    distance_vec = np.sqrt(x_deformed[:, 0] ** 2 + x_deformed[:, 1] ** 2)

    index = np.where(distance_vec == np.min(distance_vec))

    x_min = X[index[0][0]]

    print(x_min)


def task_7():
    x_vec = np.arange(-3, 4)
    # x_vec = np.linspace(-3, 3, 10)

    x_a = x_vec[np.where(x_vec < 0)]
    x_b = x_vec[np.where(x_vec <= 2)][len(x_a):]
    x_c = x_vec[np.where(x_vec > 2)]

    a_res = x_a ** 2 + 2 * x_a + 6
    b_res = x_b + 6
    c_res = x_c ** 2 + 4 * x_c - 4

    result = np.concatenate((a_res, b_res, c_res))

    print(result)


def task_8():
    X = np.random.random((10, 10))
    a_vec = [np.mean(X, axis=1)]
    a_mat = np.reshape(np.tile(a_vec, 10), (10, 10)).transpose()

    result = X - a_mat

    print(result)
    print(np.mean(result, axis=1))


def task_9():
    X = np.random.normal(loc=5, scale=2., size=1000)
    s_m = np.sum(X) / len(X)
    s_v = np.sqrt(np.sum((X - s_m) ** 2) / len(X))

    print(s_m)
    print(s_v)


def task_10():
    x_vec = np.arange(0, 5)
    x_long = np.tile(x_vec, 5)

    x_mat = np.reshape(x_long, (5, 5))
    x_mat_t = x_mat.transpose()

    result = (x_mat + x_mat_t) % 5

    print(result)


def sample(x, c):
    assert len(x) > 0

    '''
    Метод генерирует c - случайных чисел
    и считает для каждого числа, сколько максимально
    первых элементов входного вектора можно сложить,
    чтобы сумма была не больше случайного числа
    '''

    n = len(x)

    s = np.sum(x)
    random_vec = s * np.random.random(c)
    offset_vec = np.cumsum(x).astype('float64')

    offset_mat = np.reshape(np.tile(offset_vec, c), (c, n))
    random_mat = np.reshape(np.tile(random_vec, n), (n, c)).transpose()

    offset_mat -= random_mat
    result = np.sum(offset_mat < 0, axis=1)
    return result


print(sample([50, 3, 1, 7, 20], 7))
