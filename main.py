import numpy as np
import matplotlib.pyplot as plt


def function(x):
    """
    Расчет значения функции входного сигнала
    f(x) = exp(ix/10)

    :param x:
    :return: f(x) = exp(ix/10)
    """
    return np.exp(complex(0, 1)*x/10)


def draw(title, x, y):
    """
    Функция для постройки графика

    :param title:
    :param x:
    :param y:
    :return:
    """
    pass


def kernel(alpha, ksi, x):
    """
    Расчет ядра

    :param alpha:
    :param ksi:
    :param x:
    """
    pass


if __name__ == '__main__':
    a, b = 0, 5
    p, q = 0, 5
    m, n = 1000, 1000
    alpha = 1
