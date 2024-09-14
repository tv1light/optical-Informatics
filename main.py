import numpy as np
import matplotlib.pyplot as plt


def function(x):
    """
    Расчет значения функции входного сигнала
    f(x) = exp(ix/10)

    :param x:
    :return: f(x) = exp(ix/10)
    """
    return np.exp(complex(0, 1) * x / 10)


def draw(title, arr_x, arr_y):
    """
    Функция для постройки графика

    :param title:
    :param arr_x:
    :param arr_y:
    :return:
    """
    plt.plot(arr_x, arr_y, 'ro')
    plt.title(title)
    plt.grid(True)
    plt.show()


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

    #Task 1
    figure = plt.figure()
    arr_x = np.linspace(a, b, n)
    arr_y = function(arr_x)
    draw("График амплитуды входного сигнала", arr_x, np.abs(arr_y))
    draw("График фазы входного сигнала", arr_x, np.angle(arr_y))
