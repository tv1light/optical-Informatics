import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

def draw(title, arr_x, arr_y):
    """
    Функция для постройки графика

    :param title:
    :param arr_x:
    :param arr_y:
    :return:
    """

    plt.plot(arr_x, arr_y, 'm')
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    a, b = 0, 5
    p, q = 0, 5
    m, n = 1000, 1000
    alpha = 1

    #Исходный сигнал
    betta = 1 / 10
    func = lambda x: np.exp(complex(0, 1) * x * betta)
    # arr_x = np.arange(a, b)
    # arr_y = func(arr_x)
    # draw("График амплитуды входного сигнала", arr_x, np.abs(arr_y))
    # draw("График фазы входного сигнала", arr_x, np.angle(arr_y))

    #Выходной сигнал
    hx = (b - a) / m
    hxi = (q - p) / 1000
    x = lambda k: a + k * hx
    xi = lambda l: p + l * hxi
    F = lambda l: (hx * sum([jv(2, x(k)*xi(l)*alpha) * func(x(k))
                             for k in range(0, n)]))
    Fl = [F(l) for l in range(0, n+1)]
    draw("График амплитуды выходного сигнала", np.arange(p, q, (q - p) / 1001), np.abs(Fl))
    draw("График фазы выходного сигнала", np.arange(p, q, (q - p) / 1001), np.angle(Fl))
