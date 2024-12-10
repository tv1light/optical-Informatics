import numpy as np
import matplotlib.pyplot as plt

# Параметры
a1 = -5
a2 = 5
N = 1 << 10

# Функция, согласно 3 варианту
light = lambda x: np.exp(2 * np.pi * 1j * x) + np.exp(-5 * np.pi * 1j * x)
x = np.linspace(a1, a2, N, endpoint=False)
f = light(x)

M = 1 << 16
b2 = (N * N) / (4 * a2 * M)
b1 = -b2
u = np.linspace(b1, b2, N, endpoint=False)
hx = (a2 - a1) / N

# БПФ
def bpf(f, N, M, hx):
    k = (M - N) // 2
    F = np.insert(f, 0, np.zeros(k))
    F = np.append(F, np.zeros(k))

    F = np.fft.fftshift(F)
    F = np.fft.fft(F) * hx
    F = np.fft.fftshift(F)

    F = F[len(F) // 2 - N // 2: len(F) // 2 + N // 2]
    return F

F = bpf(f, N, M, hx)

# Аналитическое преобразование Фурье для 3 варианта
analytical_F = lambda u: (np.sinc(u - 1) + np.sinc(u + 5)) * 2 * np.pi
F_analytical = analytical_F(u)

# Стандартное численное интегрирование
def numerical_integration(f, x, u):
    h_x = x[1] - x[0]
    result = np.zeros_like(u, dtype=complex)
    for i, ui in enumerate(u):
        result[i] = np.sum(f * np.exp(-2j * np.pi * ui * x)) * h_x
    return result

F_numerical = numerical_integration(f, x, u)

# Графики исходного сигнала, согласно варианту
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.plot(x, np.abs(f))
plt.title('Амплитуда светового поля, согласно варианту')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(x, np.angle(f))
plt.title('Фаза светового поля, согласно варианту')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(u, np.abs(F), label='БПФ', c='c', lw=2)
plt.plot(u, np.abs(F_analytical), label='Аналитическое', c='k', lw=1)
plt.plot(u, np.abs(F_numerical), label='Численное', c='r', lw=1)
plt.title('Амплитуда светового поля')
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(u, np.angle(F), label='БПФ', c='c', lw=2)
plt.plot(u, np.angle(F_analytical), label='Аналитическое', c='k', lw=1)
plt.plot(u, np.angle(F_numerical), label='Численное', c='r', lw=1)
plt.title('Фаза светового поля')
plt.legend()
plt.grid()

plt.show()

# Двумерное преобразование Фурье
X, Y = np.meshgrid(x, x)
light_2d = lambda x, y: light(x) * light(y)
f_2d = light_2d(X, Y)

# Двумерное БПФ
def bpf_2d(Z, N, M, hx):
    for i in range(Z.shape[0]):
        Z[i, :] = bpf(Z[i, :], N, M, hx)
    for j in range(Z.shape[1]):
        Z[:, j] = bpf(Z[:, j], N, M, hx)
    return Z

F_2d = bpf_2d(f_2d.astype(np.complex128), N, M, hx)

# Аналитическое двумерное преобразование
F_2d_analytical = np.outer(F_analytical, F_analytical)

# Графики двумерного преобразования
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].imshow(np.abs(f_2d), extent=[a1, a2, a1, a2], cmap='hot')
axes[0, 0].set_title('Амплитуда исходного поля')
axes[0, 1].imshow(np.angle(f_2d), extent=[a1, a2, a1, a2], cmap='hot')
axes[0, 1].set_title('Фаза исходного поля')

axes[1, 0].imshow(np.abs(F_2d), extent=[b1, b2, b1, b2], cmap='hot')
axes[1, 0].set_title('Амплитуда БПФ')
axes[1, 1].imshow(np.angle(F_2d), extent=[b1, b2, b1, b2], cmap='hot')
axes[1, 1].set_title('Фаза БПФ')

plt.tight_layout()
plt.show()

# Исследование влияния параметров N и M
N_values = [256, 512, 1024, 2048]
M_values = [N, N * 2, N * 4]

for N in N_values:
    hx = (a2 - a1) / N
    x = np.linspace(a1, a2, N, endpoint=False)
    f = light(x)

    for M in M_values:
        b2 = (N * N) / (4 * a2 * M)
        b1 = -b2
        u = np.linspace(b1, b2, N, endpoint=False)

        F = bpf(f, N, M, hx)

        plt.figure(figsize=(10, 5))
        plt.plot(u, np.abs(F), label=f'N={N}, M={M}')
        plt.title('Амплитуда светового поля для разных N и M')
        plt.legend()
        plt.grid()
        plt.show()

