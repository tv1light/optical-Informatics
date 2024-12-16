import numpy as np
import matplotlib.pyplot as plt
a1 = -5
a2 = 5
N = 1 << 10

#функция, согласно варианту
light = lambda x: np.sin(4*np.pi*x)
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

# Аналитическое преобразование Фурье
def analitic_Furie(f):
    h_x = 2*a2 / N
    h_u = 2*b2 / N
    F = 0
    arr_F = []
    for i_u in range(N):
        for i_x in range (N):
           F += h_x * (f[i_x]) * np.exp(-2 * np.pi * 1j * x[i_x] * (-b2 + i_u * h_u))
        arr_F.append(F)
        F = 0
    return arr_F

F_analytical = analitic_Furie(f)

#графики исходного сигнала, согласно варианту
plt.figure(figsize=(20,10))

plt.subplot(2, 2, 1)
plt.plot(x, np.abs(f))
plt.title('Амплитуда светового поля, согласно варианту')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(x, np.angle(f))
plt.title('Фаза светового поля, согласно варианту')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(u, np.abs(F), label='БПФ', c = 'c', lw = 10)
plt.title('Амплитуда светового поля БПФ, согласно варианту')
plt.plot(u, np.abs(F_analytical), label = 'Аналитическое', c = 'k', lw = 3)
plt.title('Амплитуда светового поля БПФ и аналитическое, согласно варианту')
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(u, np.angle(F), label='БПФ', c = 'c', lw = 10)
plt.title('Фаза светового поля БПФ, согласно варианту')
plt.plot(u, np.angle(F_analytical), label='Аналитическое', c = 'k', lw = 3)
plt.title('Фаза светового поля БПФ и аналитическое, согласно варианту')
plt.legend()
plt.grid()

# двухмерная функция, согласно варианту
light_2d = lambda x, y: np.sin(4*np.pi*x) * np.sin(4*np.pi*y)
X, Y = np.meshgrid(x, x)
f_2d = light_2d(X, Y)

#графики исходного сигнала в 2D, согласно варианту
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
amp = arr[0].imshow(np.abs(f_2d), cmap='hot', interpolation='nearest')
arr[0].set_title('Амплитуда светового поля в 2D, согласно варианту')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(f_2d), cmap='hot', interpolation='nearest')
arr[1].set_title('Фаза светового поля в 2D, согласно варианту')
fig.colorbar(phase, ax=arr[1])
plt.show()

# двухмерное быстрое преобразование Фурье
def bpf_2d(Z, a, b, N, M):
    for i in range(N):
      h = (b - a) / (N - 1)
      Z[:, i] = bpf(Z[:, i], N, M, h)
    for i in range(N):
      h = (b - a) / (N - 1)
      Z[i, :] = bpf(Z[i, :], N, M, h)
    return Z

F_2d = f_2d.astype(np.complex128) # в переменную записываем двухмерную функцию, массив комплексных чисел
F_2d = bpf_2d(F_2d, a1, a2, N, M) # двухмерное БПФ
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
amp = arr[0].imshow(np.abs(F_2d), cmap='hot', interpolation='nearest')
arr[0].set_title('Амплитуда светового поля БПФ в 2D, согласно варианту')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(F_2d), cmap='hot', interpolation='nearest')
arr[1].set_title('Фаза светового поля БПФ в 2D, согласно варианту')
fig.colorbar(phase, ax=arr[1])
plt.show()

analytical_2D = np.zeros((N, N), dtype=complex) # создание комплексного массива для аналитического двухмерного решения

# Двухмерное аналитическое решение
for i in range(N):
  for j in range(N):
    analytical_2D[i][j] = F_analytical[i] * F_analytical[j]

#графики исходного сигнала аналитически в 2D, согласно варианту
fig, arr = plt.subplots(1, 2, figsize=(15, 5))
arr[0].imshow(np.abs(analytical_2D), cmap='hot', interpolation='nearest')
arr[0].set_title('Амплитуда светового поля аналитическое в 2D, согласно варианту')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(analytical_2D), cmap='hot', interpolation='nearest')
arr[1].set_title('Фаза светового поля аналитическое в 2D, согласно варианту')
fig.colorbar(phase, ax=arr[1])
plt.show()

#GAUSS
gauss = lambda x: np.exp(-x ** 2)
f_gauss = gauss(x)
F_gauss = bpf(f_gauss, N, M, hx) # результат после БПФ

#графики гауссова пучка
plt.figure(figsize=(20,10))

plt.subplot(2, 2, 1)
plt.plot(x, np.abs(f_gauss))
plt.title('Амплитуда гауссова пучка')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(x, np.angle(f_gauss))
plt.title('Фаза гауссова пучка')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(u, np.abs(F_gauss))
plt.title('Амплитуда гауссова пучка БПФ')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(u, np.angle(F_gauss))
plt.title('Фаза гауссова пучка БПФ')
plt.grid()

interval = abs(N ** 2 / (4 * a2 * M))
# Шаг дисретизации
step = 2 * interval / (N - 1)
F_rect = analitic_Furie(f_gauss) # аналитическое преобразование Фурье гауссова пучка

#графики исходного сигнала БПФ
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(u, np.abs(F_rect))
plt.title('Амплитуда гауссова пучка аналитическое')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(u, np.angle(F_rect))
plt.title('Фаза гауссова пучка поля аналитически')
plt.grid()

# Двухмерный гауссов пучок
gauss_2d = lambda x, y: np.exp(-x ** 2 - y ** 2)
f_gauss_2d = gauss_2d(X, Y)

#графики гауссова пучка в 2D
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
amp = arr[0].imshow(np.abs(f_gauss_2d), cmap='hot', interpolation='nearest')
arr[0].set_title('Амплитуда гауссова пучка в 2d')
phase = arr[1].imshow(np.angle(f_gauss_2d), cmap='hot', interpolation='nearest')
arr[1].set_title('Фаза гауссова пучка в 2d')
fig.colorbar(phase, ax=arr[1])
plt.show()

# Двухмерное БПФ двухмерного гауссова пучка
F_gauss_2d = f_gauss_2d.astype(np.complex128)
F_gauss_2d = bpf_2d(F_gauss_2d, a1, a2, N, M)

#графики гауссова пучка БПФ в 2D
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
amp = arr[0].imshow(np.abs(F_gauss_2d), cmap='hot', interpolation='nearest')
arr[0].set_title('Амплитуда гауссова пучка БПФ в 2d')
phase = arr[1].imshow(np.angle(F_gauss_2d), cmap='hot', interpolation='nearest')
arr[1].set_title('Фаза гауссова пучка БПФ в 2d')
fig.colorbar(phase, ax=arr[1])
plt.show()

# Аналитическое двухмерное преобразование Фурье для гауссова пучка
F_gauss_2d_analit = np.zeros((N, N), dtype=complex)
for i in range(N):
  for j in range(N):
    F_gauss_2d_analit[i][j] = F_rect[i] * F_rect[j]

#графики   аналитически гауссова пучка БПФ в 2D
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
amp = arr[0].imshow(np.abs(F_gauss_2d_analit), cmap='hot', interpolation='nearest')
arr[0].set_title('Амплитуда гауссова пучка аналитически в 2d')
phase = arr[1].imshow(np.angle(F_gauss_2d_analit), cmap='hot', interpolation='nearest')
arr[1].set_title('Фаза гауссова пучка аналитически в 2d')
fig.colorbar(phase, ax=arr[1])
plt.show()
