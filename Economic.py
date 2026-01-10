#import math
import numpy as np
import time
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

# Входные данные

# Пример
N = 5
lamb = 0.02       # 2% износ - новое оборудование
delta = 0.05
gam = 0.01       # 6% инфляция - умеренная для растущей экономики
a_0 = 730        # $5,000 ВВП на душу (низкий старт)
k = 1.22           # высокая производительность за счет дешевой рабочей силы
b = 0.38           # хорошая отдача от капитала
k_term = 0.099      # ценность накопленного капитала
b_term = 8.51
p_maximize = 1  # агрессивные инвестиции 40%

a = 1 #a_0 * 0.01  # длина интервалов в множестве возможных состояний k


'''
# Практическое правило для Кобба-Дугласа
def practical_max_control(alpha, time_horizon):
    """
    alpha - эластичность производства по капиталу
    time_horizon - горизонт планирования
    """
    if time_horizon <= 30:
        return min(alpha * 1.2, 0.8)
    elif time_horizon <= 100:
        return min(alpha * 1.0, 0.7)
    else:
        return min(alpha * 0.8, 0.6)

p_maximize = practical_max_control(b, N)
print(f"Аналитическая оценка оптимального управления: {p_maximize:.3f}")
'''

# Функция производительности труда
def f_x(x):
    return k * (x ** b)

# Функция полезности
def utility(p, f):
    c = (1 - p) * f # потребление
    if c <= 1e-10:  # маленькое положительное число вместо 0
        return -1e10
    return np.log(c)


# Подготовка вспомогательных массивов
start_time = time.time()  # начало отсчета времени
rho = np.arange(0, 1 + delta, delta)

'''
k_i_test = np.zeros((N, len(rho)))
max_elts = np.zeros(N)

for q in range(0, len(rho), 1):
    k_i_test[0][q] = a_0  # значение состояния в начальный момент времени (а_0)

max_elts = np.zeros(N)

for i in range(1, N, 1):
    # if i < N/2-1:
    #     lamb = 0.0001
    # else:
    #     lamb = 0.001
    for j in range(0, len(rho), 1):
        k_i_j_max = 0.0
        for k in range(0, len(rho), 1):
            k_i_j_max = max(k_i_j_max, (1 - lamb) * k_i_test[i - 1][k] + rho[j] * f_x(k_i_test[i - 1][k]))
        k_i_test[i][j] = k_i_j_max

    max_elts[i] = max(k_i_test[i])

maxim = max(a_0, max(max_elts))  # максимальное значение состояния при всех возможных значениях управления в каждый момент времени
'''

k_max_trajectory = np.zeros(N)
k_max_trajectory[0] = a_0
for i in range(1, N):
    k_max_trajectory[i] = (1 - lamb) * k_max_trajectory[i-1] + p_maximize * f_x(k_max_trajectory[i-1])
maxim = max(a_0, k_max_trajectory[N-1])

print("максимальное возможное значение состояния k = ", maxim)
print()

A_k = np.arange(0, maxim + a, a)  # массив возможных значений состояния

ttt = time.time() - start_time  # конец замера времени
print("Время подготовки вспомогательных массивов U и A_K = %s сек." % ttt)

# ПЕРВЫЙ ЭТАП
start_time = time.time()

A = A_k
U = rho

# Инициализация массивов
F_i_all = np.zeros((N, len(A)))  # [время, состояние]
s_i_all = np.zeros((N, len(A)))  # [время, состояние]

# Конечное условие
for j in range(len(A)):
    #F_i_all[N - 1, j] = utility(0, f_x(A[j])) / ((1 + gam) ** (N - 1))  # ρ=0, весь продукт на потребление # терминальная функция
    F_i_all[N - 1, j] = k_term * A[j] + b_term # терминальная функция

# Обратный ход Беллмана
for i in range(N - 2, -1, -1):  # от N-2 до 0
    for j in range(len(A)):  # по всем состояниям
        current_state = A[j]
        f_val = f_x(current_state)

        best_value = -np.inf
        best_control = 0

        # Перебор всех возможных управлений
        for t in range(len(U)):
            control = U[t]

            # Текущая полезность
            current_utility = utility(control, f_val)

            # Следующее состояние
            next_state = (1 - lamb) * current_state + control * f_val

            # Находим ближайшее состояние в сетке
            state_idx = np.argmin(np.abs(A - next_state))

            # Дисконтированное значение
            discounted_utility = current_utility / ((1 + gam) ** i)
            future_value = F_i_all[i + 1, state_idx]

            total_value = discounted_utility + future_value

            if total_value > best_value:
                best_value = total_value
                best_control = control

        # Сохраняем оптимальное управление и значение функции Беллмана
        s_i_all[i, j] = best_control
        F_i_all[i, j] = best_value

print("Обратный ход Беллмана: %s секунд" % (time.time() - start_time))


# ВТОРОЙ ЭТАП - ОБРАТНЫЙ ХОД
start_time = time.time()

# Инициализация массивов
k_opt = np.zeros(N)  # оптимальные состояния
p_opt = np.zeros(N)  # оптимальные управления

# Начальное условие
k_opt[0] = a_0
print("k[0] = ", k_opt[0])

# Находим оптимальное управление для начального состояния
start_idx = np.argmin(np.abs(A_k - k_opt[0]))
p_opt[0] = s_i_all[0, start_idx]
print("p[0] = ", p_opt[0])

# Прямой ход: вычисляем оптимальную траекторию
for i in range(1, N):
    # Вычисляем следующее состояние по динамике системы
    k_opt[i] = (1 - lamb) * k_opt[i - 1] + p_opt[i - 1] * f_x(k_opt[i - 1])

    # Находим оптимальное управление для текущего состояния
    if i < N - 1:  # для последнего шага управление не нужно
        state_idx = np.argmin(np.abs(A_k - k_opt[i]))
        p_opt[i] = s_i_all[i, state_idx]

# Для последнего момента берем управление равное последнему
p_opt[N - 1] = p_opt[N - 2]

print("Время второго этапа: %s секунд" % (time.time() - start_time))

print(k_opt)
print(p_opt)

plt.plot(k_opt)
plt.xlabel('i, момент времени')
plt.ylabel('k, значение удельного капитала')
plt.grid(True)
plt.show()

plt.plot(p_opt)
plt.xlabel('i, момент времени')
plt.ylabel('p, доля продукта, направ. на инвестирование')
plt.grid(True)
plt.show()