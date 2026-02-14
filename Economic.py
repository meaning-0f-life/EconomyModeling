import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

plt.rcParams['font.family'] = 'serif'

# Исходные параметры (базовые)
BASE_PARAMS = {
    'N': 5,
    'lamb': 0.1,  # износ
    'delta': 0.01,
    'gam': 0.01,  # инфляция
    'a_0': 1000,  # стартовый капитал
    'k': 1.5,  # производительность
    'b': 0.95,  # отдача от капитала
    'p_maximize': 1,  # ограничение инвестиций
    'sigma_util': 2,
    'a': 10  # длина интервалов в множестве возможных состояний k
}


# Функция производительности труда
def f_x(x, k, b):
    return k * (x ** b)


# Функция полезности
def utility(p, f, sigma):
    c = (1 - p) * f  # потребление
    if (1 - sigma) <= 1e-10:
        if c <= 1e-10:  # маленькое положительное число вместо 0
            return -1e10
        return np.log(c)
    return (c ** (1 - sigma) - 1) / (1 - sigma)


def solve_dp(params, show_graph=False):
    """
    Решает задачу динамического программирования с заданными параметрами
    """
    N = params['N']
    lamb = params['lamb']
    delta = params['delta']
    gam = params['gam']
    a_0 = params['a_0']
    k = params['k']
    b = params['b']
    p_maximize = params['p_maximize']
    sigma_util = params['sigma_util']
    a = params['a']

    # Максимальная траектория для определения границ сетки
    k_max_trajectory = np.zeros(N)
    k_max_trajectory[0] = a_0
    for i in range(1, N):
        k_max_trajectory[i] = (1 - lamb) * k_max_trajectory[i - 1] + p_maximize * f_x(k_max_trajectory[i - 1], k, b)
    maxim = max(a_0, k_max_trajectory[N - 1])

    # Сетки состояний и управлений
    A = np.arange(0, maxim + a, a)
    U = np.arange(0, 1 + delta, delta)

    # Инициализация массивов
    F_i_all = np.zeros((N, len(A)))
    s_i_all = np.zeros((N, len(A)))

    # Конечное условие
    for j in range(len(A)):
        current_capital = A[j]
        k_term = utility(0, current_capital, sigma_util)
        F_i_all[N - 1, j] = k_term

    # Обратный ход Беллмана
    for i in range(N - 2, -1, -1):
        for j in range(len(A)):
            current_state = A[j]
            f_val = f_x(current_state, k, b)

            best_value = -np.inf
            best_control = 0

            for t in range(len(U)):
                control = U[t]
                current_utility = utility(control, f_val, sigma_util)
                next_state = (1 - lamb) * current_state + control * f_val
                state_idx = np.argmin(np.abs(A - next_state))

                discounted_utility = current_utility / ((1 + gam) ** i)
                future_value = F_i_all[i + 1, state_idx]
                total_value = discounted_utility + future_value

                if total_value > best_value:
                    best_value = total_value
                    best_control = control

            s_i_all[i, j] = best_control
            F_i_all[i, j] = best_value

    # Построение оптимальной траектории
    k_opt = np.zeros(N)
    p_opt = np.zeros(N)

    k_opt[0] = a_0
    start_idx = np.argmin(np.abs(A - k_opt[0]))
    p_opt[0] = s_i_all[0, start_idx]

    for i in range(1, N):
        k_opt[i] = (1 - lamb) * k_opt[i - 1] + p_opt[i - 1] * f_x(k_opt[i - 1], k, b)
        if i < N - 1:
            state_idx = np.argmin(np.abs(A - k_opt[i]))
            p_opt[i] = s_i_all[i, state_idx]

    p_opt[N - 1] = p_opt[N - 2]

    if show_graph:
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

    return k_opt, p_opt


def analyze_parameter(param_name, param_values, base_params, print_results=True, save_graphs=True):
    """
    Анализирует влияние одного параметра на оптимальное решение

    Parameters:
    -----------
    param_name : str
        Название параметра для анализа ('lamb', 'gam', 'b', 'sigma_util', 'N')
    param_values : list or array
        Сетка значений параметра
    base_params : dict
        Базовые параметры модели
    print_results : bool
        Флаг для вывода результатов в консоль
    save_graphs : bool
        Флаг для сохранения графиков в файлы
    """
    # Создаем нормализацию для цветовой шкалы
    norm = Normalize(vmin=min(param_values), vmax=max(param_values))
    cmap = plt.colormaps.get_cmap('cool')  # От красного (большие) к синему (маленькие)

    # Создаем фигуру с двумя подграфиками и выделяем место для colorbar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    # Добавляем место для colorbar справа
    fig.subplots_adjust(right=0.85)

    # Словарь для хранения результатов
    results = {}

    for param_value in param_values:
        # Модифицируем параметры
        params = base_params.copy()
        params[param_name] = param_value

        # Решаем задачу
        k_opt, p_opt = solve_dp(params)

        if print_results:
            print(f"{param_name}: {param_value}")
            print(f"p_opt: {p_opt}")
            print(f"k_opt: {k_opt}")

        # Сохраняем результаты
        results[param_value] = {'k': k_opt, 'p': p_opt}

        # Определяем цвет
        color = cmap(norm(param_value))

        # Строим графики
        time_points = range(params['N'])

        # График капитала
        ax1.plot(time_points, k_opt, color=color, linewidth=2,
                 label=f'{param_name} = {param_value:.3f}')

        # График управления
        ax2.plot(time_points, p_opt, color=color, linewidth=2,
                 label=f'{param_name} = {param_value:.3f}')

    # Настройка первого графика
    ax1.set_xlabel('Момент времени')
    ax1.set_ylabel('Капитал k')
    ax1.set_title(f'Влияние параметра {param_name} на оптимальную траекторию капитала')
    ax1.grid(True)
    ax1.legend(loc='best', fontsize='small')

    # Настройка второго графика
    ax2.set_xlabel('Момент времени')
    ax2.set_ylabel('Доля инвестиций p')
    ax2.set_title(f'Влияние параметра {param_name} на оптимальное управление')
    ax2.grid(True)
    ax2.legend(loc='best', fontsize='small')

    # Добавляем цветовую шкалу справа от графиков
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(param_name, fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Оставляем место для colorbar

    # Сохраняем график, если требуется
    if save_graphs:
        # Создаем папку для графиков, если её нет
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        # Сохраняем в файл
        filename = f'graphs/analysis_{param_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранен в файл: {filename}")

    plt.show()

    return results


def analyze_lamb(lamb_values, base_params, save_graphs=True):
    """Анализ влияния параметра износа lamb"""
    return analyze_parameter('lamb', lamb_values, base_params, save_graphs=save_graphs)


def analyze_gam(gam_values, base_params, save_graphs=True):
    """Анализ влияния параметра инфляции gam"""
    return analyze_parameter('gam', gam_values, base_params, save_graphs=save_graphs)


def analyze_b(b_values, base_params, save_graphs=True):
    """Анализ влияния эластичности производства по капиталу b"""
    return analyze_parameter('b', b_values, base_params, save_graphs=save_graphs)


def analyze_sigma(sigma_values, base_params, save_graphs=True):
    """Анализ влияния параметра функции полезности sigma_util"""
    return analyze_parameter('sigma_util', sigma_values, base_params, save_graphs=save_graphs)


def analyze_N(N_values, base_params, print_results=True, save_graphs=True):
    """Анализ влияния временного горизонта N"""
    # Для N нужно также адаптировать другие параметры
    results = {}

    # Создаем фигуру с двумя подграфиками и выделяем место для colorbar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    # Добавляем место для colorbar справа
    fig.subplots_adjust(right=0.85)

    norm = Normalize(vmin=min(N_values), vmax=max(N_values))
    cmap = plt.colormaps.get_cmap('cool')

    for N_value in N_values:
        params = base_params.copy()
        params['N'] = N_value

        k_opt, p_opt = solve_dp(params)
        if print_results:
            print(f"N: {N_value}")
            print(f"p_opt: {p_opt}")
            print(f"k_opt: {k_opt}")

        results[N_value] = {'k': k_opt, 'p': p_opt}

        color = cmap(norm(N_value))
        time_points = range(N_value)

        ax1.plot(time_points, k_opt, color=color, linewidth=2,
                 label=f'N = {N_value}')
        ax2.plot(time_points, p_opt, color=color, linewidth=2,
                 label=f'N = {N_value}')

    ax1.set_xlabel('Момент времени')
    ax1.set_ylabel('Капитал k')
    ax1.set_title('Влияние временного горизонта N на траекторию капитала')
    ax1.grid(True)
    ax1.legend(loc='best', fontsize='small')

    ax2.set_xlabel('Момент времени')
    ax2.set_ylabel('Доля инвестиций p')
    ax2.set_title('Влияние временного горизонта N на оптимальное управление')
    ax2.grid(True)
    ax2.legend(loc='best', fontsize='small')

    # Добавляем цветовую шкалу справа от графиков
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('N', fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Оставляем место для colorbar

    # Сохраняем график, если требуется
    if save_graphs:
        # Создаем папку для графиков, если её нет
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        # Сохраняем в файл
        filename = f'graphs/analysis_N.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранен в файл: {filename}")

    plt.show()

    return results


# Пример использования
if __name__ == "__main__":
    # Анализ влияния параметра износа
    lamb_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    print("Анализ влияния параметра износа lamb...")
    results_lamb = analyze_lamb(lamb_values, BASE_PARAMS, save_graphs=True)

    # Анализ влияния инфляции
    gam_values = [0.0, 0.01, 0.02, 0.05, 0.1]
    print("\nАнализ влияния инфляции gam...")
    results_gam = analyze_gam(gam_values, BASE_PARAMS, save_graphs=True)

    # Анализ влияния производительности
    k_values = [0.6, 0.8, 1, 1.5, 3]
    print("\nАнализ влияния производительности k...")
    results_k = analyze_parameter("k", k_values, BASE_PARAMS, save_graphs=True)

    # Анализ влияния эластичности производства
    b_values = [0.7, 0.8, 0.9, 0.95, 0.99]
    print("\nАнализ влияния эластичности производства b...")
    results_b = analyze_b(b_values, BASE_PARAMS, save_graphs=True)

    # Анализ влияния параметра функции полезности
    sigma_values = [0.5, 1, 2, 3, 5]
    print("\nАнализ влияния параметра функции полезности sigma...")
    results_sigma = analyze_sigma(sigma_values, BASE_PARAMS, save_graphs=True)

    # Анализ влияния временного горизонта
    N_values = [3, 5, 7, 9]
    print("\nАнализ влияния временного горизонта N...")
    results_N = analyze_N(N_values, BASE_PARAMS, save_graphs=True)