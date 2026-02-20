import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

# Настройка шрифтов и размера текста
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

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

        plt.figure(figsize=(10, 6))
        plt.plot(k_opt, linewidth=2.5)
        plt.xlabel('i, момент времени', fontsize=14)
        plt.ylabel('k, значение удельного капитала', fontsize=14)
        plt.title('Траектория капитала', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=12)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(p_opt, linewidth=2.5)
        plt.xlabel('i, момент времени', fontsize=14)
        plt.ylabel('p, доля продукта, направ. на инвестирование', fontsize=14)
        plt.title('Траектория управления', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=12)
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

    # Создаем фигуру с двумя подграфиками большего размера
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    # Добавляем место для colorbar и легенды справа
    fig.subplots_adjust(right=0.8)

    # Словарь для хранения результатов
    results = {}

    # Список маркеров для лучшего различения линий
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, param_value in enumerate(param_values):
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

        # Выбираем маркер (циклически)
        marker = markers[idx % len(markers)]

        # Строим графики
        time_points = range(params['N'])

        # График капитала с маркерами
        ax1.plot(time_points, k_opt, color=color, linewidth=3,
                marker=marker, markersize=8, markevery=1,
                label=f'{param_name} = {param_value:.3f}')

        # График управления с маркерами
        ax2.plot(time_points, p_opt, color=color, linewidth=3,
                marker=marker, markersize=8, markevery=1,
                label=f'{param_name} = {param_value:.3f}')

    # Настройка первого графика
    ax1.set_xlabel('Момент времени', fontsize=14)
    ax1.set_ylabel('Капитал k', fontsize=14)
    ax1.set_title(f'Влияние параметра {param_name} на оптимальную траекторию капитала', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    # Размещаем легенду справа от графика
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)

    # Настройка второго графика
    ax2.set_xlabel('Момент времени', fontsize=14)
    ax2.set_ylabel('Доля инвестиций p', fontsize=14)
    ax2.set_title(f'Влияние параметра {param_name} на оптимальное управление', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)

    # Добавляем цветовую шкалу справа от графиков (немного смещаем влево из-за легенды)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(param_name, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Оставляем место для colorbar и легенды

    # Сохраняем график, если требуется
    if save_graphs:
        # Создаем папку для графиков, если её нет
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        # Сохраняем в файл с высоким разрешением
        filename = f'graphs/analysis_{param_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
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

    # Создаем фигуру с двумя подграфиками большего размера
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    # Добавляем место для colorbar и легенды справа
    fig.subplots_adjust(right=0.8)

    norm = Normalize(vmin=min(N_values), vmax=max(N_values))
    cmap = plt.colormaps.get_cmap('cool')

    # Список маркеров для лучшего различения линий
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, N_value in enumerate(N_values):
        params = base_params.copy()
        params['N'] = N_value

        k_opt, p_opt = solve_dp(params)
        if print_results:
            print(f"N: {N_value}")
            print(f"p_opt: {p_opt}")
            print(f"k_opt: {k_opt}")

        results[N_value] = {'k': k_opt, 'p': p_opt}

        color = cmap(norm(N_value))
        marker = markers[idx % len(markers)]
        time_points = range(N_value)

        ax1.plot(time_points, k_opt, color=color, linewidth=3,
                marker=marker, markersize=8, markevery=1,
                label=f'N = {N_value}')
        ax2.plot(time_points, p_opt, color=color, linewidth=3,
                marker=marker, markersize=8, markevery=1,
                label=f'N = {N_value}')

    ax1.set_xlabel('Момент времени', fontsize=14)
    ax1.set_ylabel('Капитал k', fontsize=14)
    ax1.set_title('Влияние временного горизонта N на траекторию капитала', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)

    ax2.set_xlabel('Момент времени', fontsize=14)
    ax2.set_ylabel('Доля инвестиций p', fontsize=14)
    ax2.set_title('Влияние временного горизонта N на оптимальное управление', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)

    # Добавляем цветовую шкалу справа от графиков
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('N', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Оставляем место для colorbar и легенды

    # Сохраняем график, если требуется
    if save_graphs:
        # Создаем папку для графиков, если её нет
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        # Сохраняем в файл с высоким разрешением
        filename = f'graphs/analysis_N.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График сохранен в файл: {filename}")

    plt.show()

    return results

FAST_TEST_PARAMS_50 = {
    'N': 50,
    'lamb': 0.05,
    'delta': 0.1,
    'gam': 0.01,
    'a_0': 100,
    'k': 1.2,
    'b': 0.8,
    'p_maximize': 1,
    'sigma_util': 1.5,
    'a': 50
}

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

    '''
    print("Запуск оптимизации на 50 периодах...")
    k_opt, p_opt = solve_dp(FAST_TEST_PARAMS_50, show_graph=True)
    print(f"Финальный капитал: {k_opt[-1]:.2f}")
    print(f"Средняя доля инвестиций: {np.mean(p_opt):.3f}")
    '''