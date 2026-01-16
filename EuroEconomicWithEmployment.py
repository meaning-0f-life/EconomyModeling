#import math
import numpy as np
import time
import matplotlib.pyplot as plt
#from datetime import datetime
#from scipy import stats
import pandas as pd

plt.rcParams['font.family'] = 'serif'

# Страны с их кодами
countries_needed = {
    'Китай': 'CHN',
    'Россия': 'RUS',
    'Объединенные Арабские Эмираты': 'ARE',
    'Армения': 'ARM',
    'Беларусь': 'BLR',
    'Бразилия': 'BRA',
    'Канада': 'CAN',
    'Германия': 'DEU',
    'Индия': 'IND',
    'Япония': 'JPN',
    'Казахстан': 'KAZ',
    'США': 'USA'
}

# Годы, которые нам нужны
years_needed = [str(year) for year in range(1995, 2025)]
years = range(1995, 2025)

# Читаем CSV файлы (все данные взяты из https://data.worldbank.org)
df_capital = pd.read_csv('data/CapitalData.csv', skiprows=3)
df_gdp = pd.read_csv('data/GDP_Data.csv', skiprows=3)
df_inflation = pd.read_csv('data/InflationData.csv', skiprows=3)
df_population = pd.read_csv('data/Population.csv', skiprows=3)
df_employment = pd.read_csv('data/Employment.csv', skiprows=3)

# Словарь для результатов
capital_data = {}
gdp_data = {}
inflation_data_by_country = {}
population_data = {}
employment_data = {}

# Фильтруем по нужным странам и индикатору для капитала
for rus_name, code in countries_needed.items():
    # Находим строку для страны и индикатора капитала
    country_row = df_capital[(df_capital['Country Code'] == code) &
                             (df_capital['Indicator Code'] == 'NE.GDI.FTOT.CD')]

    if not country_row.empty:
        # Извлекаем данные за нужные годы
        data = []
        for year in years_needed:
            if year in df_capital.columns:
                value = country_row[year].values[0]
                if pd.notna(value):
                    # Преобразуем в млн долларов и округляем
                    data.append(round(float(value) / 1000000, 2))
                else:
                    data.append(0.0)
            else:
                data.append(0.0)

        capital_data[rus_name] = data

# Фильтруем по нужным странам и индикатору для ВВП
for rus_name, code in countries_needed.items():
    # Находим строку для страны и индикатора ВВП
    country_row = df_gdp[(df_gdp['Country Code'] == code) &
                         (df_gdp['Indicator Code'] == 'NY.GDP.MKTP.CD')]

    if not country_row.empty:
        # Извлекаем данные за нужные годы
        data = []
        for year in years_needed:
            if year in df_gdp.columns:
                value = country_row[year].values[0]
                if pd.notna(value):
                    # Преобразуем в млн долларов и округляем до 2 знаков
                    data.append(round(float(value) / 1000000, 2))
                else:
                    data.append(0.0)
            else:
                data.append(0.0)

        gdp_data[rus_name] = data

# Фильтруем по нужным странам и индикатору для инфляции
for rus_name, code in countries_needed.items():
    # Находим строку для страны и индикатора инфляции
    country_row = df_inflation[(df_inflation['Country Code'] == code) &
                               (df_inflation['Indicator Code'] == 'NY.GDP.DEFL.KD.ZG')]

    if not country_row.empty:
        # Создаем словарь инфляции по годам
        inflation_dict = {}
        for year in years_needed:
            if year in df_inflation.columns:
                value = country_row[year].values[0]
                if pd.notna(value) and value != '':
                    try:
                        # Преобразуем в проценты (данные уже в %)
                        inflation_rate = float(value)
                        inflation_dict[int(year)] = inflation_rate / 100  # переводим % в доли
                    except:
                        inflation_dict[int(year)] = 0.0
                else:
                    inflation_dict[int(year)] = 0.0
            else:
                inflation_dict[int(year)] = 0.0

        inflation_data_by_country[rus_name] = inflation_dict

# Фильтруем по нужным странам для популяции
for rus_name, code in countries_needed.items():
    # Находим строку для страны и индикатора популяции
    country_row = df_population[(df_population['Country Code'] == code) &
                                (df_population['Indicator Code'] == 'SP.POP.TOTL')]

    if not country_row.empty:
        # Извлекаем данные за нужные годы
        data = []
        for year in years_needed:
            if year in df_population.columns:
                value = country_row[year].values[0]
                if pd.notna(value):
                    # Преобразуем в миллионы человек
                    data.append(round(float(value) / 1000000, 4))
                else:
                    data.append(0.0)
            else:
                data.append(0.0)

        population_data[rus_name] = data

# Фильтруем по нужным странам для занятости
for rus_name, code in countries_needed.items():
    # Находим строку для страны и индикатора занятости
    country_row = df_employment[(df_employment['Country Code'] == code) &
                                (df_employment['Indicator Code'] == 'SL.EMP.TOTL.SP.ZS')]

    if not country_row.empty:
        # Извлекаем данные за нужные годы
        data = []
        for year in years_needed:
            if year in df_employment.columns:
                value = country_row[year].values[0]
                if pd.notna(value):
                    # Данные уже в процентах
                    data.append(float(value) / 100)  # переводим % в доли
                else:
                    data.append(0.0)
            else:
                data.append(0.0)

        employment_data[rus_name] = data

# Функция для получения инфляции по году (в долях)
def get_inflation(years, country):
    # Собираем значения инфляции для указанных годов
    inflation_values = []

    for year in years:
        # Получаем значение инфляции и переводим проценты в доли
        inflation_value = inflation_data_by_country[country][year]
        inflation_values.append(inflation_value)

    # Возвращаем среднее значение
    return sum(inflation_values) / len(inflation_values)

# Вычисляем численность рабочей силы (труда)
def calculate_labor_force(population_data, employment_data):
    """Вычисляет численность рабочей силы: L = население * уровень занятости"""
    labor_force_data = {}

    for country in population_data.keys():
        if country in employment_data:
            population = population_data[country]
            employment_rate = employment_data[country]

            labor_force = []
            for pop, emp_rate in zip(population, employment_rate):
                if pop > 0 and emp_rate > 0:
                    labor_force.append(pop * emp_rate)  # млн человек
                else:
                    labor_force.append(0.0)

            labor_force_data[country] = labor_force

    return labor_force_data


# Получаем данные о рабочей силе
labor_data = calculate_labor_force(population_data, employment_data)


def estimate_cobb_douglas_params(gdp_values, capital_values, labor_values, country_name):
    """
    Оценка параметров производственной функции Кобба-Дугласа:
    Y = A * K^α * L^β

    Через линеаризацию: ln(Y) = ln(A) + α*ln(K) + β*ln(L)
    """
    # Проверяем, что данные корректны
    if len(gdp_values) != len(capital_values) or len(gdp_values) != len(labor_values):
        raise ValueError(f"Длины массивов не совпадают для {country_name}: "
                         f"GDP={len(gdp_values)}, K={len(capital_values)}, L={len(labor_values)}")

    # Убираем нулевые и отрицательные значения
    Y = np.array(gdp_values)
    K = np.array(capital_values)
    L = np.array(labor_values)

    # Проверяем наличие положительных значений
    mask = (Y > 0) & (K > 0) & (L > 0)

    if np.sum(mask) < 3:  # нужно минимум 3 наблюдения для регрессии
        print(f"  ⚠️  Недостаточно положительных данных для {country_name}")
        return None

    Y_clean = Y[mask]
    K_clean = K[mask]
    L_clean = L[mask]

    # Берем логарифмы
    ln_Y = np.log(Y_clean)
    ln_K = np.log(K_clean)
    ln_L = np.log(L_clean)

    # Создаем матрицу регрессоров
    X = np.column_stack((np.ones_like(ln_K), ln_K, ln_L))

    # Множественная линейная регрессия
    try:
        # Используем метод наименьших квадратов
        coeffs, _, _, _ = np.linalg.lstsq(X, ln_Y, rcond=None)

        intercept, alpha, beta = coeffs

        # Параметры модели
        A = np.exp(intercept)  # технологический параметр

        # Рассчитываем статистику
        Y_pred = intercept + alpha * ln_K + beta * ln_L
        residuals = ln_Y - Y_pred
        SSR = np.sum(residuals ** 2)  # сумма квадратов остатков
        SST = np.sum((ln_Y - np.mean(ln_Y)) ** 2)  # общая сумма квадратов
        R_squared = 1 - SSR / SST if SST != 0 else 0

        # Проверка на отдачу от масштаба
        returns_to_scale = alpha + beta

        '''
        print(f"\n{'=' * 60}")
        print(f"ОЦЕНКА ПАРАМЕТРОВ ДЛЯ: {country_name}")
        print(f"{'=' * 60}")
        print(f"Уравнение: ln(Y) = {intercept:.4f} + {alpha:.4f} * ln(K) + {beta:.4f} * ln(L)")
        print(f"Параметры:")
        print(f"  A (технология) = {A:.4f}")
        print(f"  α (эластичность по капиталу) = {alpha:.4f}")
        print(f"  β (эластичность по труду) = {beta:.4f}")
        print(f"  Σ = α + β (отдача от масштаба) = {returns_to_scale:.4f}")
        print(f"Статистика:")
        print(f"  R² = {R_squared:.4f}")
        print(f"  MSE = {SSR/len(ln_Y):.4f}")

        # Проверка экономической осмысленности
        if alpha < 0:
            print(f"  ⚠️  ВНИМАНИЕ: α отрицательный!")
        elif alpha > 1:
            print(f"  ⚠️  ВНИМАНИЕ: α > 1")
        elif 0.2 <= alpha <= 0.4:
            print(f"  ✓ α в реалистичном диапазоне (0.2-0.4)")
        else:
            print(f"  ⚠️  α вне типичного диапазона")

        if beta < 0:
            print(f"  ⚠️  ВНИМАНИЕ: β отрицательный!")
        elif beta > 1:
            print(f"  ⚠️  ВНИМАНИЕ: β > 1")
        elif 0.6 <= beta <= 0.8:
            print(f"  ✓ β в реалистичном диапазоне (0.6-0.8)")
        else:
            print(f"  ⚠️  β вне типичного диапазона")

        if returns_to_scale > 1:
            print(f"  ↯ Возрастающая отдача от масштаба (α+β > 1)")
        elif returns_to_scale < 1:
            print(f"  ↘ Убывающая отдача от масштаба (α+β < 1)")
        else:
            print(f"  → Постоянная отдача от масштаба (α+β = 1)")
        '''

        return {
            'country': country_name,
            'A': A,
            'alpha': alpha,
            'beta': beta,
            'intercept': intercept,
            'returns_to_scale': returns_to_scale,
            'r_squared': R_squared,
            'mse': SSR / len(ln_Y),
            'n_obs': len(Y_clean)
        }

    except Exception as e:
        print(f"  ❌ Ошибка при оценке параметров для {country_name}: {e}")
        return None


# Оценка для всех стран
production_params = {}
for country in gdp_data.keys():
    if country in capital_data and country in labor_data:
        print(f"\nАнализ производственной функции для: {country}")

        params = estimate_cobb_douglas_params(
            gdp_data[country],
            capital_data[country],
            labor_data[country],
            country
        )

        if params is not None:
            production_params[country] = params
            print(f"  Успешно оценены параметры для {country}")
            print(
                f"  A={params['A']:.4f}, α={params['alpha']:.4f}, β={params['beta']:.4f}, R²={params['r_squared']:.4f}")
        else:
            print(f"  Не удалось оценить параметры для {country}")
    else:
        print(f"  ❌ Отсутствуют данные о капитале или рабочей силе для {country}")

print(f"\nУспешно оценены параметры для {len(production_params)} стран")


# Модифицированная производственная функция
def f_x(x, L, params):
    """
    Производственная функция Кобба-Дугласа:
    f(K, L) = A * K^α * L^β
    """
    A = params['A']
    alpha = params['alpha']
    beta = params['beta']

    return A * (x ** alpha) * (L ** beta)


# Основные параметры модели
lamb = 0.1  # 10% износ
delta = 0.01  # шаг для управления


def run_model_for_country(country_name, start_year=2000, end_year=2010):
    """Запуск модели для конкретной страны с началом в определенный год"""

    # Проверяем наличие данных
    global mape_gdp
    if country_name not in production_params:
        print(f"❌ Отсутствуют параметры производственной функции для {country_name}")
        return None

    # Получаем данные для страны
    gdp_values = gdp_data[country_name]
    capital_values = capital_data[country_name]
    labor_values = labor_data[country_name]

    start_year_idx = list(years).index(start_year)
    a_0 = capital_values[start_year_idx]  # начальный капитал в выбранном году
    L_0 = labor_values[start_year_idx]  # начальная рабочая сила
    end_year_idx = list(years).index(end_year)

    # Определяем горизонт планирования
    N = end_year - start_year + 1  # от start_year до end_year включительно

    # Создаем массив годов для горизонта планирования
    model_years = list(range(start_year, end_year + 1))
    gam = get_inflation(model_years, country_name)

    print(f"\n{'=' * 70}")
    print(f"Анализ для страны: {country_name}")
    print(f"Начальный год: {start_year}, Горизонт планирования: {N} лет")
    print(f"{'=' * 70}")

    # Параметры производственной функции
    params = production_params[country_name]

    # Получаем траекторию рабочей силы для моделирования
    L_trajectory = []
    for year in model_years:
        if year in years:
            idx = years.index(year)
            L_trajectory.append(labor_values[idx])
        else:
            # Линейная интерполяция, если данных нет
            L_trajectory.append(L_0)

    # Терминальные параметры
    def calculate_terminal_params(discount_rate, last_year_value):
        """Рассчитать терминальные параметры на основе реальных данных"""
        k_term = discount_rate
        b_term = np.log(last_year_value)
        return k_term, b_term

    k_term, b_term = calculate_terminal_params(gam, gdp_values[end_year_idx])

    print(f"Параметры: A={params['A']:.4f}, α={params['alpha']:.4f}, β={params['beta']:.4f}")
    print(f"Отдача от масштаба: α+β={params['returns_to_scale']:.4f}")
    print(f"Начальный капитал ({start_year}): {a_0:.2f} млн.долл.")
    print(f"Начальная рабочая сила ({start_year}): {L_0:.4f} млн.чел.")
    print(f"Инфляция: {(gam * 100):.2f} %")

    a = a_0 * 0.01  # длина интервалов в множестве возможных состояний

    # Функция полезности (учитываем теперь и рабочую силу)
    def utility(p, K, L, params):
        production = f_x(K, L, params)
        c = (1 - p) * production  # потребление
        if c <= 1e-10:
            return -1e10
        return np.log(c)

    # Подготовка вспомогательных массивов
    start_time = time.time()
    rho = np.arange(0, 1 + delta, delta)

    # Оцениваем максимальное возможное значение состояния
    maxim = max(capital_values) * 5
    print(f"Максимальное аналитически возможное значение состояния: {maxim:.2f}")

    A_k = np.arange(0, maxim + a, a)  # массив возможных значений состояния
    print(f"Время подготовки массивов: {time.time() - start_time:.2f} сек.")

    # ПЕРВЫЙ ЭТАП - ОБРАТНЫЙ ХОД БЕЛЛМАНА
    print("\n1. Обратный ход Беллмана...")
    start_time = time.time()

    A = A_k
    U = rho

    # Инициализация массивов
    F_i_all = np.zeros((N, len(A)))  # [время, состояние]
    s_i_all = np.zeros((N, len(A)))  # [время, состояние]

    # Конечное условие
    for j in range(len(A)):
        F_i_all[N - 1, j] = k_term * A[j] + b_term

    # Обратный ход Беллмана
    for i in range(N - 2, -1, -1):  # от N-2 до 0
        current_year = model_years[i]
        current_gam = gam
        current_L = L_trajectory[i]

        for j in range(len(A)):  # по всем состояниям
            current_state = A[j]

            best_value = -np.inf
            best_control = 0

            # Перебор всех возможных управлений
            for t in range(len(U)):
                control = U[t]

                # Текущая полезность (теперь с учетом труда)
                current_utility = utility(control, current_state, current_L, params)

                # Следующее состояние
                next_state = (1 - lamb) * current_state + control * f_x(current_state, current_L, params)

                # Находим ближайшее состояние в сетке
                state_idx = np.argmin(np.abs(A - next_state))

                # Дисконтированное значение с учетом инфляции текущего года
                discounted_utility = current_utility / ((1 + current_gam) ** i)
                future_value = F_i_all[i + 1, state_idx]

                total_value = discounted_utility + future_value

                if total_value > best_value:
                    best_value = total_value
                    best_control = control

            # Сохраняем оптимальное управление и значение функции Беллмана
            s_i_all[i, j] = best_control
            F_i_all[i, j] = best_value

    print(f"Время обратного хода: {time.time() - start_time:.2f} сек.")

    # ВТОРОЙ ЭТАП - ПРЯМОЙ ХОД
    print("\n2. Прямой ход...")
    start_time = time.time()

    # Инициализация массивов
    k_opt = np.zeros(N)  # оптимальные состояния
    p_opt = np.zeros(N)  # оптимальные управления
    gam_actual = np.zeros(N)  # фактическая инфляция по годам
    L_actual = np.zeros(N)  # фактическая рабочая сила по годам

    # Начальное условие
    k_opt[0] = a_0
    gam_actual[0] = gam
    L_actual[0] = L_0

    # Находим оптимальное управление для начального состояния
    start_idx = np.argmin(np.abs(A_k - k_opt[0]))
    p_opt[0] = s_i_all[0, start_idx]

    # Прямой ход: вычисляем оптимальную траекторию
    for i in range(1, N):
        # Вычисляем следующее состояние по динамике системы (учитываем труд)
        production = f_x(k_opt[i - 1], L_actual[i - 1], params)
        k_opt[i] = (1 - lamb) * k_opt[i - 1] + p_opt[i - 1] * production
        gam_actual[i] = gam
        L_actual[i] = L_trajectory[i]

        # Находим оптимальное управление для текущего состояния
        if i < N - 1:
            state_idx = np.argmin(np.abs(A_k - k_opt[i]))
            p_opt[i] = s_i_all[i, state_idx]

    # Для последнего момента берем управление равное последнему
    p_opt[N - 1] = p_opt[N - 2]

    print(f"Время прямого хода: {time.time() - start_time:.2f} сек.")

    # Анализ результатов
    print(f"\n{'=' * 70}")
    print("РЕЗУЛЬТАТЫ:")
    print(f"{'=' * 70}")

    # Сопоставляем с фактическими данными
    actual_gdp_in_period = []
    actual_capital_in_period = []
    actual_labor_in_period = []
    actual_years_in_period = []

    # Находим фактические значения для соответствующих годов
    for year in model_years:
        if year in years:
            idx = years.index(year)
            actual_gdp_in_period.append(gdp_values[idx])
            actual_capital_in_period.append(capital_values[idx])
            actual_labor_in_period.append(labor_values[idx])
            actual_years_in_period.append(year)

    print(f"\nФактические данные ВВП ({country_name}):")
    for year, gdp in zip(actual_years_in_period, actual_gdp_in_period):
        print(f"  {year}: {gdp:.2f} млн.долл.")

    print(f"\nФактические данные капитала ({country_name}):")
    for year, capital in zip(actual_years_in_period, actual_capital_in_period):
        print(f"  {year}: {capital:.2f} млн.долл.")

    print(f"\nФактические данные рабочей силы ({country_name}):")
    for year, labor in zip(actual_years_in_period, actual_labor_in_period):
        print(f"  {year}: {labor:.4f} млн.чел.")

    # Находим оптимальные значения капитала для тех же годов
    optimal_capital_in_period = []
    optimal_gdp_in_period = []
    for year in actual_years_in_period:
        idx = model_years.index(year)
        optimal_capital_in_period.append(k_opt[idx])
        optimal_gdp_in_period.append(f_x(k_opt[idx], L_actual[idx], params))

    print(f"\nОптимальная траектория капитала (модель):")
    for year, capital in zip(actual_years_in_period, optimal_capital_in_period):
        print(f"  {year}: {capital:.2f} млн.долл.")

    print(f"\nОптимальные доли инвестирования (p_opt):")
    for i, year in enumerate(model_years):
        if i % 5 == 0 or i == N - 1:  # выводим каждые 5 лет и последний год
            print(f"  {year}: {p_opt[i]:.3f} (инфляция: {gam_actual[i] * 100:.1f}%)")

    # Вычисление среднегодовых темпов роста
    if len(actual_gdp_in_period) > 1:
        actual_start = actual_gdp_in_period[0]
        actual_end = actual_gdp_in_period[-1]
        years_diff = actual_years_in_period[-1] - actual_years_in_period[0]

        actual_growth = (actual_end - actual_start) / actual_start
        optimal_growth = (k_opt[-1] - k_opt[0]) / k_opt[0]

        print(f"\nСреднегодовые темпы роста:")
        print(f"  Фактические ВВП: {actual_growth * 100:.2f}%")
        print(f"  Оптимальные капитал: {optimal_growth * 100:.2f}%")

        # Ошибка модели для лет, где есть фактические данные
        mape_values_capital = []
        mape_values_gdp = []
        for year in actual_years_in_period:
            idx_actual = years.index(year)
            idx_model = model_years.index(year)
            actual_capital_val = capital_values[idx_actual]
            actual_gdp_val = gdp_values[idx_actual]
            model_capital_val = k_opt[idx_model]
            model_gdp_val = f_x(k_opt[idx_model], L_actual[idx_model], params)

            if actual_capital_val > 0:
                mape_values_capital.append(abs((actual_capital_val - model_capital_val) / actual_capital_val))
            if actual_gdp_val > 0:
                mape_values_gdp.append(abs((actual_gdp_val - model_gdp_val) / actual_gdp_val))

        if mape_values_capital:
            mape_capital = np.mean(mape_values_capital) * 100
            print(f"  Средняя ошибка модели капитала (MAPE): {mape_capital:.2f}%")

        if mape_values_gdp:
            mape_gdp = np.mean(mape_values_gdp) * 100
            print(f"  Средняя ошибка модели ВВП (MAPE): {mape_gdp:.2f}%")

        # ВИЗУАЛИЗАЦИЯ
        # Создаем 3D график производственной функции (отдельная фигура)
        print("\nСоздание 3D графика производственной функции...")

        # Создаем отдельную фигуру для 3D графика
        fig3d = plt.figure(figsize=(12, 9))
        ax3d = fig3d.add_subplot(111, projection='3d')

        # Создаем сетку для K и L
        K_min = min(actual_capital_in_period) * 0.8
        K_max = max(actual_capital_in_period) * 1.2
        L_min = min(actual_labor_in_period) * 0.8
        L_max = max(actual_labor_in_period) * 1.2

        K_range = np.linspace(K_min, K_max, 20)
        L_range = np.linspace(L_min, L_max, 20)
        K_grid, L_grid = np.meshgrid(K_range, L_range)

        # Вычисляем производство
        Y_grid = f_x(K_grid, L_grid, params)

        # Построение поверхности
        surf = ax3d.plot_surface(K_grid, L_grid, Y_grid, cmap='viridis', alpha=0.7, edgecolor='none')

        # Фактические точки
        ax3d.scatter(actual_capital_in_period, actual_labor_in_period, actual_gdp_in_period,
                     color='red', s=50, label='Фактические точки', depthshade=False)

        ax3d.set_xlabel('Капитал, млн.долл.')
        ax3d.set_ylabel('Рабочая сила, млн.чел.')
        ax3d.set_zlabel('ВВП, млн.долл.')
        ax3d.set_title(f'{country_name}: Y = {params["A"]:.2f} * K^{params["alpha"]:.2f} * L^{params["beta"]:.2f}')
        ax3d.legend()
        fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5, label='ВВП, млн.долл.')

        # Сохраняем 3D график
        plt.savefig(f'images/{country_name}_{start_year}_{end_year}_production_function_3d.png', dpi=300, bbox_inches='tight')
        print(f"3D график сохранен как: images/{country_name}_{start_year}_{end_year}_production_function_3d.png")

        # Показываем 3D график
        plt.show()
        plt.close(fig3d)

        # Создаем основной график с 6 подграфиками
        print("\nСоздание основного графика с результатами...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Динамическая оптимизация для {country_name} (с {start_year} по {end_year})', fontsize=16,
                     fontweight='bold')

        # 1. Фактический vs оптимальный капитал
        axes[0, 0].plot(actual_years_in_period, actual_capital_in_period, 'b-o',
                        label='Фактический капитал', linewidth=2, markersize=6)
        axes[0, 0].plot(model_years, k_opt, 'r--s', label='Оптимальный капитал (модель)',
                        linewidth=2, markersize=4, alpha=0.7)
        axes[0, 0].set_xlabel('Год', fontsize=11)
        axes[0, 0].set_ylabel('Капитал, млн.долл.', fontsize=11)
        axes[0, 0].set_title('Сравнение капиталов', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='both', labelsize=10)

        # 2. Оптимальная доля инвестирования и инфляция
        ax2 = axes[0, 1]
        ax2.plot(model_years, p_opt, 'g-o', linewidth=2, markersize=4, label='Доля инвестирования')
        ax2.set_xlabel('Год', fontsize=11)
        ax2.set_ylabel('Доля инвестирования, p', color='green', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_title('Оптимальное управление и инфляция', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=10)

        '''
        ax2_infl = ax2.twinx()
        ax2_infl.plot(model_years, gam_actual * 100, 'm--', linewidth=1.5,
                      label='Инфляция (%)', alpha=0.7)
        ax2_infl.set_ylabel('Инфляция, %', color='m', fontsize=11)
        ax2_infl.tick_params(axis='y', labelcolor='m')
        '''

        # Объединяем легенды
        lines1, labels1 = ax2.get_legend_handles_labels()
        #lines2, labels2 = ax2_infl.get_legend_handles_labels()
        #ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        ax2.legend(lines1, labels1, loc='upper left', fontsize=9)

        # 3. Рабочая сила
        axes[0, 2].plot(model_years, L_actual, 'm-o', label='Рабочая сила',
                        linewidth=2, markersize=4, alpha=0.7)
        axes[0, 2].set_xlabel('Год', fontsize=11)
        axes[0, 2].set_ylabel('Рабочая сила, млн.чел.', fontsize=11)
        axes[0, 2].set_title('Динамика рабочей силы', fontsize=13, fontweight='bold')
        axes[0, 2].legend(fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].tick_params(axis='both', labelsize=10)

        # 4. Отклонение модели от фактических данных ВВП
        axes[1, 0].plot(actual_years_in_period, actual_gdp_in_period, 'b-o',
                        label='Фактический ВВП', linewidth=2, markersize=6)
        axes[1, 0].plot(model_years, optimal_gdp_in_period, 'r--s', label='Оптимальный ВВП (модель)',
                        linewidth=2, markersize=4, alpha=0.7)
        axes[1, 0].set_xlabel('Год', fontsize=11)
        axes[1, 0].set_ylabel('ВВП, млн.долл.', fontsize=11)
        axes[1, 0].set_title('Сравнение ВВП', fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='both', labelsize=10)

        # 5. Доля факторов в производстве
        factors = ['Капитал (α)', 'Труд (β)', 'Сумма (α+β)']
        values = [params['alpha'], params['beta'], params['returns_to_scale']]
        colors = ['blue', 'green', 'red']

        bars = axes[1, 1].bar(factors, values, color=colors, edgecolor='black', linewidth=1.5)
        axes[1, 1].set_ylabel('Эластичность', fontsize=11)
        axes[1, 1].set_title('Вклад факторов в производство', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='both', labelsize=10)

        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        # Добавляем горизонтальную линию на уровне 1 для постоянной отдачи от масштаба
        axes[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # 6. Статистика модели
        axes[1, 2].axis('off')
        stats_text = (
            f"Параметры модели:\n"
            f"A = {params['A']:.4f}\n"
            f"α = {params['alpha']:.4f}\n"
            f"β = {params['beta']:.4f}\n"
            f"α+β = {params['returns_to_scale']:.4f}\n"
            f"R² = {params['r_squared']:.4f}\n"
            f"MSE = {params['mse']:.4f}\n"
            f"Наблюдения: {params['n_obs']}\n"
            f"\nРезультаты оптимизации:\n"
            f"Средняя p: {np.mean(p_opt):.3f}\n"
            f"Рост капитала: {optimal_growth * 100:.2f}%\n"
            f"Ошибка ВВП: {mape_gdp:.2f}%"
        )

        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                        verticalalignment='center', horizontalalignment='left',
                        transform=axes[1, 2].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Сохраняем основной график
        plt.savefig(f'images/{country_name}_{start_year}_{end_year}_analysis_cobb_douglas.png',
                    dpi=300, bbox_inches='tight')
        print(f"Основной график сохранен как: images/{country_name}_{start_year}_{end_year}_analysis_cobb_douglas.png")

        # Показываем основной график
        plt.show()
        plt.close(fig)

    return {
        'country': country_name,
        'start_year': start_year,
        'model_years': model_years,
        'actual_years': actual_years_in_period,
        'actual_gdp': actual_gdp_in_period,
        'actual_capital': actual_capital_in_period,
        'actual_labor': actual_labor_in_period,
        'optimal_capital': k_opt,
        'optimal_gdp': optimal_gdp_in_period,
        'optimal_investment': p_opt,
        'inflation': gam_actual,
        'labor': L_actual,
        'params': params
    }


# Запуск анализа для США с учетом труда
run_model_for_country('Индия', start_year=1996, end_year=2001)