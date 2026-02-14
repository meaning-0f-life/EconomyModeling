#import math
import numpy as np
import time
import matplotlib.pyplot as plt
#from datetime import datetime
from scipy import stats
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
    'США': 'USA',
    'Венгрия': 'HUN',
    'Польша': 'POL'
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

# НОВЫЕ ДАННЫЕ: фиксированные инвестиции и потребление основного капитала
df_gross_fixed_capital = pd.read_csv('data/GrossFixedCapitalFormation.csv', skiprows=3)
df_consumption_capital = pd.read_csv('data/ConsumptionOfFixedCapital.csv', skiprows=3)

# Словарь для результатов
capital_data = {}
gdp_data = {}
inflation_data_by_country = {}
population_data = {}
employment_data = {}
gross_fixed_capital_data = {}  # НОВОЕ: данные о доле инвестиций в ВВП
consumption_capital_data = {}  # НОВОЕ: данные о потреблении основного капитала

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

# Фильтруем по нужным странам для валовых фиксированных инвестиций (% от ВВП)
for rus_name, code in countries_needed.items():
    # Находим строку для страны и индикатора валовых фиксированных инвестиций
    country_row = df_gross_fixed_capital[(df_gross_fixed_capital['Country Code'] == code) &
                                         (df_gross_fixed_capital['Indicator Code'] == 'NE.GDI.FTOT.ZS')]

    if not country_row.empty:
        # Извлекаем данные за нужные годы
        data = []
        for year in years_needed:
            if year in df_gross_fixed_capital.columns:
                value = country_row[year].values[0]
                if pd.notna(value):
                    # Данные уже в процентах
                    data.append(float(value) / 100)  # переводим % в доли (0-1)
                else:
                    data.append(0.0)
            else:
                data.append(0.0)

        gross_fixed_capital_data[rus_name] = data

# Фильтруем по нужным странам для потребления основного капитала
for rus_name, code in countries_needed.items():
    # Это абсолютное значение в текущих долларах США, а не процент от ВВП
    country_row = df_consumption_capital[(df_consumption_capital['Country Code'] == code) &
                                         (df_consumption_capital['Indicator Code'] == 'NY.ADJ.DKAP.CD')]

    if not country_row.empty:
        # Извлекаем данные за нужные годы
        data = []
        for year in years_needed:
            if year in df_consumption_capital.columns:
                value = country_row[year].values[0]
                if pd.notna(value):
                    # Преобразуем в млн долларов (абсолютное значение)
                    data.append(float(value) / 1000000)
                else:
                    data.append(0.0)
            else:
                data.append(0.0)

        consumption_capital_data[rus_name] = data

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

def estimate_cobb_douglas_params(gdp_values, capital_values, labor_values, country_name, lamb=0.2):
    """
    Оценка параметров Y = A * K^α
    через линеаризацию: ln(Y) = ln(A) + α*ln(K)
    """
    # Проверяем, что данные корректны
    if len(gdp_values) != len(capital_values):
        raise ValueError("Длины массивов не совпадают")

    # Убираем нулевые и отрицательные значения
    Y = np.array(gdp_values)
    K = np.array(capital_values)
    L = np.array(labor_values)

    mask = (Y > 0) & (K > 0) & (L > 0)

    Y_clean = Y[mask]
    K_clean = K[mask]
    L_clean = L[mask]
    Y_clean = Y_clean / L_clean # ВВП на рабочего (удельный ВВП)
    K_clean = K_clean / L_clean # капитал на рабочего (удельный капитал)

    # Берем логарифмы
    ln_Y = np.log(Y_clean)
    ln_K = np.log(K_clean)

    # Линейная регрессия: ln_Y = intercept + slope * ln_K
    slope, intercept, r_value, p_value, std_err = stats.linregress(ln_K, ln_Y)

    # Параметры модели
    alpha = slope  # эластичность по капиталу
    A = np.exp(intercept)  # технологический параметр

    # Статистика
    r_squared = r_value ** 2

    # Дополнительные расчеты
    residuals = ln_Y - (intercept + slope * ln_K)
    mse = np.mean(residuals ** 2)

    '''
    print(f"\n{'=' * 60}")
    print(f"ОЦЕНКА ПАРАМЕТРОВ ДЛЯ: {country_name}")
    print(f"{'=' * 60}")
    print(f"Уравнение: ln(Y) = {intercept:.4f} + {slope:.4f} * ln(K)")
    print(f"Параметры:")
    print(f"  A (технология) = {A:.4f}")
    print(f"  α (эластичность по капиталу) = {alpha:.4f}")
    print(f"Статистика:")
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  Стандартная ошибка = {std_err:.4f}")
    print(f"  MSE = {mse:.4f}")

    # Проверка экономической осмысленности
    if alpha < 0:
        print(f"  ⚠️  ВНИМАНИЕ: α отрицательный!")
    elif alpha > 1:
        print(f"  ⚠️  ВНИМАНИЕ: α > 1 (возрастающая отдача)")
    elif 0.2 <= alpha <= 0.4:
        print(f"  ✓ α в реалистичном диапазоне (0.2-0.4)")
    else:
        print(f"  ⚠️  α вне типичного диапазона")
    '''

    return {
        'country': country_name,
        'A': A,
        'alpha': alpha,
        'intercept': intercept,
        'slope': slope,
        'r_squared': r_squared,
        'p_value': p_value,
        'mse': mse
    }

# Основные параметры модели
delta = 0.01  # шаг для управления

# Оценка для всех стран
production_params = {}
for country in gdp_data.keys():
    if country in capital_data:
        params = estimate_cobb_douglas_params(
            gdp_data[country],
            capital_data[country],
            labor_data[country],
            country
        )
        production_params[country] = params

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


# Используем реальные данные для доли инвестиций
def calculate_actual_investment_share_from_data(gross_fixed_capital_data):
    """
    Использует реальные данные о доле валовых фиксированных инвестиций в ВВП
    из таблицы GrossFixedCapitalFormation.csv

    Параметры:
    gross_fixed_capital_data -- словарь с данными о доле инвестиций в ВВП (% в долях)

    Возвращает:
    actual_investment_share -- словарь с фактическими долями инвестиций по годам
    """
    actual_investment_share = {}

    for country in gross_fixed_capital_data.keys():
        # Берем значения напрямую из данных (уже преобразованы в доли)
        p_vals = gross_fixed_capital_data[country]

        # Ограничиваем разумными пределами (0-1)
        p_vals = [max(0, min(p, 1)) for p in p_vals]

        actual_investment_share[country] = p_vals

    return actual_investment_share


# ИСПРАВЛЕНО: Используем реальные данные для доли инвестиций
actual_gfcf_share = calculate_actual_investment_share_from_data(gross_fixed_capital_data)


# Расчет коэффициента выбытия фондов по годам
def calculate_depreciation_rate_by_year(capital_data, consumption_capital_data, years_list):
    """
    Вычисляет коэффициент выбытия удельных фондов (λ) для каждого года
    на основе данных о потреблении основного капитала

    λ_t = Потребление основного капитала_t / Капитал_t

    Параметры:
    capital_data -- словарь с данными по капиталу (млн долл.)
    consumption_capital_data -- словарь с данными о потреблении основного капитала (млн долл.)
    years_list -- список годов

    Возвращает:
    depreciation_rates_by_country -- словарь с коэффициентами выбытия по годам для каждой страны
    """
    depreciation_rates_by_country = {}

    for country in capital_data.keys():
        if country in consumption_capital_data:
            capital_vals = capital_data[country]
            consumption_vals = consumption_capital_data[country]  # абсолютное значение в млн долл.

            depreciation_rates = []
            years_for_country = []


            for i, year in enumerate(reversed(years_list)):
                if i < len(capital_vals) and i < len(consumption_vals):
                    capital = capital_vals[i]
                    consumption_absolute = consumption_vals[i]  # уже в млн долл.

                    global lamb

                    if capital > 0 and consumption_absolute > 0:
                        # Вычисляем коэффициент выбытия: потребление капитала / капитал
                        lamb = consumption_absolute / capital

                    # Ограничиваем разумными пределами (0-0.5)
                    #lamb = max(0, min(lamb, 0.5))

                    depreciation_rates.append(lamb)
                    #years_for_country.append(year)
                    #depreciation_rates.insert(0, lamb)
                    years_for_country.insert(0, year)
                #else:
                    # Если вышли за пределы данных
                #    depreciation_rates.append(0.1)
                #    years_for_country.append(year)

            # Создаем словарь для быстрого доступа по годам
            depreciation_dict = {year: rate for year, rate in zip(years_for_country, depreciation_rates)}
            depreciation_rates_by_country[country] = depreciation_dict


    return depreciation_rates_by_country

# Рассчитываем коэффициенты выбытия для всех стран
depreciation_rates_by_country = calculate_depreciation_rate_by_year(
    capital_data, consumption_capital_data, list(years)
)
#print(depreciation_rates_by_country)

def calculate_actual_investment_share(capital_data, gdp_data, depreciation_rates_by_country, years_list):
    """
    Вычисляет фактическую долю инвестиций: p_actual = инвестиции / ВВП

    Инвестиции вычисляются из динамики капитала:
    Iₜ = Kₜ₊₁ - (1-λₜ)Kₜ

    Параметры:
    capital_data -- словарь с данными по капиталу (млн долл.)
    gdp_data -- словарь с данными по ВВП (млн долл.)
    depreciation_rates_by_country -- словарь с коэффициентами выбытия по годам
    years_list -- список годов в данных

    Возвращает:
    actual_investment_share -- словарь с фактическими долями инвестиций по годам
    """
    actual_investment_share = {}

    for country in capital_data.keys():
        if country in gdp_data and country in depreciation_rates_by_country:
            depreciation_dict = depreciation_rates_by_country[country]  # словарь {год: λ}
            capital_vals = capital_data[country]
            gdp_vals = gdp_data[country]

            if len(capital_vals) != len(gdp_vals):
                print(f"Ошибка: разная длина данных для {country}")
                continue

            p_vals = []
            years_for_p = []  # для отладки: сохраняем соответствующие годы

            # Для первого года не можем вычислить инвестиции
            p_vals.append(0.0)
            years_for_p.append(years_list[0])

            for i in range(1, len(capital_vals)):
                # Получаем год для индекса i-1 (предыдущий год)
                year = years_list[i - 1]

                if gdp_vals[i] > 0:
                    # Получаем λ для соответствующего года
                    if year in depreciation_dict:
                        lamb = depreciation_dict[year]
                    else:
                        # Если нет данных для этого года, используем значение по умолчанию
                        lamb = 0.1
                        print(f"Предупреждение: нет данных о λ для {country} в {year}, используем λ={lamb}")

                    # Вычисляем инвестиции из динамики капитала
                    investment = capital_vals[i] - (1 - lamb) * capital_vals[i - 1]

                    # Доля инвестиций в ВВП
                    p_actual = investment / gdp_vals[i]

                    # Ограничиваем разумными пределами
                    p_actual = max(-1.0, min(p_actual, 1.0))  # расширяем диапазон для отрицательных инвестиций

                    p_vals.append(p_actual)
                    years_for_p.append(year)
                else:
                    p_vals.append(0.0)
                    years_for_p.append(year)

            # Выравниваем длины: последний год не имеет следующего для расчета
            # Оставляем как есть или добавляем последний год с 0
            if len(p_vals) < len(years_list):
                p_vals.append(0.0)
                years_for_p.append(years_list[-1])

            # Отладка: выводим несколько значений
            # print(f"{country}: годы={years_for_p[:5]}, p={p_vals[:5]}")

            actual_investment_share[country] = p_vals

    return actual_investment_share

# Вызывайте функцию с передачей списка годов
actual_investment_share = calculate_actual_investment_share(
    capital_data,
    gdp_data,
    depreciation_rates_by_country,
    list(years)  # передаем список годов
)



def run_model_for_country(country_name, start_year=2000, end_year=2010):
    """Запуск модели для конкретной страны с началом в определенный год"""

    # Получаем данные для страны
    gdp_values = gdp_data[country_name]
    labor_values = labor_data[country_name]
    specific_gdp_values = [x / y for x, y in zip(gdp_values, labor_values)]
    capital_values = capital_data[country_name]
    specific_capital_values = [x / y for x, y in zip(capital_values, labor_values)]
    start_year_idx = list(years).index(start_year)
    a_0 = specific_capital_values[start_year_idx]  # начальный удельный капитал в выбранном году
    end_year_idx = list(years).index(end_year)
    country = country_name

    # Определяем горизонт планирования
    N = end_year - start_year + 1  # от start_year до end_year включительно

    # Создаем массив годов для горизонта планирования
    model_years = list(range(start_year, end_year + 1))

    # Получаем фактические управления (реальные данные об инвестициях)
    actual_gfcf = actual_gfcf_share[country_name]

    # Получаем коэффициенты выбытия по годам
    depreciation_dict = depreciation_rates_by_country[country_name]

    # Берем только те годы, которые входят в период моделирования
    actual_p_in_period = []
    actual_years_for_p = []
    actual_lamb_in_period = []
    actual_gfcf_in_period = []
    actual_p_share = actual_investment_share[country_name]

    for year in model_years:
        if year in years:
            idx = years.index(year)
            actual_gfcf_in_period.append(actual_gfcf[idx])
            actual_p_in_period.append(actual_p_share[idx])
            actual_years_for_p.append(year)

            # Получаем коэффициент выбытия для конкретного года
            if year in depreciation_dict:
                actual_lamb_in_period.append(depreciation_dict[year])
            else:
                # Если нет данных для конкретного года, используем среднее
                actual_lamb_in_period.append(0.1)

    print(f"\n{'=' * 70}")
    print(f"Анализ для страны: {country_name}")
    print(f"Начальный год: {start_year}, Горизонт планирования: {N} лет")
    print(f"{'=' * 70}")

    # Параметры производственной функции
    params = production_params[country_name]
    k = params['A']  # параметр A
    b = params['alpha']  # параметр alpha

    # Функция производительности (Кобба-Дугласа)
    def f_x(x):
        return k * (x ** b)

    '''
    # Терминальный параметр
    def calculate_marginal_product_of_capital(country_name, data_years, capital_data,
                                              production_params):
        """
        Вычисляет k_term как среднюю предельную производительность капитала
        с использованием оцененной производственной функции
        """

        if country_name not in production_params:
            print(f"Внимание: нет оцененных параметров для {country_name}")
            return 0.0

        # Получаем параметры производственной функции
        params = production_params[country_name]
        A = params['A']
        alpha = params['alpha']

        # Определяем диапазон лет для анализа
        start_year = data_years[0]
        end_year = data_years[-1]

        # Собираем данные по капиталу за указанный период
        capital_values = []
        years_available = []

        for year in range(start_year, end_year + 1):
            if year in data_years:
                idx = list(data_years).index(year)
                if idx < len(capital_data[country_name]):
                    capital = capital_data[country_name][idx]
                    if capital > 0:
                        capital_values.append(capital)
                        years_available.append(year)

        if len(capital_values) < 2:
            return 0.0

        # Расчет предельной производительности капитала
        mpk_values = []

        for i, year in enumerate(years_available):
            capital = capital_values[i]

            # Предельная производительность капитала для функции Кобба-Дугласа:
            # MPK = α * A * K^(α-1)
            if capital > 0:
                mpk = alpha * A * (capital ** (alpha - 1))
                mpk_values.append(mpk)

        if not mpk_values:
            return 0.0

        # Вычисляем среднее значение
        k_term_mpk = np.mean(mpk_values)

        return k_term_mpk
    '''

    def utility(p, f, sigma=2.0):
        c = (1 - p) * f
        if c <= 0:
            return -np.inf
        if abs(sigma - 1.0) < 1e-6:
            return np.log(c)
        return (c ** (1 - sigma) - 1) / (1 - sigma)

    gam = get_inflation(model_years, country)

    print(f"Параметры: A={k:.2f}, alpha={b:.2f}")
    print(f"Средний коэффициент выбытия удельных фондов за период: {np.mean(actual_lamb_in_period) * 100:.2f} %")
    print(f"Начальный удельный капитал ({start_year}): a_0={a_0:.2f} долл.")
    print(f"Средняя инфляция за период: gam={(gam * 100):.2f} %")
    print(f"Выбывания удельных фондов lamb:")
    for year in model_years:
        print(f"{year}: {depreciation_dict[year]:.2f}")

    a = a_0 * 0.01  # длина интервалов в множестве возможных состояний

    # Подготовка вспомогательных массивов
    start_time = time.time()
    rho = np.arange(0, 1 + delta, delta)

    maxim = max(specific_capital_values) * 4
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

    # Конечное условие (последний год)
    for j in range(len(A)):
        current_capital = A[j]
        k_term = utility(0, current_capital)
        F_i_all[N - 1, j] = k_term

    # Обратный ход Беллмана с переменным коэффициентом выбытия
    for i in range(N - 2, -1, -1):  # от N-2 до 0
        current_year = model_years[i]
        current_gam = gam
        current_lamb = actual_lamb_in_period[i]  # берем коэффициент выбытия для конкретного года

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

                # Следующее состояние с учетом коэффициента выбытия для года i
                next_state = (1 - current_lamb) * current_state + control * f_val

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
    lamb_actual = np.zeros(N)  # фактические коэффициенты выбытия по годам

    # Начальное условие
    k_opt[0] = a_0
    gam_actual[0] = gam
    lamb_actual[0] = actual_lamb_in_period[0]

    # Находим оптимальное управление для начального состояния
    start_idx = np.argmin(np.abs(A_k - k_opt[0]))
    p_opt[0] = s_i_all[0, start_idx]

    # Прямой ход: вычисляем оптимальную траекторию с переменным коэффициентом выбытия
    for i in range(1, N):
        # Берем коэффициент выбытия для текущего года
        current_lamb = actual_lamb_in_period[i-1]
        lamb_actual[i] = current_lamb

        # Вычисляем следующее состояние по динамике системы
        k_opt[i] = (1 - current_lamb) * k_opt[i - 1] + p_opt[i - 1] * f_x(k_opt[i - 1])
        gam_actual[i] = gam

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

    # Сопоставляем с фактическими данными ВВП
    actual_specific_gdp_in_period = []
    actual_specific_capital_in_period = []
    actual_years_in_period = []

    # Находим фактические значения ВВП для соответствующих годов
    for year in model_years:
        if year in years:
            idx = years.index(year)
            actual_specific_gdp_in_period.append(specific_gdp_values[idx])
            actual_specific_capital_in_period.append(specific_capital_values[idx])
            actual_years_in_period.append(year)

    print(f"\nФактические данные удельного ВВП ({country_name}):")
    for year, gdp in zip(actual_years_in_period, actual_specific_gdp_in_period):
        print(f"  {year}: {gdp:.2f} долл.")

    print(f"\nФактические данные удельного капитала ({country_name}):")
    for year, capital in zip(actual_years_in_period, actual_specific_capital_in_period):
        print(f"  {year}: {capital:.2f} долл.")

    # Находим оптимальные значения капитала для тех же годов
    optimal_capital_in_period = []
    for year in actual_years_in_period:
        idx = model_years.index(year)
        optimal_capital_in_period.append(k_opt[idx])

    print(f"\nОптимальная траектория капитала (модель):")
    for year, gdp in zip(actual_years_in_period, optimal_capital_in_period):
        print(f"  {year}: {gdp:.2f} млн.долл.")

    print(f"\nОптимальные доли инвестирования (p_opt):")
    for i, year in enumerate(actual_years_in_period):
        print(f"  {year}: {p_opt[i]:.3f} (λ={lamb_actual[i]:.3f}, инфляция: {gam_actual[i] * 100:.1f}%)")

    print(f"\nРеальные доли инвестирования (actual_p_in_period):")
    for i, year in enumerate(actual_years_for_p):
        print(f"  {year}: {actual_p_in_period[i]:.3f} (λ={lamb_actual[i]:.3f}, инфляция: {gam_actual[i] * 100:.1f}%)")
    print(f"\nGFCF (actual_gfcf_in_period):")
    for i, year in enumerate(actual_years_for_p):
        print( f"  {year}: {actual_gfcf_in_period[i]:.3f} (λ={lamb_actual[i]:.3f}, инфляция: {gam_actual[i] * 100:.1f}%)")

    # Вычисление среднегодовых темпов роста
    if len(actual_specific_gdp_in_period) > 1:
        actual_start = actual_specific_gdp_in_period[0]
        actual_end = actual_specific_gdp_in_period[-1]

        actual_growth = (actual_end - actual_start) / actual_start
        optimal_growth = (k_opt[-1] - k_opt[0]) / k_opt[0]

        print(f"\nСреднегодовые темпы роста:")
        print(f"  Фактические ({actual_years_in_period[0]}-{actual_years_in_period[-1]}): {actual_growth * 100:.2f}%")
        print(f"  Оптимальные ({model_years[0]}-{model_years[-1]}): {optimal_growth * 100:.2f}%")

        # Ошибка модели для лет, где есть фактические данные
        mape_values = []
        for year in actual_years_in_period:
            idx_actual = years.index(year)
            idx_model = model_years.index(year)
            actual_val = capital_values[idx_actual]
            model_val = k_opt[idx_model]
            if actual_val > 0:
                mape_values.append(abs((actual_val - model_val) / actual_val))

        if mape_values:
            mape = np.mean(mape_values) * 100
            print(f"  Средняя ошибка модели (MAPE): {mape:.2f}%")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Динамическая оптимизация для {country_name} (с {start_year} по {end_year})', fontsize=16)

    # 1. Фактический vs оптимальный капитал
    axes[0, 0].plot(actual_years_in_period, actual_specific_capital_in_period, 'b-o',
                    label='Фактический удельный капитал', linewidth=2, markersize=6)
    axes[0, 0].plot(model_years, k_opt, 'r--s', label='Оптимальный удельный капитал (модель)',
                    linewidth=2, markersize=4, alpha=0.7)
    axes[0, 0].set_xlabel('Год')
    axes[0, 0].set_ylabel('Удельный капитал, долл.')
    axes[0, 0].set_title(f'Сравнение капиталов')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Оптимальная доля инвестирования и фактические данные
    ax2 = axes[0, 1]

    # Оптимальные управления (из модели)
    ax2.plot(model_years, p_opt, 'r-o', linewidth=2,
             label='Оптимальная доля инвестирования (модель)', markersize=6)

    # Фактические управления (из рассчётов на исторических данных)
    ax2.plot(actual_years_for_p, actual_p_in_period, 'b-8', linewidth=2,
             label='Фактическая доля инвестирования (данные)', markersize=6, alpha=0.7)

    # Фактические управления (из исторических данных GrossFixedCapitalFormation)
    ax2.plot(actual_years_for_p, actual_gfcf_in_period, 'm--s', linewidth=2,
             label='Gross Fixed Capital Formation', markersize=6, alpha=0.7)

    ax2.set_xlabel('Год')
    ax2.set_ylabel('Доля инвестирования, p', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_title('Сравнение оптимальных и фактических управлений')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. Производственная функция
    x_range = np.linspace(min(actual_specific_capital_in_period) * 0.8, max(actual_specific_capital_in_period) * 1.2, 100)
    y_range = f_x(x_range)
    axes[1, 0].plot(x_range, y_range, 'r-', linewidth=2)
    axes[1, 0].scatter(actual_specific_capital_in_period, actual_specific_gdp_in_period,
                       color='blue', s=50, label='Фактические точки')
    axes[1, 0].set_xlabel('Удельный капитал, долл.')
    axes[1, 0].set_ylabel('Удельное производство (ВВП), долл.')
    axes[1, 0].set_title(f'Производственная функция: f(k) = {k:.2f} * k^{b:.2f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Коэффициент выбытия капитала по годам
    axes[1, 1].plot(model_years, lamb_actual * 100, 'g-o', linewidth=2,
                    label='Коэффициент выбытия λ (%)', markersize=6)
    axes[1, 1].set_xlabel('Год')
    axes[1, 1].set_ylabel('Коэффициент выбытия λ, %', color='green')
    axes[1, 1].tick_params(axis='y', labelcolor='green')
    axes[1, 1].set_title(f'Динамика коэффициента выбытия капитала')
    axes[1, 1].legend(loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)

    '''
    # Добавляем вторую ось для инфляции
    ax4_infl = axes[1, 1].twinx()
    ax4_infl.plot(model_years, gam_actual * 100, 'm--', linewidth=1.5,
                  label='Инфляция (%)', alpha=0.7)
    ax4_infl.set_ylabel('Инфляция, %', color='m')
    ax4_infl.tick_params(axis='y', labelcolor='m')
    '''

    # Объединяем легенды
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    #lines2, labels2 = ax4_infl.get_legend_handles_labels()
    #axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    axes[1, 1].legend(lines1, labels1, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'images/{country_name}_{start_year}_{end_year}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'country': country_name,
        'start_year': start_year,
        'model_years': model_years,
        'actual_years': actual_years_in_period,
        'actual_gdp': actual_specific_gdp_in_period,
        'optimal_capital': k_opt,
        'optimal_investment': p_opt,
        'inflation': gam_actual,
        'depreciation_rates': lamb_actual,
        'params': params
    }


# Запуск анализа
run_model_for_country('Венгрия', start_year=1996, end_year=2001)