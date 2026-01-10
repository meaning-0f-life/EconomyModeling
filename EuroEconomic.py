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
    'США': 'USA'
}
'''
    'Афганистан': 'AFG',
    'Ангола': 'AGO',
    'Албания': 'ALB',
    'Андорра': 'AND',
    'Арабский мир': 'ARB',
    'Объединенные Арабские Эмираты': 'ARE',
    'Аргентина': 'ARG',
    'Армения': 'ARM',
    'Американское Самоа': 'ASM',
    'Антигуа и Барбуда': 'ATG',
    'Австралия': 'AUS',
    'Австрия': 'AUT',
    'Азербайджан': 'AZE',
    'Бурунди': 'BDI',
    'Бельгия': 'BEL',
    'Бенин': 'BEN',
    'Буркина-Фасо': 'BFA',
    'Бангладеш': 'BGD',
    'Болгария': 'BGR',
    'Бахрейн': 'BHR',
    'Багамы': 'BHS',
    'Босния и Герцеговина': 'BIH',
    'Беларусь': 'BLR',
    'Белиз': 'BLZ',
    'Бермуды': 'BMU',
    'Боливия': 'BOL',
    'Бразилия': 'BRA',
    'Барбадос': 'BRB',
    'Бруней': 'BRN',
    'Бутан': 'BTN',
    'Ботсвана': 'BWA',
    'Центральноафриканская Республика': 'CAF',
    'Канада': 'CAN',
    'Швейцария': 'CHE',
    'Канал Острова': 'CHI',
    'Чили': 'CHL',
    'Китай': 'CHN',
    "Кот-д'Ивуар": 'CIV',
    'Камерун': 'CMR',
    'Конго, Дем. Респ.': 'COD',
    'Конго, Респ.': 'COG',
    'Колумбия': 'COL',
    'Коморы': 'COM',
    'Кабо-Верде': 'CPV',
    'Коста-Рика': 'CRI',
    'Куба': 'CUB',
    'Кюрасао': 'CUW',
    'Каймановы острова': 'CYM',
    'Кипр': 'CYP',
    'Чехия': 'CZE',
    'Германия': 'DEU',
    'Джибути': 'DJI',
    'Доминика': 'DMA',
    'Дания': 'DNK',
    'Доминиканская Республика': 'DOM',
    'Алжир': 'DZA',
    'Эквадор': 'ECU',
    'Египет': 'EGY',
    'Эритрея': 'ERI',
    'Испания': 'ESP',
    'Эстония': 'EST',
    'Эфиопия': 'ETH',
    'Финляндия': 'FIN',
    'Фиджи': 'FJI',
    'Франция': 'FRA',
    'Фарерские острова': 'FRO',
    'Микронезия': 'FSM',
    'Габон': 'GAB',
    'Великобритания': 'GBR',
    'Грузия': 'GEO',
    'Гана': 'GHA',
    'Гибралтар': 'GIB',
    'Гвинея': 'GIN',
    'Гамбия': 'GMB',
    'Гвинея-Бисау': 'GNB',
    'Экваториальная Гвинея': 'GNQ',
    'Греция': 'GRC',
    'Гренада': 'GRD',
    'Гренландия': 'GRL',
    'Гватемала': 'GTM',
    'Гуам': 'GUM',
    'Гайана': 'GUY',
    'Гонконг': 'HKG',
    'Гондурас': 'HND',
    'Хорватия': 'HRV',
    'Гаити': 'HTI',
    'Венгрия': 'HUN',
    'Индонезия': 'IDN',
    'Индия': 'IND',
    'Ирландия': 'IRL',
    'Иран': 'IRN',
    'Ирак': 'IRQ',
    'Исландия': 'ISL',
    'Израиль': 'ISR',
    'Италия': 'ITA',
    'Ямайка': 'JAM',
    'Иордания': 'JOR',
    'Япония': 'JPN',
    'Казахстан': 'KAZ',
    'Кения': 'KEN',
    'Кыргызстан': 'KGZ',
    'Камбоджа': 'KHM',
    'Кирибати': 'KIR',
    'Сент-Китс и Невис': 'KNA',
    'Корея, Южная': 'KOR',
    'Кувейт': 'KWT',
    'Лаос': 'LAO',
    'Ливан': 'LBN',
    'Сент-Люсия': 'LCA',
    'Лихтенштейн': 'LIE',
    'Шри-Ланка': 'LKA',
    'Либерия': 'LBR',
    'Лесото': 'LSO',
    'Литва': 'LTU',
    'Люксембург': 'LUX',
    'Латвия': 'LVA',
    'Македония': 'MKD',
    'Марокко': 'MAR',
    'Молдова': 'MDA',
    'Мадагаскар': 'MDG',
    'Мексика': 'MEX',
    'Маршалловы Острова': 'MHL',
    'Северная Македония': 'MKD',
    'Мали': 'MLI',
    'Мальта': 'MLT',
    'Мьянма': 'MMR',
    'Черногория': 'MNE',
    'Монголия': 'MNG',
    'Мозамбик': 'MOZ',
    'Мавритания': 'MRT',
    'Маврикий': 'MUS',
    'Малави': 'MWI',
    'Малайзия': 'MYS',
    'Намибия': 'NAM',
    'Новая Каледония': 'NCL',
    'Нигер': 'NER',
    'Нигерия': 'NGA',
    'Никарагуа': 'NIC',
    'Нидерланды': 'NLD',
    'Норвегия': 'NOR',
    'Непал': 'NPL',
    'Науру': 'NRU',
    'Новая Зеландия': 'NZL',
    'Оман': 'OMN',
    'Пакистан': 'PAK',
    'Панама': 'PAN',
    'Перу': 'PER',
    'Филиппины': 'PHL',
    'Палау': 'PLW',
    'Папуа - Новая Гвинея': 'PNG',
    'Польша': 'POL',
    'Пуэрто-Рико': 'PRI',
    'Корея, Северная': 'PRK',
    'Португалия': 'PRT',
    'Парагвай': 'PRY',
    'Палестина': 'PSE',
    'Французская Полинезия': 'PYF',
    'Катар': 'QAT',
    'Румыния': 'ROU',
    'Россия': 'RUS',
    'Руанда': 'RWA',
    'Саудовская Аравия': 'SAU',
    'Судан': 'SDN',
    'Сенегал': 'SEN',
    'Сингапур': 'SGP',
    'Соломоновы Острова': 'SLB',
    'Сьерра-Леоне': 'SLE',
    'Сальвадор': 'SLV',
    'Сан-Марино': 'SMR',
    'Сомали': 'SOM',
    'Сербия': 'SRB',
    'Южный Судан': 'SSD',
    'Сан-Томе и Принсипи': 'STP',
    'Суринам': 'SUR',
    'Словакия': 'SVK',
    'Словения': 'SVN',
    'Швеция': 'SWE',
    'Эсватини': 'SWZ',
    'Сейшельские Острова': 'SYC',
    'Сирия': 'SYR',
    'Чад': 'TCD',
    'Того': 'TGO',
    'Таиланд': 'THA',
    'Таджикистан': 'TJK',
    'Туркменистан': 'TKM',
    'Тимор-Лесте': 'TLS',
    'Тонга': 'TON',
    'Тринидад и Тобаго': 'TTO',
    'Тунис': 'TUN',
    'Турция': 'TUR',
    'Тувалу': 'TUV',
    'Тайвань': 'TWN',
    'Танзания': 'TZA',
    'Уганда': 'UGA',
    'Украина': 'UKR',
    'Уругвай': 'URY',
    'США': 'USA',
    'Узбекистан': 'UZB',
    'Сент-Винсент и Гренадины': 'VCT',
    'Венесуэла': 'VEN',
    'Вьетнам': 'VNM',
    'Вануату': 'VUT',
    'Самоа': 'WSM',
    'Косово': 'XKX',
    'Йемен': 'YEM',
    'Южная Африка': 'ZAF',
    'Замбия': 'ZMB',
    'Зимбабве': 'ZWE'
'''

# Годы, которые нам нужны
years_needed = [str(year) for year in range(1995, 2025)]
years = range(1995, 2025)

# Читаем CSV файлы (все данные взяты из https://data.worldbank.org)
df_capital = pd.read_csv('data/CapitalData.csv', skiprows=3)
df_gdp = pd.read_csv('data/GDP_Data.csv', skiprows=3)
df_inflation = pd.read_csv('data/InflationData.csv', skiprows=4)

# Словарь для результатов
capital_data = {}
gdp_data = {}
inflation_data_by_country = {}

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

'''
# Выводим результат
print("# Данные по капиталу (млн. долл.) за 1995–2024 годы")
print("capital_data = {")
for country, data in capital_data.items():
    print(f"    '{country}': {data},")
print("}")

print("\n# Данные по ВВП (млн. долл.) за 1995–2024 годы")
print("gdp_data = {")
for country, data in gdp_data.items():
    print(f"    '{country}': {data},")
print("}")

print("\n# Данные по инфляции (доли) за 1995–2024 годы")
print("inflation_data_by_country = {")
for country, data in inflation_data_by_country.items():
    print(f"    '{country}': {data},")
print("}")
'''

def estimate_cobb_douglas_params(gdp_values, capital_values, country_name):
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

    mask = (Y > 0) & (K > 0)

    Y_clean = Y[mask]
    K_clean = K[mask]

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


# Оценка для всех стран
production_params = {}
for country in gdp_data.keys():
    if country in capital_data:
        params = estimate_cobb_douglas_params(
            gdp_data[country],
            capital_data[country],
            country
        )
        production_params[country] = params

# Функция для получения инфляции по году (в долях)
def get_inflation(year, country):
    return inflation_data_by_country[country][year] / 100  # переводим % в доли

# Основные параметры модели
lamb = 0.05  # 5% износ
delta = 0.01  # шаг для управления

def run_model_for_country(country_name, start_year=2000, end_year=2010):
    """Запуск модели для конкретной страны с началом в определенный год"""

    # Получаем данные для страны
    gdp_values = gdp_data[country_name]
    capital_values = capital_data[country_name]
    start_year_idx = list(years).index(start_year)
    a_0 = capital_values[start_year_idx]  # начальный капитал в выбранном году
    end_year_idx = list(years).index(end_year)

    # Определяем горизонт планирования
    N = end_year - start_year + 1  # от start_year до end_year включительно

    # Создаем массив годов для горизонта планирования
    model_years = list(range(start_year, end_year + 1))

    print(f"\n{'=' * 70}")
    print(f"Анализ для страны: {country_name}")
    print(f"Начальный год: {start_year}, Горизонт планирования: {N} лет")
    print(f"{'=' * 70}")

    # Параметры производственной функции
    params = production_params[country_name]
    k = params['A']  # параметр A
    b = params['alpha']  # параметр alpha

    # Терминальные параметры
    def calculate_terminal_params(discount_rate, last_year_value):
        """Рассчитать терминальные параметры на основе реальных данных"""
        # Например: терминальное значение = (1 - норма дисконта) * последнее значение
        k_term = discount_rate
        b_term = np.log(last_year_value)  # логарифмическая полезность
        return k_term, b_term

    k_term, b_term = calculate_terminal_params(get_inflation(end_year, country), gdp_values[end_year_idx])

    '''
    # Практическое правило для Кобба-Дугласа
    def practical_max_control(alpha, time_horizon):
        if time_horizon <= 30:
            return min(alpha * 1.2, 0.8)
        elif time_horizon <= 100:
            return min(alpha * 1.0, 0.7)
        else:
            return min(alpha * 0.8, 0.6)

    # Используем аналитическую оценку
    #p_maximize = practical_max_control(b, N)
    p_maximize = 1 # !!!!!!!!!!!!!!!!
    '''

    print(f"Параметры: A={k:.2f}, alpha={b:.2f}")
    print(f"Начальный капитал ({start_year}): {a_0:.2f} млн.долл.")

    a = a_0 * 0.01  # длина интервалов в множестве возможных состояний

    # Функция производительности (Кобба-Дугласа)
    def f_x(x):
        return k * (x ** b)

    norm_coef = 1

    # Функция полезности
    def utility(p, f):
        c = (1 - p) * f  # потребление
        if c <= 1e-10:
            return -1e10
        return np.log(c/norm_coef)

    # Подготовка вспомогательных массивов
    start_time = time.time()
    rho = np.arange(0, 1 + delta, delta)

    # Оцениваем максимальное возможное значение состояния
    k_max_trajectory = np.zeros(N)
    k_max_trajectory[0] = a_0

    # Для оценки максимума используем среднюю инфляцию
    #avg_inflation = np.mean([get_inflation(year) for year in model_years])

    '''
    for i in range(1, N):
        k_max_trajectory[i] = (1 - lamb) * k_max_trajectory[i - 1] + p_maximize * f_x(k_max_trajectory[i - 1])
    '''

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

    # Конечное условие (2013 год)
    for j in range(len(A)):
        F_i_all[N - 1, j] = k_term * A[j] + b_term

    # Обратный ход Беллмана
    for i in range(N - 2, -1, -1):  # от N-2 до 0
        current_year = model_years[i]
        current_gam = get_inflation(current_year, country)

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

    # Начальное условие
    k_opt[0] = a_0
    gam_actual[0] = get_inflation(model_years[0], country)

    # Находим оптимальное управление для начального состояния
    start_idx = np.argmin(np.abs(A_k - k_opt[0]))
    p_opt[0] = s_i_all[0, start_idx]

    # Прямой ход: вычисляем оптимальную траекторию
    for i in range(1, N):
        # Вычисляем следующее состояние по динамике системы
        k_opt[i] = (1 - lamb) * k_opt[i - 1] + p_opt[i - 1] * f_x(k_opt[i - 1])
        gam_actual[i] = get_inflation(model_years[i], country)

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
    actual_gdp_in_period = []
    actual_capital_in_period = []
    actual_years_in_period = []

    # Находим фактические значения ВВП для соответствующих годов
    for year in model_years:
        if year in years:
            idx = years.index(year)
            actual_gdp_in_period.append(gdp_values[idx])
            actual_capital_in_period.append(capital_values[idx])
            actual_years_in_period.append(year)

    print(f"\nФактические данные ВВП ({country_name}):")
    for year, gdp in zip(actual_years_in_period, actual_gdp_in_period):
        print(f"  {year}: {gdp:.2f} млн.долл.")

    print(f"\nФактические данные капитала ({country_name}):")
    for year, capital in zip(actual_years_in_period, actual_capital_in_period):
        print(f"  {year}: {capital:.2f} млн.долл.")

    # Находим оптимальные значения капитала для тех же годов
    optimal_capital_in_period = []
    for year in actual_years_in_period:
        idx = model_years.index(year)
        optimal_capital_in_period.append(k_opt[idx])

    print(f"\nОптимальная траектория капитала (модель):")
    for year, gdp in zip(actual_years_in_period, optimal_capital_in_period):
        print(f"  {year}: {gdp:.2f} млн.долл.")

    print(f"\nОптимальные доли инвестирования (p_opt):")
    for i, year in enumerate(model_years):
        if i % 5 == 0 or i == N - 1:  # выводим каждые 5 лет и последний год
            print(f"  {year}: {p_opt[i]:.3f} (инфляция: {gam_actual[i] * 100:.1f}%)")

    # Вычисление среднегодовых темпов роста
    if len(actual_gdp_in_period) > 1:
        actual_start = actual_gdp_in_period[0]
        actual_end = actual_gdp_in_period[-1]

        actual_growth = (actual_end / actual_start) / actual_start
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
    axes[0, 0].plot(actual_years_in_period, actual_capital_in_period, 'b-o',
                    label='Фактический капитал', linewidth=2, markersize=6)
    axes[0, 0].plot(model_years, k_opt, 'r--s', label='Оптимальный капитал (модель)',
                    linewidth=2, markersize=4, alpha=0.7)
    axes[0, 0].set_xlabel('Год')
    axes[0, 0].set_ylabel('Капитал, млн.долл.')
    axes[0, 0].set_title(f'Сравнение капиталов')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Оптимальная доля инвестирования и инфляция
    ax2 = axes[0, 1]
    ax2.plot(model_years, p_opt, 'g-o', linewidth=2, label='Доля инвестирования')
    ax2.set_xlabel('Год')
    ax2.set_ylabel('Доля инвестирования, p', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_title('Оптимальное управление и инфляция')
    ax2.grid(True, alpha=0.3)

    ax2_infl = ax2.twinx()
    ax2_infl.plot(model_years, gam_actual * 100, 'm--', linewidth=1.5,
                  label='Инфляция (%)', alpha=0.7)
    ax2_infl.set_ylabel('Инфляция, %', color='m')
    ax2_infl.tick_params(axis='y', labelcolor='m')

    # Объединяем легенды
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_infl.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 3. Производственная функция
    x_range = np.linspace(min(actual_capital_in_period) * 0.8, max(actual_capital_in_period) * 1.2, 100)
    y_range = f_x(x_range)
    axes[1, 0].plot(x_range, y_range, 'b-', linewidth=2)
    axes[1, 0].scatter(actual_capital_in_period, actual_gdp_in_period,
                       color='red', s=50, label='Фактические точки')
    axes[1, 0].set_xlabel('Капитал, млн.долл.')
    axes[1, 0].set_ylabel('Производство (ВВП), млн.долл.')
    axes[1, 0].set_title(f'Производственная функция: f(k) = {k:.2f} * k^{b:.2f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Отклонение модели от фактических данных
    axes[1, 1].plot(actual_years_in_period, actual_gdp_in_period, 'b-o',
                    label='Фактический ВВП', linewidth=2, markersize=6)
    axes[1, 1].plot(model_years, f_x(k_opt), 'r--s', label='Оптимальный ВВП (модель)',
                    linewidth=2, markersize=4, alpha=0.7)
    axes[1, 1].set_xlabel('Год')
    axes[1, 1].set_ylabel('ВВП, млн.долл.')
    axes[1, 1].set_title(f'Сравнение ВВП')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'images/{country_name}_{start_year}_{end_year}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'country': country_name,
        'start_year': start_year,
        'model_years': model_years,
        'actual_years': actual_years_in_period,
        'actual_gdp': actual_gdp_in_period,
        'optimal_capital': k_opt,
        'optimal_investment': p_opt,
        'inflation': gam_actual,
        'params': params
    }

'''
# Сравнительный анализ для разных начальных точек
def compare_start_years(country_name):
    """Сравнение результатов для разных начальных годов"""

    results = {}

    print(f"\n{'=' * 80}")
    print(f"СРАВНИТЕЛЬНЫЙ АНАЛИЗ ДЛЯ {country_name.upper()}")
    print(f"{'=' * 80}")

    for i, start_year in enumerate(years[:-1]):  # все кроме 2013
        print(f"\n{'─' * 40}")
        print(f"Начальный год: {start_year}")
        print(f"{'─' * 40}")

        result = run_model_for_country(country_name, i)
        results[start_year] = result

        # Краткая сводка
        actual_gdp = result['actual_gdp']
        optimal_gdp = []
        for year in result['actual_years']:
            idx = result['model_years'].index(year)
            optimal_gdp.append(result['optimal_capital'][idx])

        if len(actual_gdp) > 1:
            actual_growth = (actual_gdp[-1] / actual_gdp[0]) ** (
                        1 / (result['actual_years'][-1] - result['actual_years'][0])) - 1
            optimal_growth = (result['optimal_capital'][-1] / result['optimal_capital'][0]) ** (
                        1 / (len(result['model_years']) - 1)) - 1

            print(f"  Рост фактический: {actual_growth * 100:.2f}%")
            print(f"  Рост оптимальный: {optimal_growth * 100:.2f}%")
            print(f"  Средняя доля инвестирования: {np.mean(result['optimal_investment']):.3f}")

    return results


# Запуск анализа
print("ДОСТУПНЫЕ СТРАНЫ:")
for i, country in enumerate(gdp_data.keys(), 1):
    print(f"{i}. {country}")

# Выберите страну для анализа
selected_country = 'Польша'

if selected_country in gdp_data:
    # Анализ для разных начальных точек
    results = compare_start_years(selected_country)

    # Сводный график для разных начальных точек
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Сравнение оптимальных траекторий для {selected_country} (разные начальные годы)', fontsize=16)

    colors = plt.cm.tab10(np.linspace(0, 1, len(years[:-1])))

    # 1. Оптимальные траектории капитала
    ax1 = axes[0, 0]
    for idx, start_year in enumerate(years[:-1]):
        result = results[start_year]
        ax1.plot(result['model_years'], result['optimal_capital'],
                 color=colors[idx], linewidth=2, alpha=0.7,
                 label=f'Начало {start_year}')
    ax1.set_xlabel('Год')
    ax1.set_ylabel('Оптимальный капитал, млн.долл.')
    ax1.set_title('Оптимальные траектории капитала')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Оптимальные доли инвестирования
    ax2 = axes[0, 1]
    for idx, start_year in enumerate(years[:-1]):
        result = results[start_year]
        ax2.plot(result['model_years'], result['optimal_investment'],
                 color=colors[idx], linewidth=1.5, alpha=0.7,
                 label=f'Начало {start_year}')
    ax2.set_xlabel('Год')
    ax2.set_ylabel('Доля инвестирования')
    ax2.set_title('Оптимальные доли инвестирования')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Фактический ВВП
    ax3 = axes[1, 0]
    actual_years = years
    actual_gdp = gdp_data[selected_country]
    ax3.plot(actual_years, actual_gdp, 'ko-', linewidth=2, markersize=6)
    ax3.set_xlabel('Год')
    ax3.set_ylabel('Фактический ВВП, млн.долл.')
    ax3.set_title(f'Фактический ВВП {selected_country}')
    ax3.grid(True, alpha=0.3)

    # 4. Инфляция
    ax4 = axes[1, 1]
    all_years = list(range(1992, 2014))
    inflation_vals = [get_inflation(year) * 100 for year in all_years]
    ax4.plot(all_years, inflation_vals, 'b-', linewidth=2)
    ax4.fill_between(all_years, 0, inflation_vals, alpha=0.3)
    ax4.set_xlabel('Год')
    ax4.set_ylabel('Инфляция, %')
    ax4.set_title('Динамика инфляции (HICP)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{selected_country}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\n" + "=" * 80)
print("КРАТКИЕ ВЫВОДЫ:")
print("=" * 80)
print("1. Модель динамической оптимизации позволяет найти оптимальную траекторию")
print("   инвестирования для каждого начального года.")
print("2. Горизонт планирования зависит от начального года: N = 2013 - start_year + 1")
print("3. Инфляция учитывается динамически по годам согласно данным HICP.")
print("4. Результаты показывают, как оптимальная стратегия зависит от:")
print("   - Начального уровня капитала (ВВП)")
print("   - Параметров производственной функции")
print("   - Динамики инфляции")
print("   - Горизонта планирования")
'''

run_model_for_country('Индия', start_year=1996, end_year=2001)

'''
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
'''