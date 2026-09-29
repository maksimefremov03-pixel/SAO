import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import json
import mmap
import re
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import warnings
import time
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Конфигурация
STATION_NAME = 'BC840'
BASE_DIR = f"D:\\PycharmProjects\\SAO\\{STATION_NAME}\\"
CACHE_DIR = os.path.join(BASE_DIR, 'binary_cache')


def freq_to_period_days(omega, points_per_day):
    """Преобразование normalized частоты в период в днях"""
    if omega <= 1e-10:
        return float('inf')
    period_points = 1.0 / omega
    period_days = period_points / points_per_day
    return period_days


class BinaryDataReader:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        self.stats = self._load_stats()
        self.years_index = self._load_years_index()

    def _load_stats(self):
        """Загрузка статистики кэша"""
        stats_file = os.path.join(self.cache_dir, 'cache_stats.json')
        if not os.path.exists(stats_file):
            raise FileNotFoundError(
                f"Кэш не найден! Сначала запустите create_binary_cache.py\n"
                f"Ожидаемый файл: {stats_file}"
            )

        with open(stats_file, 'r') as f:
            return json.load(f)

    def _load_years_index(self):
        """Загрузка индекса годов"""
        index_file = os.path.join(self.cache_dir, 'years_index.json')
        if not os.path.exists(index_file):
            return None

        with open(index_file, 'r') as f:
            return json.load(f)

    def get_available_years(self):
        """Получение списка доступных годов"""
        if self.years_index:
            return sorted([int(y) for y in self.years_index.keys()])
        return []

    def get_info(self):
        """Получение информации о кэше"""
        return {
            'total_points': self.stats['total_points'],
            'years': self.get_available_years(),
            'date_range': self.stats['date_range'],
            'value_stats': self.stats['value_stats']
        }

    def _resample_to_uniform(self, dates, values, target_step_minutes):
        """Интерполяция данных на равномерную временную сетку"""
        timestamps = np.array([d.timestamp() for d in dates])

        start_time = timestamps[0]
        end_time = timestamps[-1]

        n_points = int((end_time - start_time) / (target_step_minutes * 60)) + 1
        uniform_timestamps = np.linspace(start_time, end_time, n_points)
        uniform_dates = [datetime.fromtimestamp(ts) for ts in uniform_timestamps]

        f = interp1d(timestamps, values, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        uniform_values = f(uniform_timestamps)

        compression_ratio = len(dates) / n_points
        print(f"  Сжатие: {len(dates)} → {n_points} точек (коэф. {compression_ratio:.1f})")

        return uniform_dates, uniform_values

    def load_data_with_filters(self, selected_years=None, filter_params=None):
        """Загрузка данных с автоматической интерполяцией"""
        start_time = time.time()

        if filter_params is None:
            filter_params = {}

        if selected_years is None:
            selected_years = self.get_available_years()

        print(f"\nЗагрузка данных из кэша...")
        print(f"  Годы: {selected_years}")

        all_timestamps = []
        all_values = []

        for year in selected_years:
            year_file = os.path.join(self.cache_dir, f'year_{year}.bin')
            if not os.path.exists(year_file):
                print(f"  Предупреждение: год {year} не найден в кэше")
                continue

            year_data = np.fromfile(year_file, dtype=np.float64)
            year_data = year_data.reshape(-1, 2)

            all_timestamps.append(year_data[:, 0])
            all_values.append(year_data[:, 1])

        if not all_timestamps:
            return [], np.array([])

        timestamps = np.concatenate(all_timestamps)
        values = np.concatenate(all_values)

        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        values = values[sort_idx]

        mask = np.ones(len(timestamps), dtype=bool)
        dts = [datetime.fromtimestamp(ts) for ts in timestamps]

        if 'start_day' in filter_params or 'end_day' in filter_params:
            days = np.array([dt.timetuple().tm_yday for dt in dts])
            if 'start_day' in filter_params:
                mask &= (days >= filter_params['start_day'])
            if 'end_day' in filter_params:
                mask &= (days <= filter_params['end_day'])

        if 'start_time' in filter_params or 'end_time' in filter_params:
            hours = np.array([dt.hour + dt.minute / 60.0 + dt.second / 3600.0 for dt in dts])
            if 'start_time' in filter_params:
                mask &= (hours >= filter_params['start_time'])
            if 'end_time' in filter_params:
                mask &= (hours <= filter_params['end_time'])

        timestamps = timestamps[mask]
        values = values[mask]
        dts = [dt for dt, m in zip(dts, mask) if m]

        # Автоматическая интерполяция
        if len(dts) > 1:
            time_diffs = []
            for i in range(1, min(100, len(dts))):
                diff = (dts[i] - dts[i-1]).total_seconds() / 60
                if diff > 0:
                    time_diffs.append(diff)

            if time_diffs:
                time_diffs = np.array(time_diffs)
                median_step = np.median(time_diffs)
                std_step = np.std(time_diffs)

                if std_step > median_step * 0.2:
                    from collections import Counter
                    rounded_steps = np.round(time_diffs / 5) * 5
                    step_counts = Counter(rounded_steps)
                    most_common = step_counts.most_common(1)[0][0]

                    standard_steps = [5, 10, 15, 30, 60]
                    optimal_step = min(standard_steps, key=lambda x: abs(x - most_common))
                    optimal_step = max(5, min(60, optimal_step))

                    print(f"\n  Обнаружен неравномерный шаг (медиана={median_step:.1f} мин, σ={std_step:.1f})")
                    print(f"  Автоматическая интерполяция на {optimal_step} мин...")
                    dts, values = self._resample_to_uniform(dts, values, optimal_step)

        elapsed = time.time() - start_time
        print(f"  Загружено {len(dts)} точек за {elapsed:.3f} сек")

        return dts, values


def get_years_input(available_years):
    """Ввод годов для обучения с поддержкой диапазона"""
    print("\n" + "=" * 50)
    print("ВВОД ГОДОВ ДЛЯ ОБУЧЕНИЯ")
    print("=" * 50)

    print(f"\nДоступные годы: {available_years}")

    while True:
        years_input = input("\nВведите годы для обучения (например: 2015 2020 или 2005-2020): ").strip()
        if not years_input:
            print("Пожалуйста, введите хотя бы один год")
            continue

        try:
            years = []
            parts = re.split(r'[,\s]+', years_input)

            for part in parts:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    years.extend(range(start, end + 1))
                elif part:
                    years.append(int(part))

            valid_years = [y for y in years if y in available_years]

            if not valid_years:
                print(f"Ни один из указанных годов не найден. Доступные: {available_years}")
                continue

            invalid_years = [y for y in years if y not in available_years]
            if invalid_years:
                print(f"Предупреждение: годы {invalid_years} не найдены и будут проигнорированы")

            valid_years = sorted(list(set(valid_years)))
            print(f"Выбраны годы для обучения: {valid_years}")
            return valid_years

        except ValueError:
            print("Ошибка: неправильный формат ввода")


def get_filters_input():
    """Ввод фильтров с значениями по умолчанию"""
    print("\n" + "=" * 50)
    print("НАСТРОЙКА ФИЛЬТРОВ")
    print("=" * 50)

    params = {}

    use_day = input("Фильтровать по дням года? (y/n): ").lower() == 'y'
    if use_day:
        try:
            start_input = input("  Начальный день (1-365, Enter=1): ").strip()
            params['start_day'] = int(start_input) if start_input else 1

            end_input = input("  Конечный день (1-365, Enter=365): ").strip()
            params['end_day'] = int(end_input) if end_input else 365
        except ValueError:
            print("Ошибка ввода дней, используются значения по умолчанию")
            params['start_day'] = 1
            params['end_day'] = 365

    use_time = input("Фильтровать по времени суток? (y/n): ").lower() == 'y'
    if use_time:
        try:
            start_input = input("  Начальное время в часах (0-24, Enter для пропуска): ").strip()
            if start_input:
                params['start_time'] = float(start_input)

            end_input = input("  Конечное время в часах (0-24, Enter для пропуска): ").strip()
            if end_input:
                params['end_time'] = float(end_input)
        except ValueError:
            print("Ошибка ввода времени")

    return params


def plot_data(datetimes, foF2_values, selected_years, filter_info=""):
    """Построение графика с отображением UTC-6 (сдвиг на -6 часов)"""
    if not datetimes:
        return

    # Сдвигаем время на -6 часов для отображения как UTC-6
    shifted_datetimes = [dt - timedelta(hours=6) for dt in datetimes]

    plt.figure(figsize=(14, 6))

    years = np.array([dt.year for dt in datetimes])  # исходные годы для статистики
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(years))))

    unique_years = sorted(set(years))
    for i, year in enumerate(unique_years):
        mask = years == year
        year_dates = [shifted_datetimes[j] for j, m in enumerate(mask) if m]
        year_values = foF2_values[mask]

        color_idx = i % len(colors)
        linewidth = 1.5 if year in selected_years else 0.5
        alpha = 0.8 if year in selected_years else 0.3

        plt.plot(year_dates, year_values, color=colors[color_idx],
                 linewidth=linewidth, alpha=alpha, label=str(year))

    plt.xlabel('Дата (местное время UTC-6)')
    plt.ylabel('foF2 (MHz)')

    title = f'Данные foF2 (станция {STATION_NAME})'
    if filter_info:
        title += f'\n{filter_info}'
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', ncol=3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'foF2_data.png', dpi=150)
    plt.show()

    print("\nСтатистика по выбранным годам:")
    for year in selected_years:
        mask = years == year
        if np.any(mask):
            values = foF2_values[mask]
            print(f"  {year}: {len(values)} точек, среднее={np.mean(values):.3f}, "
                  f"мин={np.min(values):.3f}, макс={np.max(values):.3f}")


def vmd_physical(signal, data_time_step, dates):
    """
    VMD разложение с физической интерпретацией компонент.
    Частоты фиксированы для каждой компоненты.
    Автоматический выбор K в зависимости от длины данных.
    """
    T = len(signal)

    # Приводим к четной длине
    if T % 2 != 0:
        T = T - 1
        signal = signal[:T]
        dates = dates[:T]

    # Нормализация
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if signal_std < 0.1:
        signal_std = 0.5
    signal_norm = (signal - signal_mean) / signal_std

    # Время и частоты
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - 1 / T

    # FFT исходного сигнала
    f_hat = np.fft.fft(signal_norm)
    f_hat_shifted = np.fft.fftshift(f_hat)

    # Вычисляем реальное количество лет данных
    time_span_seconds = dates[-1].timestamp() - dates[0].timestamp()
    time_span_days = time_span_seconds / (24 * 3600)
    years_of_data = time_span_days / 365.25

    points_per_day = 24 * 60 // data_time_step
    points_per_year = points_per_day * 365

    print(f"\n  Временной анализ:")
    print(f"    Шаг дискретизации: {data_time_step} мин")
    print(f"    Точек в сутках: {points_per_day}")
    print(f"    Временной диапазон: {time_span_days:.1f} дней ({years_of_data:.1f} лет)")
    print(f"    Длина сигнала: {T} точек")

    # Определяем компоненты в зависимости от длины данных
    # ВСЕГДА добавляем высокочастотную компоненту для шума
    if years_of_data >= 11:
        K = 5
        component_names = ['11-летний цикл', 'Годовая', 'Сезонная', 'Суточная', 'Высокочастотная']
        expected_periods = [11 * 365, 365, 90, 1, 0.25]  # 0.25 дня = 6 часов
    elif years_of_data >= 3:
        K = 4
        component_names = ['Годовая', 'Сезонная', 'Суточная', 'Высокочастотная']
        expected_periods = [365, 90, 1, 0.25]
    elif years_of_data >= 1:
        K = 3
        component_names = ['Сезонная', 'Суточная', 'Высокочастотная']
        expected_periods = [90, 1, 0.25]
    else:
        K = 2
        component_names = ['Суточная', 'Высокочастотная']
        expected_periods = [1, 0.25]

    # Параметры VMD
    if years_of_data >= 11:
        alpha = 1000
    elif years_of_data >= 5:
        alpha = 800
    elif years_of_data >= 3:
        alpha = 500
    else:
        alpha = 300

    tau = 0.0
    tol = 1e-6

    # ФИКСИРОВАННЫЕ ЦЕЛЕВЫЕ ЧАСТОТЫ (не обновляются)
    fixed_freqs = []
    for period_days in expected_periods:
        period_points = period_days * points_per_day
        freq = 1.0 / period_points
        fixed_freqs.append(freq)

    fixed_freqs = np.array(sorted(fixed_freqs))

    # Нормализуем частоты к диапазону [0, 0.5]
    max_freq = 0.5
    fixed_freqs = fixed_freqs / (2.0 * max_freq)
    fixed_freqs = np.clip(fixed_freqs, 1e-6, 0.49)

    print(f"\n  Выбраны параметры VMD:")
    print(f"    K = {K} (на основе {years_of_data:.1f} лет данных)")
    print(f"    alpha = {alpha}")
    print(f"    Компоненты: {component_names}")
    print(f"    Фиксированные частоты: {fixed_freqs}")

    # Инициализация мод
    u_hat = np.zeros((T, K), dtype=complex)
    lambda_hat = np.zeros(T, dtype=complex)
    omega = fixed_freqs.copy()

    N = 300
    n = 0
    uDiff = tol + 1

    u_hat_old = u_hat.copy()

    print(f"    VMD итерации (фиксированные частоты): ", end="")

    while uDiff > tol and n < N:
        for k in range(K):
            sum_uk = np.sum(u_hat[:, np.arange(K) != k], axis=1)
            denominator = 1.0 + alpha * (freqs - omega[k]) ** 2
            u_hat[:, k] = (f_hat_shifted - sum_uk - lambda_hat / 2.0) / denominator

        residual_hat = f_hat_shifted - np.sum(u_hat, axis=1)
        lambda_hat = lambda_hat + tau * residual_hat

        uDiff = np.sum(np.abs(u_hat - u_hat_old) ** 2) / (np.sum(np.abs(u_hat_old) ** 2) + 1e-10)

        u_hat_old = u_hat.copy()
        n += 1
        if n % 50 == 0:
            print(f"{n} ", end="")

    print(f"выполнено за {n} итераций")

    # Обратное преобразование
    u = np.zeros((T, K))
    for k in range(K):
        u_shifted = np.fft.ifftshift(u_hat[:, k])
        u[:, k] = np.real(np.fft.ifft(u_shifted))

    # Возвращаем к исходному масштабу
    u = u * signal_std

    # Распределяем среднее пропорционально амплитудам
    component_amplitudes = np.std(u, axis=0)
    if np.sum(component_amplitudes) > 0:
        mean_distribution = signal_mean * component_amplitudes / np.sum(component_amplitudes)
        for k in range(K):
            u[:, k] += mean_distribution[k]

    # Вычисляем периоды
    actual_periods = [freq_to_period_days(omega[k], points_per_day) for k in range(K)]

    print(f"\n  Результаты:")
    for k in range(K):
        expected = expected_periods[k]
        actual = actual_periods[k]
        ratio = actual / expected if expected > 0 else 0

        if 0.8 < ratio < 1.25:
            status = "✓✓ Отлично"
        elif 0.5 < ratio < 2:
            status = "✓ Хорошо"
        else:
            status = "✗ Плохо"

        if expected < 1:
            expected_str = f"{expected * 24:.0f} ч"
            actual_str = f"{actual * 24:.0f} ч"
        else:
            expected_str = f"{expected:.0f} дн"
            actual_str = f"{actual:.0f} дн"

        print(f"    {status}: {component_names[k]} - ожидалось {expected_str}, получено {actual_str}")

    return u, omega, T, alpha, component_names, expected_periods, points_per_day


def predict_next_vmd(datetimes, foF2_values, train_info, test_info):
    """Прогнозирование на основе VMD разложения с физическими компонентами"""

    print(f"\nПрогнозирование...")

    years = np.array([dt.year for dt in datetimes])
    day_of_year = np.array([dt.timetuple().tm_yday for dt in datetimes])

    train_year_start, train_year_end, train_start, train_end = train_info
    test_year_start, test_year_end, test_start, test_end = test_info

    print(f"\nПериод обучения: годы {train_year_start}-{train_year_end}, дни {train_start}-{train_end}")
    print(f"Период прогноза: годы {test_year_start}-{test_year_end}, дни {test_start}-{test_end}")

    # Сбор обучающих данных
    train_dates = []
    train_values = []

    for year in range(train_year_start, train_year_end + 1):
        train_mask = (years == year) & (day_of_year >= train_start) & (day_of_year <= train_end)
        year_dates = [dt for dt, m in zip(datetimes, train_mask) if m]
        year_values = foF2_values[train_mask]

        if len(year_values) > 0:
            train_dates.extend(year_dates)
            train_values.extend(year_values)
            print(f"  {year}: {len(year_values)} точек")

    # Сбор тестовых данных
    test_dates = []
    test_values = []

    for year in range(test_year_start, test_year_end + 1):
        test_mask = (years == year) & (day_of_year >= test_start) & (day_of_year <= test_end)
        year_dates = [dt for dt, m in zip(datetimes, test_mask) if m]
        year_values = foF2_values[test_mask]

        if len(year_values) > 0:
            test_dates.extend(year_dates)
            test_values.extend(year_values)
            print(f"  Тестовый {year}: {len(year_values)} точек")

    print(f"\nВсего обучающих точек: {len(train_values)}")
    print(f"Всего тестовых точек: {len(test_values)}")

    if len(train_values) < 30:
        print("Слишком мало данных для обучения")
        return None, None, None

    # Сортировка
    sorted_idx = np.argsort([d.timestamp() for d in train_dates])
    train_dates = [train_dates[i] for i in sorted_idx]
    train_values = np.array([train_values[i] for i in sorted_idx])

    if len(test_dates) > 0:
        sorted_idx = np.argsort([d.timestamp() for d in test_dates])
        test_dates = [test_dates[i] for i in sorted_idx]
        test_values = np.array([test_values[i] for i in sorted_idx])

    # Определение временного шага
    if len(train_dates) > 1:
        time_diffs = []
        for i in range(1, min(100, len(train_dates))):
            diff = (train_dates[i] - train_dates[i - 1]).total_seconds() / 60
            if diff > 0:
                time_diffs.append(diff)
        data_time_step = int(np.median(time_diffs)) if time_diffs else 15
        print(f"  Временной шаг данных: {data_time_step} минут")
    else:
        data_time_step = 15

    points_per_day = 24 * 60 // data_time_step
    test_duration_days = (test_end - test_start + 1) * (test_year_end - test_year_start + 1)

    forecast_type = "yearly" if test_duration_days > 10 else "daily"

    # Генерация дат прогноза
    forecast_dates = []
    for year in range(test_year_start, test_year_end + 1):
        start_date = datetime(year, 1, 1) + timedelta(days=test_start - 1)
        end_date = datetime(year, 1, 1) + timedelta(days=test_end - 1)
        current_date = start_date
        while current_date <= end_date + timedelta(days=1) - timedelta(minutes=data_time_step):
            forecast_dates.append(current_date)
            current_date += timedelta(minutes=data_time_step)

    print(f"Сгенерировано {len(forecast_dates)} точек для прогноза")

    # VMD разложение с физическими компонентами
    print("\nВыполняем VMD разложение...")

    # Выполняем VMD разложение
    components, omega, processed_T, alpha, component_names, expected_periods, points_per_day = vmd_physical(
        train_values, data_time_step, train_dates
    )

    if components is None:
        return None, None, None

    # Обрезаем до обработанной длины
    train_values = train_values[:processed_T]
    train_dates = train_dates[:processed_T]

    K = components.shape[1]

    print(f"\nИтоговое разложение на {K} компонент:")
    for k in range(K):
        period = freq_to_period_days(omega[k], points_per_day)
        if period < 10:
            period_str = f"{period * 24:.1f} часов"
        elif period < 365:
            period_str = f"{period:.1f} дней"
        else:
            period_str = f"{period / 365:.1f} лет"
        print(f"  Компонента {k + 1}: {component_names[k]}, частота={omega[k]:.6f} (период ~ {period_str})")

    # Визуализация компонент
    colors = plt.cm.tab10(np.linspace(0, 1, K))

    if K == 1:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6))
        axes_list = [axes[0], axes[1]]
    else:
        fig, axes = plt.subplots(K + 1, 1, figsize=(14, 3 * (K + 1)), sharex=True)
        axes_list = axes

    axes_list[0].plot(train_dates, train_values, 'b-', linewidth=1.5, alpha=0.7)
    axes_list[0].set_ylabel('foF2 (MHz)')
    axes_list[0].set_title('Исходный сигнал foF2')
    axes_list[0].grid(True, alpha=0.3)

    for k in range(K):
        axes_list[k + 1].plot(train_dates, components[:, k], color=colors[k], linewidth=1.5)
        axes_list[k + 1].set_ylabel(f'{component_names[k]} (MHz)')
        period = freq_to_period_days(omega[k], points_per_day)
        if period < 10:
            period_str = f"{period * 24:.1f} ч"
        elif period < 365:
            period_str = f"{period:.1f} дн"
        else:
            period_str = f"{period / 365:.1f} лет"
        axes_list[k + 1].set_title(f'{component_names[k]} - T~{period_str}')
        axes_list[k + 1].grid(True, alpha=0.3)

    axes_list[-1].set_xlabel('Дата')

    for ax in axes_list:
        if forecast_type == "yearly":
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            if len(train_dates) > 1000:
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
            else:
                ax.xaxis.set_major_locator(mdates.YearLocator(1))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            if len(train_dates) > 500:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            else:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'vmd_components_physical.png', dpi=150)
    plt.show()

    # Прогнозирование компонент
    print("\nПрогнозирование компонент...")

    forecast_horizon = len(forecast_dates)
    component_predictions = np.zeros((forecast_horizon, K))

    # Берем последние 3 дня данных перед прогнозом для отображения
    n_last_days = 3
    n_last_points = n_last_days * points_per_day
    last_train_dates = train_dates[-n_last_points:] if len(train_dates) > n_last_points else train_dates
    last_train_components = components[-n_last_points:, :] if len(components) > n_last_points else components

    for k in range(K):
        component = components[:, k]
        comp_std = np.std(component)
        comp_mean = np.mean(component)

        print(f"\nКомпонента {k + 1} ({component_names[k]}): "
              f"амплитуда={comp_std:.4f}, среднее={comp_mean:.4f}")

        component_centered = component - comp_mean
        n = len(component_centered)

        # Определяем период компоненты
        period_days = freq_to_period_days(omega[k], points_per_day)

        if period_days > 500:  # 11-летний цикл - продолжаем цикл без затухания
            # Берем последние 2 периода для определения фазы и амплитуды
            period_points = int(period_days * points_per_day)
            if period_points > n:
                period_points = n // 2

            # Берем последний полный период
            last_cycle = component_centered[-period_points:]

            # Просто повторяем последний период (циклическое продолжение)
            n_repeats = forecast_horizon // period_points + 2
            centered_forecast = np.tile(last_cycle, n_repeats)[:forecast_horizon]

            print(f"    Период: {period_points} точек ({period_days:.0f} дней)")
            print(f"    Циклическое продолжение последнего периода (без затухания)")

        elif period_days > 200:  # Годовая
            # Окно в 2 года
            window = min(int(2 * period_days * points_per_day), n // 2)
            if window < 100:
                window = 100

            last_pattern = component_centered[-window:]

            # Поиск похожих паттернов
            similarities = []
            positions = []
            step = max(1, window // 30)

            for i in range(0, n - window - forecast_horizon, step):
                pattern = component_centered[i:i + window]
                if len(pattern) == window:
                    corr = np.corrcoef(last_pattern, pattern)[0, 1]
                    if not np.isnan(corr) and corr > 0.65:
                        similarities.append(corr)
                        positions.append(i)

            print(f"    Окно: {window} точек, найдено {len(similarities)} паттернов")

            if len(similarities) >= 2:
                n_best = min(3, len(similarities))
                best_idx = np.argsort(similarities)[-n_best:]

                forecasts = []
                weights = []

                for idx in best_idx:
                    pos = positions[idx]
                    seg = component_centered[pos + window:pos + window + forecast_horizon]
                    if len(seg) == forecast_horizon:
                        forecasts.append(seg)
                        weights.append(similarities[idx])

                if len(forecasts) >= 1:
                    weights = np.array(weights) / np.sum(weights)
                    centered_forecast = np.sum([f * w for f, w in zip(forecasts, weights)], axis=0)
                    print(f"    Использовано {len(forecasts)} паттернов")
                else:
                    # Периодическое продолжение
                    n_repeats = forecast_horizon // window + 2
                    centered_forecast = np.tile(last_pattern, n_repeats)[:forecast_horizon]
                    print(f"    Использовано периодическое продолжение")
            else:
                n_repeats = forecast_horizon // window + 2
                centered_forecast = np.tile(last_pattern, n_repeats)[:forecast_horizon]
                print(f"    Паттернов мало, периодическое продолжение")

        else:  # Сезонная и суточная
            if period_days > 30:  # Сезонная
                window = min(int(period_days * points_per_day), n // 3)
            else:  # Суточная
                window = min(int(7 * points_per_day), n // 4)

            if window < 50:
                window = 50

            last_pattern = component_centered[-window:]

            # Поиск похожих паттернов
            similarities = []
            positions = []
            step = max(1, window // 20)

            for i in range(0, n - window - forecast_horizon, step):
                pattern = component_centered[i:i + window]
                if len(pattern) == window:
                    corr = np.corrcoef(last_pattern, pattern)[0, 1]
                    if not np.isnan(corr) and corr > 0.7:
                        similarities.append(corr)
                        positions.append(i)

            print(f"    Окно: {window} точек, найдено {len(similarities)} паттернов")

            if len(similarities) >= 2:
                n_best = min(5, len(similarities))
                best_idx = np.argsort(similarities)[-n_best:]

                forecasts = []
                weights = []

                for idx in best_idx:
                    pos = positions[idx]
                    seg = component_centered[pos + window:pos + window + forecast_horizon]
                    if len(seg) == forecast_horizon:
                        forecasts.append(seg)
                        weights.append(similarities[idx])

                if len(forecasts) >= 1:
                    weights = np.array(weights) / np.sum(weights)
                    centered_forecast = np.sum([f * w for f, w in zip(forecasts, weights)], axis=0)
                    print(f"    Использовано {len(forecasts)} паттернов")
                else:
                    n_repeats = forecast_horizon // window + 2
                    centered_forecast = np.tile(last_pattern, n_repeats)[:forecast_horizon]
                    print(f"    Использовано периодическое продолжение")
            else:
                n_repeats = forecast_horizon // window + 2
                centered_forecast = np.tile(last_pattern, n_repeats)[:forecast_horizon]
                print(f"    Паттернов мало, периодическое продолжение")

        # Плавная сшивка - короткая, чтобы не искажать цикл
        last_value = component_centered[-1]
        pred_start = centered_forecast[0] if len(centered_forecast) > 0 else 0

        blend_len = min(2, forecast_horizon // 30)  # Всего 2 точки сшивки
        for i in range(blend_len):
            alpha = i / blend_len
            centered_forecast[i] = last_value * (1 - alpha) + centered_forecast[i] * alpha

        component_predictions[:, k] = centered_forecast + comp_mean

        print(f"    Последнее значение данных: {last_value + comp_mean:.4f} MHz")
        print(f"    Первое значение прогноза: {component_predictions[0, k]:.4f} MHz")

    # Суммирование компонент
    vmd_forecast = np.sum(component_predictions, axis=1)

    # Привязка к последним данным - минимальная коррекция
    last_10_mean = np.mean(train_values[-min(10, len(train_values)):])
    forecast_first_10_mean = np.mean(vmd_forecast[:min(10, forecast_horizon)])

    print(f"\nПривязка прогноза к последним данным:")
    print(f"  Среднее последних 10 точек данных: {last_10_mean:.4f} MHz")
    print(f"  Среднее первых 10 точек прогноза: {forecast_first_10_mean:.4f} MHz")

    # Минимальная коррекция для плавного перехода
    shift = last_10_mean - forecast_first_10_mean
    shift = np.clip(shift, -0.15, 0.15)  # Ограничиваем коррекцию ±0.15
    vmd_forecast = vmd_forecast + shift
    print(f"  Коррекция сдвига: {shift:.4f} MHz")

    # Ограничиваем значения
    min_val = max(0, np.min(train_values) - 0.2)
    max_val = np.max(train_values) + 0.2
    final_forecast = np.clip(vmd_forecast, min_val, max_val)

    # Визуализация прогнозов компонент с последними 3 днями данных
    if K == 1:
        fig3, axes3 = plt.subplots(1, 1, figsize=(14, 4))
        axes3 = [axes3]
    else:
        fig3, axes3 = plt.subplots(K, 1, figsize=(14, 3 * K), sharex=True)

    for k in range(K):
        # Отображаем последние 3 дня компоненты (обучающие данные)
        axes3[k].plot(last_train_dates, last_train_components[:, k],
                      color='blue', linewidth=1.5, alpha=0.7, label='Данные (последние 3 дня)')
        # Отображаем прогноз компоненты
        axes3[k].plot(forecast_dates, component_predictions[:, k], color=colors[k],
                      linewidth=1.5, label='Прогноз')
        axes3[k].set_ylabel(f'{component_names[k]} (MHz)')
        period = freq_to_period_days(omega[k], points_per_day)
        if period < 10:
            period_str = f"{period * 24:.1f} ч"
        elif period < 365:
            period_str = f"{period:.1f} дн"
        else:
            period_str = f"{period / 365:.1f} лет"
        axes3[k].set_title(f'{component_names[k]} - T~{period_str}')
        axes3[k].grid(True, alpha=0.3)
        axes3[k].legend(loc='upper left')
        axes3[k].axhline(y=0, color='k', linestyle='-', linewidth=0.3, alpha=0.5)

    axes3[-1].set_xlabel('Дата')

    for ax in axes3:
        if forecast_type == "yearly":
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            if len(forecast_dates) > 100:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(forecast_dates) // 10)))
            else:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(forecast_dates) // 5)))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'vmd_component_forecasts_physical.png', dpi=150)
    plt.show()

    # Сравнение с реальными данными
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ С РЕАЛЬНЫМИ ДАННЫМИ")
    print("=" * 80)

    if len(test_values) > 0:
        forecast_times = np.array([d.timestamp() for d in forecast_dates])
        test_times = np.array([d.timestamp() for d in test_dates])

        mask = (test_times >= forecast_times[0]) & (test_times <= forecast_times[-1])
        filtered_test_times = test_times[mask]
        filtered_test_values = test_values[mask]
        filtered_test_dates = [d for d, m in zip(test_dates, mask) if m]

        if len(filtered_test_values) > 0:
            forecast_interp = np.interp(filtered_test_times, forecast_times, final_forecast)

            errors = filtered_test_values - forecast_interp
            abs_errors = np.abs(errors)

            mae = np.mean(abs_errors)
            rmse = np.sqrt(np.mean(errors ** 2))
            rel_errors = np.abs(errors / (filtered_test_values + 1e-10)) * 100
            mape = np.mean(rel_errors)

            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((filtered_test_values - np.mean(filtered_test_values)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            print(f"\nСтатистика сравнения:")
            print(f"  MAE: {mae:.3f} MHz")
            print(f"  RMSE: {rmse:.3f} MHz")
            print(f"  MAPE: {mape:.1f}%")
            print(f"  R²: {r2:.3f}")
        else:
            mae = rmse = mape = r2 = None
            filtered_test_dates = []
            filtered_test_values = []
            forecast_interp = []
    else:
        mae = rmse = mape = r2 = None
        filtered_test_dates = []
        filtered_test_values = []
        forecast_interp = []

    # Финальная визуализация с последними 3 днями данных
    print("\nВизуализация прогноза...")

    fig_final, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Отображаем последние 3 дня исходного сигнала (обучающие данные)
    last_train_values = train_values[-n_last_points:] if len(train_values) > n_last_points else train_values
    last_train_dates_full = train_dates[-n_last_points:] if len(train_dates) > n_last_points else train_dates

    ax1.plot(last_train_dates_full, last_train_values, 'b-', linewidth=1.5, alpha=0.7,
             label='Данные (последние 3 дня)')
    ax1.plot(forecast_dates, final_forecast, 'r-', linewidth=2, label='Прогноз VMD', alpha=0.8)

    if len(filtered_test_values) > 0:
        ax1.plot(filtered_test_dates, filtered_test_values, 'g-', linewidth=1.5,
                 label='Реальные данные', alpha=0.7)
        ax1.plot(filtered_test_dates, filtered_test_values, 'g.', markersize=4, alpha=0.7)

    ax1.axvline(x=forecast_dates[0], color='magenta', linestyle='--', linewidth=1.5, alpha=0.7,
                label='Начало прогноза')
    ax1.set_ylabel('foF2 (MHz)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    if len(filtered_test_values) > 0:
        ax2.plot(filtered_test_dates, errors, 'b-', linewidth=1, alpha=0.7)
        ax2.plot(filtered_test_dates, errors, 'b.', markersize=3, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.axhline(y=np.std(errors), color='r', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.axhline(y=-np.std(errors), color='r', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.set_ylabel('Ошибка (MHz)')
        ax2.set_xlabel('Дата')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Нет данных для сравнения', ha='center', va='center')
        ax2.set_xlabel('Дата')

    for ax in [ax1, ax2]:
        if test_start == test_end:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            date_range = max(forecast_dates) - min(forecast_dates)
            if date_range.days > 30:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            elif date_range.days > 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(f'vmd_forecast_comparison_physical.png', dpi=150)
    plt.show()

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

    return final_forecast, forecast_dates, metrics

def main():
    print(f"\nАнализ данных станции {STATION_NAME}")
    print("=" * 60)

    try:
        reader = BinaryDataReader()
    except FileNotFoundError as e:
        print(f"\nОШИБКА: {e}")
        print("\nСначала запустите create_binary_cache.py для создания кэша!")
        return

    available_years = reader.get_available_years()
    if not available_years:
        print("Нет данных в кэше!")
        return

    info = reader.get_info()
    print(f"\nИнформация о кэше:")
    print(f"  Всего точек: {info['total_points']}")
    print(f"  Доступные годы: {info['years']}")
    print(f"  Диапазон дат: {info['date_range']['start']} - {info['date_range']['end']}")
    print(f"  foF2: {info['value_stats']['min']:.2f}-{info['value_stats']['max']:.2f} MHz")

    selected_years = get_years_input(available_years)
    if not selected_years:
        return

    filter_params = get_filters_input()

    filter_info = ""
    if 'start_day' in filter_params or 'end_day' in filter_params:
        start = filter_params.get('start_day', 1)
        end = filter_params.get('end_day', 365)
        filter_info += f"Дни: {start}-{end} "
    if 'start_time' in filter_params or 'end_time' in filter_params:
        start = filter_params.get('start_time', 0)
        end = filter_params.get('end_time', 24)
        filter_info += f"Время: {start}-{end}ч"

    print(f"\nЗагрузка данных из кэша...")
    load_start = time.time()

    all_dates, all_values = reader.load_data_with_filters(
        selected_years=selected_years,
        filter_params=filter_params
    )

    if not all_dates:
        print("Нет данных для анализа")
        return

    load_time = time.time() - load_start
    print(f"Данные загружены за {load_time:.3f} сек")

    # plot_data(all_dates, all_values, selected_years, filter_info)

    predict = input("\nВыполнить прогнозирование? (y/n): ").lower() == 'y'

    if predict:
        print("\n" + "=" * 50)
        print("ВВОД ПЕРИОДОВ ДЛЯ ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ")
        print("=" * 50)

        first_year = selected_years[0]
        last_year = selected_years[-1]
        pred_last_year = selected_years[-2] if len(selected_years) > 1 else first_year

        print("\nПериод обучения:")
        train_year_start_input = input(f"  Начальный год (Enter={first_year}): ").strip()
        train_year_start = int(train_year_start_input) if train_year_start_input else first_year

        train_year_end_input = input(f"  Конечный год (Enter={pred_last_year}): ").strip()
        train_year_end = int(train_year_end_input) if train_year_end_input else pred_last_year

        train_start_input = input("  Начальный день (1-365, Enter=1): ").strip()
        train_start = int(train_start_input) if train_start_input else 1

        train_end_input = input("  Конечный день (1-365, Enter=365): ").strip()
        train_end = int(train_end_input) if train_end_input else 365

        print("\nПериод тестирования (прогноза):")
        test_year_start_input = input(f"  Начальный год (Enter={last_year}): ").strip()
        test_year_start = int(test_year_start_input) if test_year_start_input else last_year

        test_year_end_input = input(f"  Конечный год (Enter={last_year}): ").strip()
        test_year_end = int(test_year_end_input) if test_year_end_input else last_year

        test_start_input = input("  Начальный день (1-365, Enter=1): ").strip()
        test_start = int(test_start_input) if test_start_input else 1

        test_end_input = input("  Конечный день (1-365, Enter=365): ").strip()
        test_end = int(test_end_input) if test_end_input else 365

        train_info = (train_year_start, train_year_end, train_start, train_end)
        test_info = (test_year_start, test_year_end, test_start, test_end)

        print(f"\nИтоговые периоды:")
        print(f"  Обучение: {train_year_start}-{train_year_end} (дни {train_start}-{train_end})")
        print(f"  Тест: {test_year_start}-{test_year_end} (дни {test_start}-{test_end})")

        predictions, forecast_dates, metrics = predict_next_vmd(
            all_dates, all_values, train_info, test_info
        )

        if predictions is not None:
            print(f"\nПрогноз успешно выполнен!")

    print("\nАнализ завершен!")


if __name__ == "__main__":
    mp.freeze_support()
    main()