import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import json
import re
from scipy.interpolate import interp1d
from collections import Counter
import warnings
import time
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Конфигурация
STATION_NAME = 'BC840'
BASE_DIR = f"D:\\PycharmProjects\\SAO\\{STATION_NAME}\\"
CACHE_DIR = os.path.join(BASE_DIR, 'binary_cache')


class BinaryDataReader:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        self.stats = self._load_stats()
        self.years_index = self._load_years_index()

    def _load_stats(self):
        stats_file = os.path.join(self.cache_dir, 'cache_stats.json')
        if not os.path.exists(stats_file):
            raise FileNotFoundError(
                f"Кэш не найден! Сначала запустите create_binary_cache.py\n"
                f"Ожидаемый файл: {stats_file}"
            )
        with open(stats_file, 'r') as f:
            return json.load(f)

    def _load_years_index(self):
        index_file = os.path.join(self.cache_dir, 'years_index.json')
        if not os.path.exists(index_file):
            return None
        with open(index_file, 'r') as f:
            return json.load(f)

    def get_available_years(self):
        if self.years_index:
            return sorted([int(y) for y in self.years_index.keys()])
        return []

    def get_info(self):
        return {
            'total_points': self.stats['total_points'],
            'years': self.get_available_years(),
            'date_range': self.stats['date_range'],
            'value_stats': self.stats['value_stats']
        }

    def _resample_to_uniform(self, dates, values, target_step_minutes):
        timestamps = np.array([d.timestamp() for d in dates])
        start_time = timestamps[0]
        end_time = timestamps[-1]
        n_points = int((end_time - start_time) / (target_step_minutes * 60)) + 1
        uniform_timestamps = np.linspace(start_time, end_time, n_points)
        uniform_dates = [datetime.fromtimestamp(ts) for ts in uniform_timestamps]
        f = interp1d(timestamps, values, kind='linear', bounds_error=False, fill_value='extrapolate')
        uniform_values = f(uniform_timestamps)
        print(f"  Сжатие: {len(dates)} → {n_points} точек (коэф. {len(dates)/n_points:.1f})")
        return uniform_dates, uniform_values

    def load_data_with_filters(self, selected_years=None, filter_params=None):
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

        if len(dts) > 1:
            time_diffs = []
            for i in range(1, min(100, len(dts))):
                diff = (dts[i] - dts[i - 1]).total_seconds() / 60
                if diff > 0:
                    time_diffs.append(diff)
            if time_diffs:
                time_diffs = np.array(time_diffs)
                median_step = np.median(time_diffs)
                std_step = np.std(time_diffs)
                if std_step > median_step * 0.2:
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


def spectral_decompose(signal, data_time_step, dates):
    """
    Спектральное разложение с правильным разделением суточной и ВЧ.
    ВЧ начинается только с 3 циклов/день (период 8 часов).
    """
    T = len(signal)

    if T % 2 != 0:
        T = T - 1
        signal = signal[:T]
        dates = dates[:T]

    points_per_day = 24 * 60 // data_time_step
    time_span_days = (dates[-1] - dates[0]).total_seconds() / (24 * 3600)
    years_of_data = time_span_days / 365.25

    print(f"\n  === СПЕКТРАЛЬНОЕ РАЗЛОЖЕНИЕ ===")
    print(f"  Длительность: {time_span_days:.0f} дней ({years_of_data:.1f} лет)")

    f_hat = np.fft.rfft(signal)
    dt_days = 1.0 / points_per_day
    freqs = np.fft.rfftfreq(T, d=dt_days)

    # 11-летняя
    g_11yr = np.exp(-freqs**2 / (2 * (1.0/(5*365)*0.3)**2))

    # Годовая
    g_year = np.exp(-(freqs - 1.0/365)**2 / (2 * (1.0/365*0.15)**2))
    g_year = g_year * (1.0 - g_11yr)

    # Сезонная
    g_season = np.exp(-(freqs - 1.0/90)**2 / (2 * (1.0/90*0.2)**2))
    g_season = g_season * (1.0 - g_11yr) * (1.0 - g_year)

    # Суточная: широкая, от 0.2 до 3 циклов/день
    g_day = np.exp(-(freqs - 1.0)**2 / (2 * 1.0**2))  # широкая суточная
    g_day = g_day * (1.0 - g_11yr) * (1.0 - g_year) * (1.0 - g_season)

    # ВЧ: только высокие частоты (> 2 циклов/день = период < 12 часов)
    # Используем сигмоиду для резкого включения после 2 циклов/день
    cutoff = 2.0  # циклов/день
    steepness = 5.0
    g_hf = 1.0 / (1.0 + np.exp(-steepness * (freqs - cutoff)))
    g_hf = g_hf * (1.0 - g_11yr) * (1.0 - g_year) * (1.0 - g_season) * (1.0 - g_day)

    comp_11yr = np.fft.irfft(f_hat * g_11yr, n=T)
    comp_year = np.fft.irfft(f_hat * g_year, n=T)
    comp_season = np.fft.irfft(f_hat * g_season, n=T)
    comp_day = np.fft.irfft(f_hat * g_day, n=T)
    comp_hf = np.fft.irfft(f_hat * g_hf, n=T)

    components = np.column_stack([comp_11yr, comp_year, comp_season, comp_day, comp_hf])

    K = 5
    component_names = ['11-летний цикл', 'Годовой ход', 'Сезонный ход', 'Суточный ход', 'ВЧ']

    residual = signal - np.sum(components, axis=1)
    residual_energy = np.sum(residual ** 2) / (np.sum(signal ** 2) + 1e-12)

    print(f"\n  Частотные диапазоны:")
    print(f"    11-летний:  гауссиана с центром 0, σ≈{1.0/(5*365)*0.3:.6f}")
    print(f"    Годовой:    гауссиана с центром 1/365={1.0/365:.4f}, σ≈{1.0/365*0.15:.6f}")
    print(f"    Сезонный:   гауссиана с центром 1/90={1.0/90:.4f}, σ≈{1.0/90*0.2:.6f}")
    print(f"    Суточный:   гауссиана с центром 1.0, σ=1.0")
    print(f"    ВЧ:         сигмоида от {cutoff} циклов/день")

    print(f"\n  Результаты:")
    for k in range(K):
        std_val = np.std(components[:, k])
        print(f"    {component_names[k]}: σ={std_val:.3f} MHz")
    print(f"    Остаток: {residual_energy:.6f}")
    print(f"    Корреляции:")
    for i in range(K):
        for j in range(i + 1, K):
            corr = np.corrcoef(components[:, i], components[:, j])[0, 1]
            status = "✓" if abs(corr) < 0.15 else "⚠" if abs(corr) < 0.3 else "✗"
            print(f"      {component_names[i]} vs {component_names[j]}: {corr:+.3f} {status}")

    return components, T, component_names, points_per_day


def predict_components(components, component_names, points_per_day, forecast_horizon):
    """
    Прогнозирование компонент с плавной сшивкой.
    11-летняя: линейное продолжение.
    Годовая, сезонная, суточная, ВЧ: поиск наилучшего продолжения.
    """
    K = components.shape[1]
    predictions = np.zeros((forecast_horizon, K))

    for k in range(K):
        name = component_names[k]
        comp = components[:, k]
        comp_std = np.std(comp)
        last_val = comp[-1]
        last_vals = comp[-min(20, len(comp)):]
        last_slope = np.polyfit(np.arange(len(last_vals)), last_vals, 1)[0] if len(last_vals) > 1 else 0

        print(f"\n  Прогноз {name}: σ={comp_std:.3f} MHz")

        if '11-лет' in name:
            x_pred = np.arange(1, forecast_horizon + 1)
            predictions[:, k] = last_val + last_slope * x_pred
            print(f"    Линейное продолжение (наклон={last_slope:.6f}/точку)")

        elif 'Годовой' in name:
            window = int(2 * 365 * points_per_day)
            window = min(window, len(comp) // 3)
            window = max(window, int(180 * points_per_day))
            _predict_smooth_continuation(comp, window, forecast_horizon, predictions, k, points_per_day)

        elif 'Сезонный' in name:
            window = int(180 * points_per_day)
            window = min(window, len(comp) // 3)
            window = max(window, int(60 * points_per_day))
            _predict_smooth_continuation(comp, window, forecast_horizon, predictions, k, points_per_day)

        elif 'Суточный' in name:
            window = int(14 * points_per_day)
            window = min(window, len(comp) // 4)
            window = max(window, int(3 * points_per_day))
            _predict_smooth_continuation(comp, window, forecast_horizon, predictions, k, points_per_day)

        else:  # Высокочастотный
            window = int(3 * points_per_day)
            window = min(window, len(comp) // 4)
            window = max(window, points_per_day // 2)
            _predict_smooth_continuation(comp, window, forecast_horizon, predictions, k, points_per_day)

    return predictions


def _predict_smooth_continuation(comp, window, forecast_horizon, predictions, k, points_per_day):
    """Поиск наилучшего продолжения с адаптивными условиями для коротких данных"""

    # Если данных очень мало - простое повторение последнего паттерна
    if len(comp) < window + forecast_horizon:
        n_repeats = forecast_horizon // max(len(comp), 1) + 2
        predictions[:, k] = np.tile(comp, n_repeats)[:forecast_horizon]
        print(f"    Повторение (мало данных)")
        return

    last_pattern = comp[-window:]
    last_val = comp[-1]
    comp_std = np.std(comp)

    if comp_std < 1e-6:
        predictions[:, k] = last_val
        return

    # Определяем длительность данных в днях
    data_length_days = len(comp) / points_per_day
    is_short_data = data_length_days < 30  # меньше месяца

    # Адаптивные пороги для коротких данных
    if is_short_data:
        corr_threshold = 0.3  # снижаем с 0.5 до 0.3
        phase_threshold = 0.3  # снижаем с 0.5 до 0.3
        amp_min, amp_max = 0.4, 2.0  # расширяем диапазон
        min_candidates = 1  # достаточно 1 кандидата вместо 3
        print(f"    Короткие данные ({data_length_days:.0f} дн) → снижены пороги")
    else:
        corr_threshold = 0.5
        phase_threshold = 0.5
        amp_min, amp_max = 0.6, 1.4
        min_candidates = 3

    n_recent = min(14 * points_per_day, len(comp))
    recent_amplitude = (np.max(comp[-n_recent:]) - np.min(comp[-n_recent:])) / 2
    last_window_amplitude = (np.max(last_pattern) - np.min(last_pattern)) / 2

    phase_window = min(points_per_day // 4, window // 4)
    candidates = []
    step = max(1, window // 30 if is_short_data else window // 50)  # мельче шаг для коротких данных

    for i in range(0, len(comp) - window - forecast_horizon, step):
        pattern = comp[i:i + window]
        pattern_amplitude = (np.max(pattern) - np.min(pattern)) / 2

        if phase_window > 2:
            phase_corr = np.corrcoef(comp[-phase_window:], pattern[-phase_window:])[0, 1]
            if np.isnan(phase_corr):
                phase_corr = 0
        else:
            phase_corr = 1.0

        full_corr = np.corrcoef(last_pattern, pattern)[0, 1]
        if np.isnan(full_corr):
            full_corr = 0

        amp_ratio = pattern_amplitude / (last_window_amplitude + 1e-8)
        amp_ok = amp_min < amp_ratio < amp_max

        # Адаптивные условия
        if full_corr > corr_threshold and phase_corr > phase_threshold and amp_ok:
            score = 0.5 * full_corr + 0.3 * phase_corr + 0.2 * (1.0 - abs(1.0 - amp_ratio))
            jump = abs(pattern[0] - last_val)
            jump_norm = jump / (comp_std + 1e-8)
            score = score - 0.05 * jump_norm
            candidates.append((score, i))

    print(f"    Окно={window} тчк ({window / points_per_day:.0f} дн), кандидатов={len(candidates)}")

    # Адаптивное минимальное количество кандидатов
    if len(candidates) >= min_candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_n = min(5 if not is_short_data else 3, len(candidates))
        forecasts_list = []
        weights_list = []

        for score, pos in candidates[:best_n]:
            seg = comp[pos + window:pos + window + forecast_horizon]
            if len(seg) == forecast_horizon:
                forecasts_list.append(seg)
                weights_list.append(max(0.01, score))

        if forecasts_list:
            w = np.array(weights_list) / np.sum(weights_list)
            continuation = np.average(forecasts_list, weights=w, axis=0)

            # Адаптивная сшивка
            if is_short_data:
                blend_len = min(points_per_day // 8, forecast_horizon // 6)  # короче сшивка
                blend_len = max(blend_len, 2)
            else:
                if window > 30000:
                    blend_len = min(points_per_day, forecast_horizon // 2)
                elif window > 10000:
                    blend_len = min(points_per_day // 2, forecast_horizon // 3)
                else:
                    blend_len = min(points_per_day // 4, forecast_horizon // 4)
                blend_len = max(blend_len, 4)

            for i in range(blend_len):
                alpha = i / blend_len
                continuation[i] = last_val * (1 - alpha) + continuation[i] * alpha

            # Коррекция амплитуды
            pred_amp = (np.max(continuation) - np.min(continuation)) / 2
            if pred_amp > recent_amplitude * 1.2 and recent_amplitude > 0.1:
                scale = recent_amplitude / pred_amp
                continuation = (continuation - np.mean(continuation)) * scale + np.mean(continuation)

            predictions[:, k] = continuation
            print(f"    Прогноз: {len(forecasts_list)} паттернов, сшивка={blend_len} тчк")
            return

    # Если кандидатов недостаточно - улучшенное периодическое продолжение
    n_repeats = forecast_horizon // window + 2
    continuation = np.tile(last_pattern, n_repeats)[:forecast_horizon]

    # Короткая сшивка для коротких данных
    if is_short_data:
        blend_len = min(points_per_day // 8, forecast_horizon // 4)
        blend_len = max(blend_len, 2)
    else:
        blend_len = min(points_per_day // 2, forecast_horizon // 3)
        blend_len = max(blend_len, 4)

    for i in range(blend_len):
        alpha = i / blend_len
        continuation[i] = last_val * (1 - alpha) + continuation[i] * alpha

    predictions[:, k] = continuation
    print(f"    Период. продолжение, сшивка={blend_len} тчк")


def plot_all_results(train_dates, train_values, components, component_names,
                     forecast_dates, component_predictions, final_forecast,
                     test_dates, test_values, metrics, points_per_day):
    """Полная визуализация"""
    K = components.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(K, 3)))

    # График 1: Компоненты разложения
    fig1, axes1 = plt.subplots(K + 1, 1, figsize=(16, 3 * (K + 1)), sharex=True)
    axes1[0].plot(train_dates, train_values, 'b-', linewidth=0.3, alpha=0.7)
    axes1[0].set_ylabel('foF2 (MHz)')
    axes1[0].set_title('Исходный сигнал')
    axes1[0].grid(True, alpha=0.3)

    for k in range(K):
        axes1[k + 1].plot(train_dates, components[:, k], color=colors[k], linewidth=0.5)
        axes1[k + 1].set_ylabel('MHz')
        axes1[k + 1].set_title(f'{component_names[k]} (σ={np.std(components[:, k]):.2f})')
        axes1[k + 1].grid(True, alpha=0.3)

    for ax in axes1:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('spectral_components.png', dpi=150, bbox_inches='tight')
    plt.show()

    # График 2: Прогнозы компонент
    n_last = min(points_per_day * 90, len(train_dates))
    fig2, axes2 = plt.subplots(K, 1, figsize=(16, 3 * K), sharex=True)
    if K == 1:
        axes2 = [axes2]

    for k in range(K):
        axes2[k].plot(train_dates[-n_last:], components[-n_last:, k],
                      color='blue', linewidth=1, alpha=0.7, label='История')
        axes2[k].plot(forecast_dates, component_predictions[:, k],
                      color=colors[k], linewidth=1.5, label='Прогноз')
        axes2[k].axvline(x=forecast_dates[0], color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes2[k].set_ylabel(f'{component_names[k]} (MHz)')
        axes2[k].grid(True, alpha=0.3)
        axes2[k].legend(loc='upper left')

    for ax in axes2:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig('spectral_component_forecasts.png', dpi=150, bbox_inches='tight')
    plt.show()

    # График 3: Финальный прогноз
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(18, 10))

    n_last_show = min(len(train_dates), 5 * points_per_day)
    ax3a.plot(train_dates[-n_last_show:], train_values[-n_last_show:],
              'b-', linewidth=0.8, alpha=0.6, label='История (30 дн)')
    ax3a.plot(forecast_dates, final_forecast, 'r-', linewidth=2, label='Прогноз')

    if len(test_values) > 0 and metrics['mae'] is not None:
        forecast_times = np.array([d.timestamp() for d in forecast_dates])
        test_times = np.array([d.timestamp() for d in test_dates])
        mask = (test_times >= forecast_times[0]) & (test_times <= forecast_times[-1])
        if np.any(mask):
            test_dates_f = [d for d, m in zip(test_dates, mask) if m]
            test_values_f = test_values[mask]
            ax3a.plot(test_dates_f, test_values_f, 'g-', linewidth=1.5, alpha=0.7, label='Факт')

    ax3a.axvline(x=forecast_dates[0], color='magenta', linestyle='--', linewidth=2, alpha=0.7)
    ax3a.set_ylabel('foF2 (MHz)')
    ax3a.legend(loc='upper left')
    ax3a.grid(True, alpha=0.3)

    title = 'Прогноз foF2'
    if metrics['mae'] is not None:
        title += f' | MAE={metrics["mae"]:.3f}, R²={metrics["r2"]:.3f}'
    ax3a.set_title(title)

    if len(test_values) > 0 and metrics['mae'] is not None:
        forecast_times = np.array([d.timestamp() for d in forecast_dates])
        test_times = np.array([d.timestamp() for d in test_dates])
        mask = (test_times >= forecast_times[0]) & (test_times <= forecast_times[-1])
        if np.any(mask):
            test_times_f = test_times[mask]
            test_values_f = test_values[mask]
            forecast_interp = np.interp(test_times_f, forecast_times, final_forecast)
            errors = test_values_f - forecast_interp
            test_dates_f = [d for d, m in zip(test_dates, mask) if m]

            ax3b.plot(test_dates_f, errors, 'b-', linewidth=1, alpha=0.7)
            ax3b.plot(test_dates_f, errors, 'b.', markersize=4)
            ax3b.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax3b.fill_between(test_dates_f, -metrics['rmse'], metrics['rmse'],
                              alpha=0.1, color='red', label=f'±RMSE={metrics["rmse"]:.3f}')
            ax3b.set_ylabel('Ошибка (MHz)')
            ax3b.legend(loc='upper right')
            ax3b.grid(True, alpha=0.3)

    for ax in [ax3a, ax3b]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax3b.set_xlabel('Дата')
    plt.tight_layout()
    plt.savefig('spectral_forecast_final.png', dpi=150, bbox_inches='tight')
    plt.show()


def predict_next_vmd(datetimes, foF2_values, train_info, test_info):
    """Главная функция прогнозирования"""
    print("\n" + "=" * 60)
    print("ПРОГНОЗИРОВАНИЕ foF2")
    print("=" * 60)

    train_year_start, train_year_end, train_start, train_end = train_info
    test_year_start, test_year_end, test_start, test_end = test_info

    print(f"\nОбучение: {train_year_start}-{train_year_end}, дни {train_start}-{train_end}")
    print(f"Прогноз:  {test_year_start}-{test_year_end}, дни {test_start}-{test_end}")

    years = np.array([dt.year for dt in datetimes])
    day_of_year = np.array([dt.timetuple().tm_yday for dt in datetimes])

    train_dates_list, train_values_list = [], []
    test_dates_list, test_values_list = [], []

    for year in range(train_year_start, train_year_end + 1):
        mask = (years == year) & (day_of_year >= train_start) & (day_of_year <= train_end)
        train_dates_list.extend([dt for dt, m in zip(datetimes, mask) if m])
        train_values_list.extend(foF2_values[mask])
        if np.any(mask):
            print(f"  {year}: {np.sum(mask)} точек")

    for year in range(test_year_start, test_year_end + 1):
        mask = (years == year) & (day_of_year >= test_start) & (day_of_year <= test_end)
        test_dates_list.extend([dt for dt, m in zip(datetimes, mask) if m])
        test_values_list.extend(foF2_values[mask])
        if np.any(mask):
            print(f"  Тест {year}: {np.sum(mask)} точек")

    if len(train_values_list) < 30:
        print("ОШИБКА: недостаточно данных")
        return None, None, None

    train_dates = np.array(train_dates_list)
    train_values = np.array(train_values_list)
    idx = np.argsort([d.timestamp() for d in train_dates])
    train_dates = train_dates[idx]
    train_values = train_values[idx]

    if len(test_dates_list) > 0:
        test_dates = np.array(test_dates_list)
        test_values = np.array(test_values_list)
        idx = np.argsort([d.timestamp() for d in test_dates])
        test_dates = test_dates[idx]
        test_values = test_values[idx]
    else:
        test_dates = np.array([])
        test_values = np.array([])

    print(f"\nОбучающих точек: {len(train_values)}")
    print(f"Тестовых точек:   {len(test_values)}")

    if len(train_dates) > 1:
        diffs = [(train_dates[i] - train_dates[i - 1]).total_seconds() / 60
                 for i in range(1, min(100, len(train_dates)))
                 if (train_dates[i] - train_dates[i - 1]).total_seconds() > 0]
        data_time_step = int(np.median(diffs)) if diffs else 15
    else:
        data_time_step = 15
    points_per_day = 24 * 60 // data_time_step
    print(f"Шаг данных: {data_time_step} мин ({points_per_day} точек/сутки)")

    # Спектральное разложение
    components, processed_T, component_names, ppd = spectral_decompose(
        train_values, data_time_step, train_dates
    )

    train_values = train_values[:processed_T]
    train_dates = train_dates[:processed_T]
    points_per_day = ppd

    # Генерация дат прогноза
    forecast_dates = []
    for year in range(test_year_start, test_year_end + 1):
        start_date = datetime(year, 1, 1) + timedelta(days=test_start - 1)
        end_date = datetime(year, 1, 1) + timedelta(days=test_end)
        current = start_date
        while current <= end_date:
            forecast_dates.append(current)
            current += timedelta(minutes=data_time_step)

    forecast_horizon = len(forecast_dates)
    print(f"\nГоризонт прогноза: {forecast_horizon} точек")

    # Прогнозирование компонент
    print(f"\n=== ПРОГНОЗИРОВАНИЕ КОМПОНЕНТ ===")
    component_predictions = predict_components(
        components, component_names, points_per_day, forecast_horizon
    )

    # Суммирование
    forecast_raw = np.sum(component_predictions, axis=1)

    # УЛУЧШЕННАЯ коррекция сдвига
    # Используем не среднее, а последнее значение для сшивки
    last_data_val = train_values[-1]
    forecast_first_val = forecast_raw[0]

    # Сдвиг = разница между последним значением данных и первым значением прогноза
    shift = last_data_val - forecast_first_val

    # Ограничиваем сдвиг
    shift = np.clip(shift, -0.5, 0.5)
    forecast = forecast_raw + shift

    # Плавный переход от последнего значения данных
    blend_len = min(12, forecast_horizon // 4)
    for i in range(blend_len):
        alpha = i / blend_len
        forecast[i] = last_data_val * (1 - alpha) + forecast[i] * alpha

    print(
        f"\nКоррекция: сдвиг={shift:+.3f} MHz (последнее данных={last_data_val:.3f}, первое прогноза={forecast_first_val:.3f})")

    # Ограничение
    vmin = max(0.5, np.min(train_values) - 1.5)
    vmax = min(25, np.max(train_values) + 1.5)
    final_forecast = np.clip(forecast, vmin, vmax)

    # Метрики
    metrics = {'mae': None, 'rmse': None, 'mape': None, 'r2': None}

    if len(test_values) > 0:
        forecast_times = np.array([d.timestamp() for d in forecast_dates])
        test_times = np.array([d.timestamp() for d in test_dates])
        mask = (test_times >= forecast_times[0]) & (test_times <= forecast_times[-1])

        if np.any(mask):
            test_times_f = test_times[mask]
            test_values_f = test_values[mask]
            forecast_interp = np.interp(test_times_f, forecast_times, final_forecast)
            errors = test_values_f - forecast_interp

            metrics['mae'] = np.mean(np.abs(errors))
            metrics['rmse'] = np.sqrt(np.mean(errors ** 2))
            metrics['mape'] = np.mean(np.abs(errors / (np.abs(test_values_f) + 1e-10))) * 100
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((test_values_f - np.mean(test_values_f)) ** 2)
            metrics['r2'] = 1 - ss_res / (ss_tot + 1e-10)

            print(f"\n{'=' * 50}")
            print(f"МЕТРИКИ КАЧЕСТВА")
            print(f"{'=' * 50}")
            print(f"  MAE:  {metrics['mae']:.4f} MHz")
            print(f"  RMSE: {metrics['rmse']:.4f} MHz")
            print(f"  MAPE: {metrics['mape']:.1f}%")
            print(f"  R²:   {metrics['r2']:.4f}")

    # Визуализация
    plot_all_results(
        train_dates, train_values, components, component_names,
        forecast_dates, component_predictions, final_forecast,
        test_dates, test_values, metrics, points_per_day
    )

    return final_forecast, forecast_dates, metrics


def main():
    print(f"\n{'=' * 60}")
    print(f"АНАЛИЗ ДАННЫХ СТАНЦИИ {STATION_NAME}")
    print(f"{'=' * 60}")

    try:
        reader = BinaryDataReader()
    except FileNotFoundError as e:
        print(f"\nОШИБКА: {e}")
        return

    available_years = reader.get_available_years()
    if not available_years:
        print("Нет данных в кэше!")
        return

    info = reader.get_info()
    print(f"\nИнформация о кэше:")
    print(f"  Всего точек: {info['total_points']:,}")
    print(f"  Годы: {info['years']}")
    print(f"  Даты: {info['date_range']['start']} — {info['date_range']['end']}")
    print(f"  foF2: {info['value_stats']['min']:.2f}–{info['value_stats']['max']:.2f} MHz")

    selected_years = get_years_input(available_years)
    if not selected_years:
        return

    print(f"\nЗагрузка данных...")
    load_start = time.time()
    all_dates, all_values = reader.load_data_with_filters(selected_years=selected_years)

    if not all_dates:
        print("Нет данных для анализа")
        return

    print(f"Загружено за {time.time() - load_start:.3f} сек")

    predict = input("\nВыполнить прогнозирование? (y/n): ").lower() == 'y'

    if predict:
        print("\n" + "=" * 50)
        print("ВВОД ПЕРИОДОВ")
        print("=" * 50)

        first_year = selected_years[0]
        last_year = selected_years[-1]
        pred_last_year = selected_years[-2] if len(selected_years) > 1 else first_year

        print("\nПериод обучения:")
        train_year_start = int(input(f"  Начальный год (Enter={first_year}): ").strip() or first_year)
        train_year_end = int(input(f"  Конечный год (Enter={pred_last_year}): ").strip() or pred_last_year)
        train_start = int(input("  Начальный день (1-365, Enter=1): ").strip() or 1)
        train_end = int(input("  Конечный день (1-365, Enter=365): ").strip() or 365)

        print("\nПериод тестирования:")
        test_year_start = int(input(f"  Начальный год (Enter={last_year}): ").strip() or last_year)
        test_year_end = int(input(f"  Конечный год (Enter={last_year}): ").strip() or last_year)
        test_start = int(input("  Начальный день (1-365, Enter=1): ").strip() or 1)
        test_end = int(input("  Конечный день (1-365, Enter=365): ").strip() or 365)

        train_info = (train_year_start, train_year_end, train_start, train_end)
        test_info = (test_year_start, test_year_end, test_start, test_end)

        print(f"\nИтоговые периоды:")
        print(f"  Обучение: {train_year_start}–{train_year_end} (дни {train_start}–{train_end})")
        print(f"  Тест:     {test_year_start}–{test_year_end} (дни {test_start}–{test_end})")

        predictions, forecast_dates, metrics = predict_next_vmd(
            all_dates, all_values, train_info, test_info
        )

        if predictions is not None:
            print(f"\nПрогноз успешно выполнен!")

    print(f"\nАнализ завершен!")


if __name__ == "__main__":
    mp.freeze_support()
    main()