import numpy as np
import os
import glob
import re
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import mmap
import json
import gc
from tqdm import tqdm

# Конфигурация
STATION_NAME = 'CB53N'
BASE_DIR = f"D:\\PycharmProjects\\SAO\\{STATION_NAME}\\"
CACHE_DIR = os.path.join(BASE_DIR, 'binary_cache')

# Создаём необходимые директории
os.makedirs(CACHE_DIR, exist_ok=True)

# Компилируем регулярное выражение
FILENAME_PATTERN = re.compile(STATION_NAME + r'_(\d{4})(\d{3})(\d{6})\.SAO', re.IGNORECASE)


def scan_all_files():
    print("=" * 60)
    print("ШАГ 1: СКАНИРОВАНИЕ ФАЙЛОВ (рекурсивно)")
    print("=" * 60)

    all_files = []
    total_dirs = 0
    start_time = time.time()

    for root, dirs, files in os.walk(BASE_DIR):
        total_dirs += 1
        for name in files:
            if not name.upper().endswith('.SAO'):
                continue

            match = FILENAME_PATTERN.match(name)
            if not match:
                continue

            year_str, day_str, time_str = match.groups()
            all_files.append((
                os.path.join(root, name),
                int(year_str),
                int(day_str),
                int(time_str[0:2]),
                int(time_str[2:4]),
                int(time_str[4:6]),
            ))

    print("\nСортировка файлов по времени...")
    all_files.sort(key=lambda x: (x[1], x[2], x[3], x[4], x[5]))

    elapsed = time.time() - start_time
    print(f"\nНайдено {len(all_files)} файлов в {total_dirs} директориях")
    print(f"Время сканирования: {elapsed:.1f} сек")

    return all_files


def extract_fof2_from_file(file_info):
    """
    Извлечение foF2 из одного файла с использованием mmap
    """
    file_path, year, day, hour, minute, second = file_info

    try:
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Пропускаем первые 5 строк
                pos = 0
                for _ in range(5):
                    next_pos = mm.find(b'\n', pos)
                    if next_pos == -1:
                        return None
                    pos = next_pos + 1

                # Читаем 6-ю строку
                end_pos = mm.find(b'\n', pos)
                if end_pos == -1:
                    line = mm[pos:].strip()
                else:
                    line = mm[pos:end_pos].strip()

                # Извлекаем первые 8 символов
                if len(line) >= 8:
                    try:
                        fof2_str = line[:8].decode('ascii').strip()
                        if fof2_str:
                            value = float(fof2_str)
                            if 1.0 <= value <= 20.0:  # Валидация
                                # Создаём datetime
                                base_date = datetime(year, 1, 1)
                                date = base_date + timedelta(days=day - 1)
                                dt = datetime(
                                    date.year, date.month, date.day,
                                    hour, minute, second
                                )
                                return (dt.timestamp(), value)
                    except (ValueError, UnicodeDecodeError):
                        pass
    except Exception:
        pass

    return None


def extract_batch(files_batch):
    """
    Обработка батча файлов
    """
    results = []
    for file_info in files_batch:
        result = extract_fof2_from_file(file_info)
        if result:
            results.append(result)
    return results


def parallel_extract_fof2(all_files, num_workers=None):
    """
    Параллельное извлечение foF2 из всех файлов
    """
    print("\n" + "=" * 60)
    print("ШАГ 2: ИЗВЛЕЧЕНИЕ foF2")
    print("=" * 60)

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # Разбиваем на батчи для более равномерной загрузки
    batch_size = max(1, len(all_files) // (num_workers * 10))
    batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]

    print(f"Файлов: {len(all_files)}")
    print(f"Процессов: {num_workers}")
    print(f"Батчей: {len(batches)}")
    print(f"Размер батча: {batch_size}")

    all_results = []

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Запускаем все задачи
        future_to_batch = {
            executor.submit(extract_batch, batch): i
            for i, batch in enumerate(batches)
        }

        # Собираем результаты с прогресс-баром
        with tqdm(total=len(batches), desc="Обработка файлов") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result(timeout=300)
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"\nОшибка в батче: {e}")
                finally:
                    pbar.update(1)

                    # Периодический вывод статистики
                    if pbar.n % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = len(all_results) / elapsed if elapsed > 0 else 0
                        pbar.set_postfix({
                            'найдено': len(all_results),
                            'скорость': f'{rate:.0f} точек/сек'
                        })

    # Сортируем по времени
    print("\nСортировка результатов...")
    all_results.sort(key=lambda x: x[0])

    elapsed = time.time() - start_time
    print(f"\nИзвлечено {len(all_results)} значений foF2")
    print(f"Процент успеха: {len(all_results) / len(all_files) * 100:.1f}%")
    print(f"Общее время: {elapsed:.1f} сек")
    print(f"Средняя скорость: {len(all_results) / elapsed:.0f} точек/сек")

    return all_results


def save_binary_cache(results):
    """
    Сохранение результатов в бинарный кэш
    """
    print("\n" + "=" * 60)
    print("ШАГ 3: СОХРАНЕНИЕ В БИНАРНЫЙ КЭШ")
    print("=" * 60)

    if not results:
        print("Нет данных для сохранения!")
        return

    # Разделяем на timestamps и значения
    timestamps = np.array([r[0] for r in results], dtype=np.float64)
    values = np.array([r[1] for r in results], dtype=np.float32)

    print(f"Timestamps: {len(timestamps)} точек, диапазон {timestamps[0]:.0f} - {timestamps[-1]:.0f}")
    print(f"Значения: мин={np.min(values):.2f}, макс={np.max(values):.2f}, среднее={np.mean(values):.2f}")

    # === ФОРМАТ 1: Полные данные (один файл) ===
    print("\nСохранение полных данных...")

    full_data_file = os.path.join(CACHE_DIR, 'fof2_full.bin')
    with open(full_data_file, 'wb') as f:
        # Заголовок: магическое число + версия + количество точек
        header = np.array([0x464F4632, 1, len(timestamps)], dtype=np.int32)
        header.tofile(f)

        # Данные: сначала все timestamps, потом все значения
        timestamps.tofile(f)
        values.tofile(f)

    full_size = os.path.getsize(full_data_file) / (1024 * 1024)
    print(f"  Полные данные: {full_size:.1f} MB")

    # === ФОРМАТ 2: По годам (отдельные файлы) ===
    print("\nСохранение по годам...")

    years_file = os.path.join(CACHE_DIR, 'years_index.json')
    years_index = {}

    # Группируем по годам
    current_year = None
    start_idx = 0

    for i, ts in enumerate(timestamps):
        year = datetime.fromtimestamp(ts).year
        if year != current_year:
            if current_year is not None:
                # Сохраняем предыдущий год
                year_data = np.column_stack((
                    timestamps[start_idx:i],
                    values[start_idx:i]
                ))

                year_file = os.path.join(CACHE_DIR, f'year_{current_year}.bin')
                year_data.tofile(year_file)

                years_index[str(current_year)] = {
                    'file': f'year_{current_year}.bin',
                    'start_idx': int(start_idx),
                    'end_idx': int(i - 1),
                    'count': int(i - start_idx),
                    'size': os.path.getsize(year_file)
                }

                print(f"  {current_year}: {i - start_idx:6d} точек, {os.path.getsize(year_file) / 1024:6.1f} KB")

            current_year = year
            start_idx = i

    # Последний год
    if current_year is not None:
        year_data = np.column_stack((
            timestamps[start_idx:],
            values[start_idx:]
        ))
        year_file = os.path.join(CACHE_DIR, f'year_{current_year}.bin')
        year_data.tofile(year_file)

        years_index[str(current_year)] = {
            'file': f'year_{current_year}.bin',
            'start_idx': int(start_idx),
            'end_idx': int(len(timestamps) - 1),
            'count': int(len(timestamps) - start_idx),
            'size': os.path.getsize(year_file)
        }

        print(f"  {current_year}: {len(timestamps) - start_idx:6d} точек, {os.path.getsize(year_file) / 1024:6.1f} KB")

    # Сохраняем индекс годов
    with open(years_file, 'w') as f:
        json.dump(years_index, f, indent=2)

    # === ФОРМАТ 3: Статистика ===
    print("\nСохранение статистики...")

    stats = {
        'total_points': int(len(timestamps)),
        'date_range': {
            'start': datetime.fromtimestamp(timestamps[0]).isoformat(),
            'end': datetime.fromtimestamp(timestamps[-1]).isoformat()
        },
        'years': list(years_index.keys()),
        'value_stats': {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        },
        'files': {
            'full': 'fof2_full.bin',
            'full_size_mb': full_size,
            'years_index': 'years_index.json'
        },
        'creation_time': datetime.now().isoformat()
    }

    stats_file = os.path.join(CACHE_DIR, 'cache_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nСтатистика сохранена в {stats_file}")
    print("\n" + "=" * 60)
    print("КЭШ УСПЕШНО СОЗДАН!")
    print("=" * 60)
    print(f"\nДиректория кэша: {CACHE_DIR}")
    print(f"Всего точек: {stats['total_points']}")
    print(f"Годы: {', '.join(map(str, sorted(map(int, stats['years']))))}")
    print(f"Диапазон: {stats['date_range']['start']} - {stats['date_range']['end']}")
    print(f"foF2: {stats['value_stats']['min']:.2f} - {stats['value_stats']['max']:.2f} MHz")


def main():
    """
    Основная функция препроцессора
    """
    print("\n" + "=" * 60)
    print(f"ПРЕПРОЦЕССОР ДАННЫХ foF2")
    print(f"Станция: {STATION_NAME}")
    print(f"Директория: {BASE_DIR}")
    print(f"Кэш: {CACHE_DIR}")
    print("=" * 60)

    # Проверяем существование кэша
    if os.path.exists(os.path.join(CACHE_DIR, 'cache_stats.json')):
        print("\nКэш уже существует!")
        response = input("Пересоздать кэш? (y/n): ").lower()
        if response != 'y':
            print("Выход.")
            return

    start_total = time.time()

    # ШАГ 1: Сканирование файлов
    all_files = scan_all_files()

    if not all_files:
        print("Файлы не найдены!")
        return

    # ШАГ 2: Извлечение foF2
    results = parallel_extract_fof2(all_files)

    if not results:
        print("Не удалось извлечь данные!")
        return

    # ШАГ 3: Сохранение в кэш
    save_binary_cache(results)

    total_time = time.time() - start_total
    print(f"\nОбщее время выполнения: {total_time / 60:.1f} минут")


if __name__ == "__main__":
    mp.freeze_support()
    main()