import logging
import os
import shutil
from datetime import datetime
import face_recognition
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import configparser
from yadisk import YaDisk
from yadisk.exceptions import PathExistsError

# Настроим логгирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()

# Чтение конфигурации
config = configparser.ConfigParser()
config.read('config.ini')

# Извлечение значений из конфигурации

# Делать ли ресайз?
resize = config.getboolean('settings', 'resize')
# И если делать – то до какого размера
max_size = tuple(map(int, config.get('settings', 'max_size').split(',')))
# Максимальный размер детектируемого лица в пикселях
min_face_size = int(max(max_size) / 33)

# Как вести себя с размытыми изображениями
laplacian_threshold = config.getint('settings', 'laplacian_threshold')
gradient_threshold = config.getint('settings', 'gradient_threshold')
high_pass_threshold = config.getint('settings', 'high_pass_threshold')
high_freq_threshold = config.getint('settings', 'high_freq_threshold')
lbp_threshold = config.getint('settings', 'lbp_threshold')
threshold = config.getfloat('settings', 'threshold')

# Всякие пути
images_folder = config.get('paths', 'images')
all_photos_folder = config.get('paths', 'all_photos')
cache_numpy_folder = config.get('paths', 'cache')
selfies_default = config.get('paths', 'selfies_default')

# Yandex.Disk настройка
cloud_selfies = config.get('cloud', 'cloud_selfies')
yadisk_token = config.get('cloud', 'token')
disk = YaDisk(token=yadisk_token)


def list_yadisk_folders(path):
    """Возвращает список папок на Yandex.Disk."""
    try:
        items = [item for item in disk.listdir(path) if item['type'] == 'dir']
        return [item['name'] for item in items]
    except Exception as e:
        logger.error(f"Ошибка при получении списка папок на Yandex.Disk: {e}")
        return []


def download_yadisk_folder(disk, cloud_path, local_path):
    """Рекурсивно загружает папку с Yandex.Disk в локальную директорию, сохраняя только файлы, начинающиеся с '_SELFIE'."""
    try:
        os.makedirs(local_path, exist_ok=True)
        for item in disk.listdir(cloud_path):
            item_cloud_path = item.path
            item_local_path = os.path.join(local_path, item.name)
            
            if item.type == 'dir':
                # Рекурсивный вызов для подпапок
                download_yadisk_folder(disk, item_cloud_path, item_local_path)
            elif item.type == 'file':
                # Загрузка только файлов, начинающихся с '_SELFIE'
                if item.name.startswith('_SELFIE'):
                    with open(item_local_path, 'wb') as f:
                        disk.download(item_cloud_path, f)
                    logger.info(f"Загружено: {item_cloud_path} -> {item_local_path}")
                else:
                    logger.debug(f"Пропущено: {item_cloud_path} (не начинается с '_SELFIE')")
        
        logger.info(f"Папка {cloud_path} успешно обработана, загружены файлы '_SELFIE' в {local_path}")
    except Exception as e:
        logger.error(f"Ошибка при обработке папки {cloud_path}: {e}")


def upload_yadisk_folder(disk, local_path, cloud_path):
    """Рекурсивно загружает папку из локальной директории в Yandex.Disk, исключая файлы с именем _SELFIE*."""
    try:
        # Проверяем существование папки перед созданием
        try:
            disk.mkdir(cloud_path)
        except PathExistsError:
            logger.info(f"Папка {cloud_path} существует на Яндекс.Диске")
        
        # Проходим по всем файлам и папкам в локальной директории
        for item_name in os.listdir(local_path):
            item_local_path = os.path.join(local_path, item_name)
            item_cloud_path = os.path.join(cloud_path, item_name)
            
            # Пропускаем файлы, начинающиеся с _SELFIE
            if item_name.startswith('_SELFIE') and item_name.endswith('.jpg'):
                logger.info(f"Пропущен файл: {item_local_path} (формат _SELFIE)")
                continue
            
            if os.path.isdir(item_local_path):
                # Рекурсивно загружаем подпапки
                upload_yadisk_folder(disk, item_local_path, item_cloud_path)
            elif os.path.isfile(item_local_path):
                # Пытаемся загрузить файл
                try:
                    disk.upload(item_local_path, item_cloud_path, overwrite=False)
                    logger.info(f"Загружен: {item_local_path} -> {item_cloud_path}")
                except PathExistsError:
                    logger.info(f"Файл {item_cloud_path} уже существует на Яндекс.Диске")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке файла {item_local_path}: {e}")
        
        logger.info(f"Папка {local_path} успешно обработана в {cloud_path}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке папки {local_path}: {e}")


def process_folder(token, selfie_folder, cloud_selfies, folder_name):
    # Создаем новый экземпляр клиента Яндекс.Диска для каждого процесса
    disk = YaDisk(token=token)
    
    folder_path = os.path.join(selfie_folder, folder_name)
    cloud_path = f"{cloud_selfies}/{selfie_folder.split('/')[-1]}/{folder_name}"
    upload_yadisk_folder(disk, folder_path, cloud_path)


def choose_all_photos_source():
    """Выбор источника для обработки фотографий."""
    logger.info(f"Введите путь к папке (или нажмите Enter для использования {all_photos_folder}):")
    
    # Ввод пути к папке
    user_folder = input(f"Путь: ").strip()

    # Если путь не введён, используем папку по умолчанию
    if not user_folder:
        logger.info(f"Используем папку по умолчанию: {all_photos_folder}")
        return all_photos_folder

    # Убираем возможные экранированные символы
    user_folder = user_folder.replace("\\ ", " ").replace("\\\\", "\\").replace("'", "")
    
    logger.info(f"Выбран путь: {user_folder}")
    if os.path.isdir(user_folder):
        logger.info(f"Папка найдена: {user_folder}")
        return user_folder
    else:
        logger.error("Указанный путь не существует или не является папкой.")
        return None


def choose_selfie_source():
    """Выбор источника для обработки селфи."""
    # Получаем папки с Yandex.Disk
    folders = list_yadisk_folders(cloud_selfies)
    print(" ")
    logger.info("Выберите источник селфи:")
    logger.info(f"0: Использовать локальную папку ({selfies_default})")
    # Если есть папки на Yandex.Disk, выводим их для выбора
    if folders:
        for idx, folder in enumerate(folders, 1):
            logger.info(f"{idx}: {folder}")
    else:
        logger.info("Нет доступных папок на Yandex.Disk.")
    
    # Пожелание выбрать
    choice = input("Введите ваш выбор (0 для локальной папки, 1 и далее для Yandex.Disk): ").strip()

    # Проверяем ввод
    if choice == '0':
        logger.info(f"Выбрана локальная папка: {selfies_default}")
        return selfies_default

    try:
        folder_idx = int(choice) - 1  # Преобразуем введённый индекс в корректный индекс для папок на Yandex.Disk
        if 0 <= folder_idx < len(folders):
            selected_folder = folders[folder_idx]
            local_path = os.path.join(images_folder, selected_folder)
            cloud_path = f"{cloud_selfies}/{selected_folder}"
            download_yadisk_folder(disk, cloud_path, local_path)
            return local_path
        else:
            logger.error("Неверный выбор папки на Yandex.Disk.")
            return None
    except ValueError:
        logger.error("Неверный ввод.")
        return None


def resize_image(image, max_size=max_size):
    """Ресайзит изображение, сохраняя пропорции."""
    height, width = image.shape[:2]
    max_height, max_width = max_size

    # Вычисляем пропорции
    scale = min(max_height / height, max_width / width)
    if scale < 1:
        new_size = (int(width * scale), int(height * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        logger.debug(f"Ресайз изображения до {new_size}")
        return resized_image
    return image


def is_face_clear(image, top, right, bottom, left):
    """Проверяет, является ли лицо четким (без боке или размытия)."""
    face = image[top:bottom, left:right]
    height, width = face.shape[:2]
    
    # Простой критерий размера лица
    if height < min_face_size or width < min_face_size:  # Лицо слишком маленькое
        logger.debug("Лицо слишком маленькое для анализа.")
        return False

    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Резкость: Лапласиан
    laplacian_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))

    # Резкость: Градиенты
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_mean = np.mean(gradient_magnitude)

    # Резкость: Фильтр высоких частот
    high_pass = gray - cv2.GaussianBlur(gray, (5, 5), 10)
    high_pass_variance = np.var(high_pass)

    # Анализ боке: Частотный спектр
    fft = np.fft.fft2(gray)
    fft_magnitude = np.abs(np.fft.fftshift(fft))
    high_freq_mean = np.mean(fft_magnitude[fft_magnitude > np.percentile(fft_magnitude, 95)])

    # Оценка текстуры: Локальные бинарные паттерны (LBP)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray.astype(np.uint8), n_points, radius, method="uniform")
    lbp_mean = np.mean(lbp)

    # Применяем все критерии с учетом порогов
    if (laplacian_var < laplacian_threshold or 
        gradient_mean < gradient_threshold or 
        high_pass_variance < high_pass_threshold or 
        high_freq_mean < high_freq_threshold or
        lbp_mean > lbp_threshold):
        logger.debug(
            f"Недостаточная резкость: Laplacian={laplacian_var:.2f}, "
            f"Gradient={gradient_mean:.2f}, HighPass={high_pass_variance:.2f}, "
            f"FFT_HighFreq={high_freq_mean:.2f}, LBP={lbp_mean:.2f}"
        )
        return False

    return True


def get_face_encoding(image_path, check_quality=False, resize=False):
    """Получает эмбеддинг лиц с изображения, опционально фильтруя по качеству и ресайзом."""
    # Получаем имя файла (без пути) для кеша
    filename = os.path.basename(image_path)
    
    # Формируем путь для кеша с файлом .npy в директории cache_numpy_folder
    cache_path = os.path.join(cache_numpy_folder, filename + ".npy")

    # Убедимся, что директория для кеша существует
    os.makedirs(cache_numpy_folder, exist_ok=True)

    # Проверяем кеш
    if os.path.exists(cache_path):
        logger.debug(f"Загружаю кеш для {image_path}")
        return np.load(cache_path, allow_pickle=True)

    # Загружаем изображение
    logger.info(f"Обрабатываю изображение: {image_path}")
    image = face_recognition.load_image_file(image_path)

    # Применяем ресайз, если требуется
    if resize:
        image = resize_image(image)

    face_locations = face_recognition.face_locations(image)
    face_encodings = []

    for (top, right, bottom, left) in face_locations:
        if check_quality:
            if not is_face_clear(image, top, right, bottom, left):
                logger.debug(f"Игнорирую лицо на {image_path} из-за качества.")
                continue

        encoding = face_recognition.face_encodings(image, [(top, right, bottom, left)])[0]
        face_encodings.append(encoding)

    # Сохраняем в кеш
    np.save(cache_path, face_encodings)
    logger.info(f"Сохранён кеш: {cache_path} (лиц найдено: {len(face_encodings)})")
    return face_encodings


def find_selfie_files(folder_path):
    """Находит все файлы _SELFIE* в папке."""
    selfies = [f for f in os.listdir(folder_path) if f.startswith("_SELFIE") and f.endswith(".jpg")]
    selfies.sort(key=lambda x: datetime.strptime(f"{x.split('_')[3]}_{x.split('_')[4]}", "%y%m%d_%H%M%S"))
    return selfies


def match_photo(photo_path, selfie_encodings, threshold, resize=False):
    """Сравнивает лицо на фотографии с эмбеддингами селфи."""
    photo_encodings = get_face_encoding(photo_path, check_quality=True, resize=resize)  # Проверяем качество и применяем ресайз
    for photo_encoding in photo_encodings:
        matches = face_recognition.compare_faces(selfie_encodings, photo_encoding, tolerance=threshold)
        if True in matches:
            return photo_path
    return None


def find_matching_faces_parallel(selfie_encodings, all_photos_folder, threshold=threshold, resize=resize):
    """Находит совпадения фотографий с использованием параллельной обработки."""
    photo_paths = [os.path.join(all_photos_folder, f) for f in os.listdir(all_photos_folder) if f.endswith('.jpg')]

    with Pool(cpu_count()) as pool:
        results = pool.starmap(match_photo, [(path, selfie_encodings, threshold, resize) for path in photo_paths])

    matching_photos = [result for result in results if result]
    logger.info(f"Найдено совпадений: {len(matching_photos)}")
    return matching_photos


def copy_photos_to_selfie_folder(matching_photos, target_folder):
    """Копирует фотографии в папку с _SELFIE*."""
    for photo in matching_photos:
        shutil.copy(photo, target_folder)
        logger.info(f"Скопировано: {photo} -> {target_folder}")


def process_selfie_files(folder_path, all_photos_folder, threshold, resize=False):
    """Обрабатывает все _SELFIE* файлы в папке."""
    selfies = find_selfie_files(folder_path)
    selfie_count = len(selfies)

    if selfie_count == 0:
        logger.info(f"В папке {folder_path} нет файлов _SELFIE*")
        return selfie_count, 0, False
    
    total_copied = 0
    found_faces = False

    for selfie_file in selfies:
        selfie_path = os.path.join(folder_path, selfie_file)
        logger.info(f"Обрабатываю _SELFIE файл: {selfie_path}")
        
        # Получаем эмбеддинги лиц на _SELFIE*, без проверки качества
        selfie_encodings = get_face_encoding(selfie_path, check_quality=False, resize=resize)
        
        if selfie_encodings is None or len(selfie_encodings) == 0:
            logger.warning(f"Не найдено лиц на изображении: {selfie_path}")
            continue
        
        found_faces = True

        # Находим совпадающие фотографии в @all_photos параллельно
        matching_photos = find_matching_faces_parallel(selfie_encodings, all_photos_folder, threshold, resize)
        
        # Копируем найденные фотографии в папку с _SELFIE*
        if matching_photos:
            copy_photos_to_selfie_folder(matching_photos, folder_path)
            total_copied += len(matching_photos)
            logger.info(f"Для {selfie_file} найдено {len(matching_photos)} совпадений.")
        else:
            logger.info(f"Для {selfie_file} совпадений не найдено.")
    
    return selfie_count, total_copied, found_faces


def main():
    # Выбор источника всех фоток
    all_photos_folder = choose_all_photos_source()
    if not all_photos_folder:
        logger.error("Не удалось выбрать источник всего фотоотчёта.")
        return

    # Выбор источника селфи
    selfie_folder = choose_selfie_source()
    if not selfie_folder:
        logger.error("Не удалось выбрать источник селфи.")
        return

    selfie_folders = [f for f in os.listdir(selfie_folder) if os.path.isdir(os.path.join(selfie_folder, f)) and f != all_photos_folder]
    total_selfie_folders = len(selfie_folders)
    total_all_photos = len([f for f in os.listdir(all_photos_folder) if f.endswith('.jpg')])

    # Начинаем считать время на распознавание
    start_time = time.time()

    logger.info(f"Всего папок с селфи: {total_selfie_folders}")
    logger.info(f"Всего фотографий в отчёте: {total_all_photos}")

    total_selfies = 0
    total_copied_photos = 0
    no_faces_folders = []

    for folder_name in selfie_folders:
        folder_path = os.path.join(selfie_folder, folder_name)
        logger.info(f"Обрабатываю папку: {folder_path}")

        # Обрабатываем все _SELFIE* файлы в текущей папке
        selfie_count, copied_count, found_faces = process_selfie_files(folder_path, all_photos_folder, threshold, resize=resize)
        total_selfies += selfie_count
        total_copied_photos += copied_count

        if not found_faces:
            no_faces_folders.append(folder_name)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Итоговая информация
    print(" ")
    logger.info(f"Всего папок с селфи: {total_selfie_folders}, фотографий в отчёте: {total_all_photos}")
    logger.info(f"Обработано селфи: {total_selfies}")
    logger.info(f"Скопировано файлов: {total_copied_photos}")
    logger.info(f"Папки без найденных лиц: {', '.join(no_faces_folders) if no_faces_folders else 'нет'}")
    logger.info(f"Время распознавания: {elapsed_time:.2f} секунд.")
    print(" ")

    # Попросим пользователя нажать Enter для начала загрузки, либо ввести что-то для выхода
    user_input = input("Нажмите Enter для загрузки всех распознанных фото на Яндекс.Диск, или введите что-либо для выхода: ").strip()

    if user_input == "":
        # Получаем токен из объекта disk
        token = disk.token
        
        # Создаем частичную функцию с токеном вместо объекта disk
        process_folder_partial = partial(process_folder, token, selfie_folder, cloud_selfies)
        
        # Определяем количество процессов
        num_processes = min(8, cpu_count())
        
        # Создаем пул процессов и запускаем параллельную обработку
        with Pool(num_processes) as pool:
            pool.map(process_folder_partial, selfie_folders)
        
        logger.info("Загрузка найденных фотографий на Яндекс.Диск завершена.")
    else:
        logger.info("Выход из программы.")
        return


if __name__ == "__main__":
    main()
