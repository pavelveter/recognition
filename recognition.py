import os
import shutil
import face_recognition
from datetime import datetime
import logging
import numpy as np
from multiprocessing import Pool, cpu_count
import time

# Настроим логгирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()

def get_face_encoding(image_path):
    """Получает эмбеддинг лиц с изображения и сохраняет его в кеш."""
    cache_path = f"{image_path}.npy"

    # Проверяем, есть ли кеш
    if os.path.exists(cache_path):
        logger.debug(f"Загружаю кеш для {image_path}")
        return np.load(cache_path, allow_pickle=True)
    
    # Вычисляем эмбеддинги
    logger.info(f"Обрабатываю изображение: {image_path}")
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    # Сохраняем в кеш
    np.save(cache_path, face_encodings)
    logger.info(f"Сохранён кеш: {cache_path} (лиц найдено: {len(face_encodings)})")
    return face_encodings


def find_selfie_files(folder_path):
    """Находит все файлы _SELFIE* в папке."""
    selfies = [f for f in os.listdir(folder_path) if f.startswith("_SELFIE") and f.endswith(".jpg")]
    selfies.sort(key=lambda x: datetime.strptime(f"{x.split('_')[3]}_{x.split('_')[4]}", "%y%m%d_%H%M%S"))
    return selfies


def match_photo(photo_path, selfie_encodings, threshold):
    """Сравнивает лицо на фотографии с эмбеддингами селфи."""
    photo_encodings = get_face_encoding(photo_path)
    for photo_encoding in photo_encodings:
        matches = face_recognition.compare_faces(selfie_encodings, photo_encoding, tolerance=threshold)
        if True in matches:
            return photo_path
    return None


def find_matching_faces_parallel(selfie_encodings, all_photos_folder, threshold=0.52):
    """Находит совпадения фотографий с использованием параллельной обработки."""
    photo_paths = [os.path.join(all_photos_folder, f) for f in os.listdir(all_photos_folder) if f.endswith('.jpg')]

    with Pool(cpu_count()) as pool:
        results = pool.starmap(match_photo, [(path, selfie_encodings, threshold) for path in photo_paths])

    matching_photos = [result for result in results if result]
    logger.info(f"Найдено совпадений: {len(matching_photos)}")
    return matching_photos


def copy_photos_to_selfie_folder(matching_photos, target_folder):
    """Копирует фотографии в папку с _SELFIE*."""
    for photo in matching_photos:
        shutil.copy(photo, target_folder)
        logger.info(f"Скопировано: {photo} -> {target_folder}")


def process_selfie_files(folder_path, all_photos_folder, threshold):
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
        
        # Получаем эмбеддинги лиц на _SELFIE*
        selfie_encodings = get_face_encoding(selfie_path)
        
        if selfie_encodings is None or len(selfie_encodings) == 0:
            logger.warning(f"Не найдено лиц на изображении: {selfie_path}")
            continue
        
        found_faces = True

        # Находим совпадающие фотографии в @all_photos параллельно
        matching_photos = find_matching_faces_parallel(selfie_encodings, all_photos_folder, threshold)
        
        # Копируем найденные фотографии в папку с _SELFIE*
        if matching_photos:
            copy_photos_to_selfie_folder(matching_photos, folder_path)
            total_copied += len(matching_photos)
            logger.info(f"Для {selfie_file} найдено {len(matching_photos)} совпадений.")
        else:
            logger.info(f"Для {selfie_file} совпадений не найдено.")
    
    return selfie_count, total_copied, found_faces


def main():
    start_time = time.time()

    images_folder = 'images'
    all_photos_folder = os.path.join(images_folder, '@all_photos')
    threshold = 0.5  # Пороговое значение для сравнения

    # Информация в начале
    selfie_folders = [f for f in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, f)) and f != '@all_photos']
    total_selfie_folders = len(selfie_folders)
    total_all_photos = len([f for f in os.listdir(all_photos_folder) if f.endswith('.jpg')])

    logger.info(f"Всего папок с селфи: {total_selfie_folders}")
    logger.info(f"Всего фотографий в @all_photos: {total_all_photos}")

    total_selfies = 0
    total_copied_photos = 0
    no_faces_folders = []

    for folder_name in selfie_folders:
        folder_path = os.path.join(images_folder, folder_name)
        logger.info(f"Обрабатываю папку: {folder_path}")
        
        # Обрабатываем все _SELFIE* файлы в текущей папке
        selfie_count, copied_count, found_faces = process_selfie_files(folder_path, all_photos_folder, threshold)
        total_selfies += selfie_count
        total_copied_photos += copied_count

        if not found_faces:
            no_faces_folders.append(folder_name)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Итоговая информация
    logger.info(f"Всего папок с селфи: {total_selfie_folders}, фотографий в @all_photos: {total_all_photos}")
    logger.info(f"Обработано селфи: {total_selfies}")
    logger.info(f"Скопировано файлов: {total_copied_photos}")
    logger.info(f"Папки без найденных лиц: {', '.join(no_faces_folders) if no_faces_folders else 'нет'}")
    logger.info(f"Время выполнения: {elapsed_time:.2f} секунд")


if __name__ == "__main__":
    main()
