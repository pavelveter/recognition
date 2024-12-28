import os
import shutil
import face_recognition
from datetime import datetime
import logging
import numpy as np
from multiprocessing import Pool, cpu_count

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


def find_selfie_file(folder_path):
    """Находит последний файл _SELFIE* в папке."""
    selfies = [f for f in os.listdir(folder_path) if f.startswith("_SELFIE") and f.endswith(".jpg")]
    selfies.sort(key=lambda x: datetime.strptime(f"{x.split('_')[3]}_{x.split('_')[4]}", "%y%m%d_%H%M%S"))
    return selfies[-1] if selfies else None


def match_photo(photo_path, selfie_encodings, threshold):
    """Сравнивает лицо на фотографии с эмбеддингами селфи."""
    photo_encodings = get_face_encoding(photo_path)
    for photo_encoding in photo_encodings:
        matches = face_recognition.compare_faces(selfie_encodings, photo_encoding, tolerance=threshold)
        if True in matches:
            return photo_path
    return None


def find_matching_faces_parallel(selfie_encodings, all_photos_folder, threshold=0.6):
    """Находит совпадения фотографий с использованием параллельной обработки."""
    photo_paths = [os.path.join(all_photos_folder, f) for f in os.listdir(all_photos_folder) if f.endswith('.jpg')]
    logger.info(f"Всего фотографий для сравнения: {len(photo_paths)}")

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


def main():
    images_folder = 'images'
    all_photos_folder = os.path.join(images_folder, '@all_photos')
    
    for folder_name in os.listdir(images_folder):
        if folder_name == '@all_photos':
            continue
        
        folder_path = os.path.join(images_folder, folder_name)
        
        if os.path.isdir(folder_path):
            logger.info(f"Обрабатываю папку: {folder_path}")
            selfie_file = find_selfie_file(folder_path)
            if selfie_file:
                selfie_path = os.path.join(folder_path, selfie_file)
                
                # Получаем эмбеддинги лиц на _SELFIE*
                selfie_encodings = get_face_encoding(selfie_path)
                
                # Находим совпадающие фотографии в @all_photos параллельно
                matching_photos = find_matching_faces_parallel(selfie_encodings, all_photos_folder)
                
                # Копируем найденные фотографии в папку с _SELFIE*
                copy_photos_to_selfie_folder(matching_photos, folder_path)
                logger.info(f"Итог: скопировано {len(matching_photos)} фотографий в {folder_path}")


if __name__ == "__main__":
    main()
