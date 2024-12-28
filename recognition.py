import os
import shutil
import face_recognition
from datetime import datetime
import logging
import numpy as np

# Настроим логгирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()

# Кеширование результатов распознавания лиц
face_cache = {}

def get_face_encoding(image_path):
    """Получает эмбеддинг лиц с изображения."""
    if image_path in face_cache:
        logger.debug(f"Using cached result for {image_path}")
        return face_cache[image_path]
    
    logger.info(f"Обрабатываю изображение: {image_path}")
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    # Кешируем результат для данного изображения
    face_cache[image_path] = face_encodings
    logger.info(f"Найдено лиц: {len(face_encodings)}")
    return face_encodings

def find_selfie_file(folder_path):
    """Находит последний файл _SELFIE* в папке."""
    selfies = [f for f in os.listdir(folder_path) if f.startswith("_SELFIE") and f.endswith(".jpg")]
    
    # Проверяем и сортируем файлы по дате и времени
    selfies.sort(key=lambda x: datetime.strptime(f"{x.split('_')[3]}_{x.split('_')[4]}", "%y%m%d_%H%M%S"))
    if selfies:
        logger.debug(f"Found selfie file: {selfies[-1]}")
    else:
        logger.debug("No selfie file found.")
    return selfies[-1] if selfies else None

def find_matching_faces(selfie_encodings, all_photos_folder, threshold=0.6):
    """Находит фотографии в папке @all_photos, на которых есть лица, совпадающие с лицами на _SELFIE*."""
    matching_photos = []
    
    for photo_name in os.listdir(all_photos_folder):
        if photo_name.endswith('.jpg'):
            photo_path = os.path.join(all_photos_folder, photo_name)
            photo_encodings = get_face_encoding(photo_path)
            
            for photo_encoding in photo_encodings:
                # Сравниваем эмбеддинги лиц
                matches = face_recognition.compare_faces(selfie_encodings, photo_encoding, tolerance=threshold)
                
                if True in matches:
                    logger.info(f"Сравниваю лица с файлом: {photo_path}")
                    logger.info(f"Найдено совпадение с лицом на {photo_path}")
                    matching_photos.append(photo_path)
                    break  # Если хотя бы одно лицо совпало, то продолжаем дальше
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
    
    # Проходим по всем папкам в images, кроме @all_photos
    for folder_name in os.listdir(images_folder):
        if folder_name == '@all_photos':
            continue
        
        folder_path = os.path.join(images_folder, folder_name)
        
        if os.path.isdir(folder_path):
            logger.info(f"Обрабатываю папку: {folder_path}")
            selfie_file = find_selfie_file(folder_path)
            if selfie_file:
                selfie_path = os.path.join(folder_path, selfie_file)
                
                # Шаг 1: Получаем эмбеддинги лиц на _SELFIE*
                selfie_encodings = get_face_encoding(selfie_path)
                
                # Шаг 2: Находим совпадающие фотографии в @all_photos
                matching_photos = find_matching_faces(selfie_encodings, all_photos_folder)
                
                # Шаг 3: Копируем найденные фотографии в папку с _SELFIE*
                copy_photos_to_selfie_folder(matching_photos, folder_path)
                logger.info(f"Итог: скопировано {len(matching_photos)} фотографий в {folder_path}")

if __name__ == "__main__":
    main()
