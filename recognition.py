#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# recognition.py — утилита для матчинга селфи и фотоотчётов
#
# Copyright (C) 2025 Pavel Borisov (github.com/pavelveter)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from __future__ import annotations

# Анти-oversubscription — до импортов численных либ
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # OpenMP
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # OpenBLAS
os.environ.setdefault("MKL_NUM_THREADS", "1")       # Intel MKL
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")   # NumExpr

import argparse
import configparser
import csv
import hashlib
import json
import logging
import posixpath
import random
import shlex
import shutil
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, get_context
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageOps  # EXIF-ориентация
from skimage.feature import local_binary_pattern
from yadisk import YaDisk
from yadisk.exceptions import PathExistsError

# HEIC/HEIF (опционально). Если пакета нет — JPEG/PNG работают как раньше.
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ----------------------- Конфиг и логирование -----------------------

config = configparser.ConfigParser()
config.read("config.ini")

# settings
resize_enabled: bool = config.getboolean("settings", "resize")
w_str, h_str = (s.strip() for s in config.get("settings", "max_size").split(",", 1))
w, h = int(w_str), int(h_str)
max_size_tuple: tuple[int, int] = (w, h)
min_face_size: int = int(max(max_size_tuple) / 33)
check_q: bool = config.getboolean("settings", "check_quality")

laplacian_threshold: int = config.getint("settings", "laplacian_threshold")
gradient_threshold: int = config.getint("settings", "gradient_threshold")
high_pass_threshold: int = config.getint("settings", "high_pass_threshold")
high_freq_threshold: int = config.getint("settings", "high_freq_threshold")
lbp_threshold: int = config.getint("settings", "lbp_threshold")
threshold_val: float = config.getfloat("settings", "threshold")

# paths
images_root: str = config.get("paths", "images")
selfies_default: str = config.get("paths", "selfies_default")
all_photos_default: str = config.get("paths", "all_photos")
cache_dir: str = config.get("paths", "cache")
os.makedirs(cache_dir, exist_ok=True)

# cloud
cloud_selfies_root: str = config.get("cloud", "cloud_selfies")
yadisk_token_cfg: str = config.get("cloud", "token", fallback="")
yadisk_token: str = os.getenv("YADISK_TOKEN", yadisk_token_cfg)

# logging
logger = logging.getLogger("recognition")
logger.setLevel(logging.INFO)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(_sh)
logging.getLogger("yadisk").setLevel(logging.WARNING)

# ----------------------- Путь на Я.Диске -----------------------

def ypath_join(*parts: str) -> str:
    """POSIX join для путей Я.Диска. Пробелы сохраняем, убираем пустые и лишние слэши."""
    cleaned: tuple[str, ...] = tuple(
        p.strip("/") for p in parts if p is not None and p != ""
    )
    if not cleaned:
        return "/"
    return posixpath.join(*cleaned)

# ----------------------- Утилиты загрузки изображений -----------------------

def _load_image_rgb(path: str) -> np.ndarray:
    """Открывает изображение с учётом EXIF-ориентации. Всегда RGB ndarray."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # корректируем поворот
        im = im.convert("RGB")
        return np.array(im)

# ----------------------- Кеш эмбеддингов -----------------------

def cache_key_for(path: str) -> str:
    """abs_path|mtime_ns|size → sha1 (устойчивый ключ для инвалидации кеша)."""
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def cache_path_for(path: str) -> str:
    """Уникальный .npy путь в кеше."""
    base = os.path.basename(path)
    return os.path.join(cache_dir, f"{base}.{cache_key_for(path)}.npy")

# ----------------------- Ресайз -----------------------

def resize_image(image: np.ndarray, max_hw: tuple[int, int]) -> np.ndarray:
    """Ресайз с сохранением пропорций (bilinear)."""
    h, w = image.shape[:2]
    mh, mw = max_hw
    scale = min(mh / h, mw / w)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return image

# ----------------------- Качество лица -----------------------

def is_face_clear(image_bgr: np.ndarray, top: int, right: int, bottom: int, left: int) -> bool:
    """Оценка качества лица. Сначала дешёвые метрики, затем дорогие."""
    face = image_bgr[top:bottom, left:right]
    h, w = face.shape[:2]
    if h < min_face_size or w < min_face_size:
        return False

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
    if lap_var >= laplacian_threshold:
        return True

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mean = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))
    if grad_mean >= gradient_threshold:
        return True

    high_pass = gray - cv2.GaussianBlur(gray, (5, 5), 10)
    high_pass_var = float(np.var(high_pass))

    fft = np.fft.fft2(gray)
    fft_mag = np.abs(np.fft.fftshift(fft))
    mask95 = fft_mag > np.percentile(fft_mag, 95)
    high_freq_mean = float(np.mean(fft_mag[mask95]))

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray.astype(np.uint8), n_points, radius, method="uniform")
    lbp_mean = float(np.mean(lbp))

    if (grad_mean < gradient_threshold or
        high_pass_var < high_pass_threshold or
        high_freq_mean < high_freq_threshold or
        lbp_mean > lbp_threshold):
        logger.debug(
            f"Low quality: Lap={lap_var:.2f}, Grad={grad_mean:.2f}, "
            f"HPVar={high_pass_var:.2f}, FFT95={high_freq_mean:.2f}, LBP={lbp_mean:.2f}"
        )
        return False
    return True

# ----------------------- Энкодинг лиц (с кешом) -----------------------

def load_encodings_for_image(image_path: str, *, check_quality: bool, do_resize: bool) -> list[np.ndarray]:
    """Читает/считает эмбеддинги (128-d) для фото. Кеш в .npy (float32 матрица)."""
    cpath = cache_path_for(image_path)
    if os.path.exists(cpath):
        try:
            arr = np.load(cpath, allow_pickle=True)
            if arr.dtype == object:           # старый формат
                return arr.tolist()
            return [row for row in np.asarray(arr, dtype=np.float32)]  # новый формат
        except Exception as e:
            logger.warning(f"Cache read failed, recompute: {cpath} ({e})")

    base = os.path.basename(image_path)
    if base.startswith('.') or base.startswith('._'):  # AppleDouble/скрытые
        np.save(cpath, np.empty((0, 128), dtype=np.float32), allow_pickle=False)
        return []

    try:
        image_rgb = _load_image_rgb(image_path)  # EXIF-aware loader
    except Exception as e:
        logger.warning(f"Skip unreadable image: {image_path} ({e})")
        np.save(cpath, np.empty((0, 128), dtype=np.float32), allow_pickle=False)
        return []

    if do_resize:
        image_rgb = resize_image(image_rgb, max_size_tuple)

    # На CPU 'hog' обычно быстрее; с CUDA dlib можно сменить на 'cnn'
    face_locs = face_recognition.face_locations(image_rgb, model="hog")
    encs: list[np.ndarray] = []

    bgr_for_q = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) if check_quality else None
    for (top, right, bottom, left) in face_locs:
        if check_quality and bgr_for_q is not None:
            if not is_face_clear(bgr_for_q, top, right, bottom, left):
                continue
        face_enc = face_recognition.face_encodings(image_rgb, [(top, right, bottom, left)])
        if face_enc:
            encs.append(face_enc[0].astype(np.float32, copy=False))  # сразу float32

    enc_mat = np.asarray(encs, dtype=np.float32) if encs else np.empty((0, 128), dtype=np.float32)
    np.save(cpath, enc_mat, allow_pickle=False)
    return [row for row in enc_mat]

def find_selfie_files(folder_path: str) -> list[str]:
    """_SELFIE*.jpg, отсортированные по времени из имени; при ошибке — по mtime."""
    files = [f for f in os.listdir(folder_path)
             if f.startswith("_SELFIE") and f.lower().endswith(".jpg")]
    def key(name: str):
        try:
            parts = name.split("_")
            return datetime.strptime(f"{parts[3]}_{parts[4]}", "%y%m%d_%H%M%S")
        except Exception:
            return datetime.fromtimestamp(os.path.getmtime(os.path.join(folder_path, name)))
    return sorted(files, key=key)

# ----------------------- Яндекс.Диск: листинг/загрузка/выгрузка -----------------------

def list_yadisk_folders(disk: YaDisk, path: str) -> list[str]:
    try:
        items = [it for it in disk.listdir(path) if getattr(it, "type", "") == "dir"]
        return [it.name for it in items if it.name is not None]
    except Exception as e:
        logger.error(f"Я.Диск: не получил список папок '{path}': {e}")
        return []

# Параллельная загрузка только _SELFIE*.jpg
def collect_selfie_files(disk: YaDisk, cloud_root: str, local_root: str, *, heartbeat_sec: float = 1.5) -> list[tuple[str, str]]:
    """
    Собирает (cloud_path, local_path) для всех _SELFIE*.jpg (со структурой подпапок).
    Периодически логирует прогресс сканирования (heartbeat).
    """
    tasks: list[tuple[str, str]] = []
    os.makedirs(local_root, exist_ok=True)

    counts = {"dirs": 0, "files": 0, "selfies": 0}
    last_log = time.time()

    def maybe_log(force: bool = False) -> None:
        nonlocal last_log
        now = time.time()
        if force or (now - last_log) >= heartbeat_sec:
            logger.info(f"Сканирую облако… папок: {counts['dirs']}, файлов просмотрено: {counts['files']}, найдено селфи: {counts['selfies']}")
            last_log = now

    def walk(cloud_dir: str, rel_local: str = "") -> None:
        counts["dirs"] += 1
        try:
            items = list(disk.listdir(cloud_dir))
        except Exception as e:
            logger.warning(f"Я.Диск: не удалось прочитать {cloud_dir}: {e}")
            return

        for item in items:
            name = item.name
            if item.type == "dir":
                walk(item.path, os.path.join(rel_local, name))
            elif item.type == "file":
                counts["files"] += 1
                if name.startswith("_SELFIE") and name.lower().endswith(".jpg"):
                    local_path = os.path.join(local_root, rel_local, name)
                    tasks.append((item.path, local_path))
                    counts["selfies"] += 1
            # heartbeat
            maybe_log()

    logger.info(f"Сканирую облако: {cloud_root}")
    walk(cloud_root, "")
    maybe_log(force=True)
    return tasks


_DL_DISK = None  # глобал для воркера скачивания

def _dl_init(token: str):
    global _DL_DISK
    _DL_DISK = YaDisk(token=token)

def _dl_one(task: tuple[str, str]) -> str | None:
    """Скачивает один файл; с ретраями/backoff на 429/5xx."""
    cloud_file, local_file = task
    for attempt in range(5):
        try:
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            with open(local_file, "wb") as f:
                _DL_DISK.download(cloud_file, f)
            return local_file
        except Exception as e:
            msg = str(e)
            if attempt < 4 and any(x in msg for x in ("429", "Rate", "Too Many", "temporar", " 5")):
                time.sleep(1.0 * (2**attempt) + random.random())
                continue
            logging.getLogger("recognition").error(f"Я.Диск: ошибка скачивания {cloud_file} -> {local_file}: {e}")
            return None
    return None

def download_yadisk_selfies_parallel(
    token: str,
    cloud_root: str,
    local_root: str,
    *,
    max_procs: int | None = None
) -> int:
    """
    Параллельно скачивает все _SELFIE*.jpg из cloud_root в local_root.
    Возвращает число успешно скачанных файлов.
    Зависимости: collect_selfie_files(), _dl_init(), _dl_one(), logger, get_context, cpu_count.
    """
    # 1) Собираем задания (heartbeat логика внутри collect_selfie_files)
    disk = YaDisk(token=token)  # один клиент для обхода дерева
    tasks = collect_selfie_files(disk, cloud_root, local_root)
    total = len(tasks)
    logger.info(f"Найдено селфи к скачиванию: {total} шт. Из {cloud_root}")
    if total == 0:
        return 0

    # 2) Готовим пул процессов
    ctx = get_context("spawn")
    nproc = min(max_procs or min(8, cpu_count()), 16)
    # разумный chunksize для коротких задач
    chunks = max(1, total // max(nproc * 8, 1))

    # 3) Качаем с прогрессом и ETA
    done_ok = 0
    done_total = 0
    t0 = time.time()
    last = t0
    with ctx.Pool(processes=nproc, initializer=_dl_init, initargs=(token,)) as pool:
        for res in pool.imap_unordered(_dl_one, tasks, chunksize=chunks):
            done_total += 1
            if res:
                done_ok += 1

            now = time.time()
            if now - last >= 1.0:  # heartbeat ~раз в секунду
                elapsed = now - t0
                rate = done_total / elapsed if elapsed > 0 else 0.0   # файлов/сек
                remain = total - done_total
                eta_sec = int(remain / rate) if rate > 0 else -1
                logger.info(
                    f"Скачивание: ok {done_ok}/{total}, обработано {done_total}/{total}, "
                    f"скорость ~{rate:.1f} ф/с, ETA ~{eta_sec} с"
                )
                last = now

    logger.info(f"Скачано селфи: {done_ok}/{total} за {round(time.time() - t0)} сек.")
    return done_ok

def upload_yadisk_folder(disk: YaDisk, local_path: str, cloud_path: str) -> None:
    """Рекурсивная загрузка, пропуская _SELFIE*.jpg (их не заливаем)."""
    try:
        try:
            disk.mkdir(cloud_path)
        except PathExistsError:
            logger.info(f"Папка уже есть: {cloud_path}")
        for name in os.listdir(local_path):
            lp = os.path.join(local_path, name)
            cp = ypath_join(cloud_path, name)
            if name.startswith("_SELFIE") and name.lower().endswith(".jpg"):
                continue
            if os.path.isdir(lp):
                upload_yadisk_folder(disk, lp, cp)
            elif os.path.isfile(lp):
                try:
                    disk.upload(lp, cp, overwrite=False)
                    logger.info(f"↑ {lp} -> {cp}")
                except PathExistsError:
                    logger.info(f"Файл существует: {cp}")
    except Exception as e:
        logger.error(f"Я.Диск: ошибка загрузки {local_path}: {e}")

def process_folder_upload(token: str, selfie_root_local: str, cloud_root: str, folder_name: str) -> None:
    """Для пула: загрузка одной локальной папки на Я.Диск."""
    disk = YaDisk(token=token)
    local_folder = os.path.join(selfie_root_local, folder_name)
    cloud_path = ypath_join(cloud_root, os.path.basename(selfie_root_local), folder_name)
    upload_yadisk_folder(disk, local_folder, cloud_path)

# ----------------------- Источники (интерактив/CLI) -----------------------

def _normalize_user_path(raw: str) -> str:
    """Нормализуем пользовательский ввод пути: снимаем кавычки, \\  -> пробел, ~, //, .."""
    s = raw.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    try:
        parts = shlex.split(s)
        if len(parts) == 1:
            s = parts[0]
    except Exception:
        s = s.replace("\\ ", " ")
    s = os.path.expanduser(s)
    s = os.path.normpath(s)
    return s

def choose_all_photos_source(default_path: str) -> str | None:
    logger.info(f"Введите путь к папке отчёта (Enter — {default_path}):")
    raw = input("Путь: ")
    path = raw.strip()
    if not path:
        path = default_path
    else:
        path = _normalize_user_path(path)

    logger.info(f"Выбран путь: {path}")
    if os.path.isdir(path):
        logger.info(f"Отчёт: {path}")
        return path

    if os.path.isfile(path):  # если дали файл — возьмём родителя
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            logger.info(f"Дан файл, беру папку: {parent}")
            return parent

    logger.error("Нет такой папки.")
    return None

def choose_selfie_source(disk: YaDisk, root_cloud: str, local_default: str, max_procs: int | None = None) -> str | None:
    folders = list_yadisk_folders(disk, root_cloud)
    print("")
    logger.info("Источник селфи:")
    logger.info(f"0: локально ({local_default})")
    if folders:
        for i, name in enumerate(folders, 1):
            logger.info(f"{i}: {name}")
    choice = input("Ваш выбор (0 — локально, 1..N — Я.Диск, либо путь): ").strip()

    # свободный путь
    if choice and not choice.isdigit():
        path = _normalize_user_path(choice)
        if os.path.isdir(path):
            logger.info(f"Локально: {path}")
            return path
        logger.error("Указанный путь к селфи не существует.")
        return None

    if choice == "0" or choice == "":
        path = _normalize_user_path(local_default)
        logger.info(f"Локально: {path}")
        return path

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(folders):
            selected = folders[idx]
            local_path = os.path.join(images_root, selected)
            cloud_path = ypath_join(root_cloud, selected)
            # Параллельная загрузка _SELFIE*.jpg
            download_yadisk_selfies_parallel(yadisk_token, cloud_path, local_path, max_procs=max_procs)
            return local_path
    except ValueError:
        pass

    logger.error("Неверный выбор.")
    return None

# ----------------------- Сканирование фотоотчёта -----------------------

def scan_jpgs(path: str) -> list[str]:
    """Сканируем только изображения; фильтруем скрытые и AppleDouble."""
    exts = ('.jpg', '.jpeg', '.heic', '.heif')  # HEIC/HEIF при наличии pillow-heif
    out: list[str] = []
    with os.scandir(path) as it:
        for e in it:
            if not e.is_file():
                continue
            name = e.name
            if name.startswith('.') or name.startswith('._'):
                continue
            if name.lower().endswith(exts):
                out.append(e.path)
    return out

# ----------------------- Прогрев кеша -----------------------

def build_photo_encoding_cache(photo_paths: list[str], *, do_resize: bool, workers: int, chunks: int) -> dict[str, list[np.ndarray]]:
    """Предзагрузка кеша эмбеддингов для отчёта (параллельно)."""
    logger.info(f"Грею кеш эмбеддингов для {len(photo_paths)} фото…")
    t0 = time.time()
    ctx = get_context("spawn")
    nproc = max(1, min(workers, 16))
    chunk_size = max(1, chunks) if chunks and chunks > 0 else max(1, len(photo_paths) // (nproc * 4) or 1)

    def _enc(path: str) -> tuple[str, list[np.ndarray]]:
        return path, load_encodings_for_image(path, check_quality=check_q, do_resize=do_resize)

    results: dict[str, list[np.ndarray]] = {}
    with ctx.Pool(processes=nproc) as pool:
        for path, encs in pool.imap_unordered(_enc, photo_paths, chunksize=chunk_size):
            results[path] = encs
    logger.info(f"Кеш готов за {round(time.time() - t0)} сек.")
    return results

# ----------------------- Быстрые матричные расстояния -----------------------

def min_l2_dist_mat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Для каждой строки A считает минимум L2 до любой строки B.
    A: (M,128) float32, B: (N,128) float32 -> (M,)
    """
    if A.size == 0:
        return np.full((0,), np.inf, dtype=np.float32)
    if B.size == 0:
        return np.full((A.shape[0],), np.inf, dtype=np.float32)
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    aa = np.sum(A*A, axis=1, keepdims=True)    # (M,1)
    bb = np.sum(B*B, axis=1, keepdims=True).T  # (1,N)
    prod = A @ B.T                             # (M,N)
    d2 = aa + bb - 2.0 * prod
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(np.min(d2, axis=1))

# ----------------------- Параллельный матчинг -----------------------

_WORKER_SELFIE_ENCS: np.ndarray | None = None
_WORKER_THRESHOLD: float | None = None
_WORKER_RESIZE: bool | None = None
_WORKER_CHECKQ: bool | None = None

def _worker_init(selfie_encs: np.ndarray, threshold_v: float, resize_flag: bool, checkq_flag: bool):
    global _WORKER_SELFIE_ENCS, _WORKER_THRESHOLD, _WORKER_RESIZE, _WORKER_CHECKQ
    _WORKER_SELFIE_ENCS = selfie_encs.astype(np.float32, copy=False)
    _WORKER_THRESHOLD = float(threshold_v)
    _WORKER_RESIZE = bool(resize_flag)
    _WORKER_CHECKQ = bool(checkq_flag)

def _match_photo_worker(photo_path: str) -> str | None:
    try:
        encs = load_encodings_for_image(photo_path, check_quality=_WORKER_CHECKQ, do_resize=_WORKER_RESIZE)
    except Exception as e:
        logging.getLogger("recognition").warning(f"Worker skip {photo_path}: {e}")
        return None

    if not encs or _WORKER_SELFIE_ENCS is None or _WORKER_THRESHOLD is None:
        return None

    E = np.asarray(encs, dtype=np.float32)  # (K,128)
    dmins = min_l2_dist_mat(E, _WORKER_SELFIE_ENCS)
    if dmins.size and float(np.min(dmins)) <= float(_WORKER_THRESHOLD):
        return photo_path
    return None

def process_selfie_folder(folder_path: str,
                          all_photo_paths: list[str],
                          preload_cache: dict[str, list[np.ndarray]] | None,
                          *, threshold_v: float,
                          resize_flag: bool,
                          workers: int,
                          chunks: int,
                          dry_run: bool,
                          do_move: bool,
                          hardlink: bool) -> tuple[int, int, bool, set[str]]:
    """
    Обрабатывает одну папку селфи.
    Возврат: (кол-во _SELFIE, скопировано, найдены_лица, set путей-матчей).
    """
    selfies = find_selfie_files(folder_path)
    if not selfies:
        logger.info(f"Нет _SELFIE*.jpg в {folder_path}")
        return 0, 0, False, set()

    # Собираем все эмбеддинги селфи
    all_selfie_encs: list[np.ndarray] = []
    for sf in selfies:
        sp = os.path.join(folder_path, sf)
        encs = load_encodings_for_image(sp, check_quality=False, do_resize=resize_flag)
        all_selfie_encs.extend(encs)

    if not all_selfie_encs:
        logger.warning(f"В {folder_path} лица не найдены.")
        return len(selfies), 0, False, set()

    selfie_mat = np.asarray(all_selfie_encs, dtype=np.float32)  # (S,128)

    # Параллельный матчинг
    ctx = get_context("spawn")
    nproc = max(1, min(workers, 16))
    chunk_size = max(1, chunks) if chunks and chunks > 0 else max(1, len(all_photo_paths) // (nproc * 8) or 1)

    match_paths: List[str | None] = []
    if preload_cache is not None:
        for p in all_photo_paths:
            encs = preload_cache.get(p, [])
            if encs:
                E = np.asarray(encs, dtype=np.float32)
                dmin = float(np.min(min_l2_dist_mat(E, selfie_mat)))
                if dmin <= threshold_v:
                    match_paths.append(p)
                    continue
            match_paths.append(None)
    else:
        with ctx.Pool(processes=nproc,
                      initializer=_worker_init,
                      initargs=(selfie_mat, threshold_v, resize_flag, check_q)) as pool:
            for res in pool.imap_unordered(_match_photo_worker, all_photo_paths, chunksize=chunk_size):
                match_paths.append(res)

    # Копирование/линк/перемещение
    matches = {p for p in match_paths if p}
    copied = 0
    for mp in matches:
        dst = os.path.join(folder_path, os.path.basename(mp))
        if os.path.exists(dst):
            continue
        if dry_run:
            logger.info(f"[DRY-RUN] Скопировал бы: {mp} -> {folder_path}")
            copied += 1
            continue
        try:
            if hardlink:
                os.link(mp, dst)
            elif do_move:
                shutil.move(mp, dst)
            else:
                shutil.copy2(mp, dst)
            copied += 1
            logger.info(f"Скопировано: {mp} -> {folder_path}")
        except OSError:
            shutil.copy2(mp, dst)
            copied += 1
            logger.info(f"Скопировано (fallback): {mp} -> {folder_path}")

    return len(selfies), copied, True, matches

# ----------------------- Отчёты -----------------------

def _save_report_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _save_report_csv(path: str, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["selfie_folder", "matched_photo"])
        w.writerows(rows)

# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser(description="Fast selfie matcher (face_recognition).")
    parser.add_argument("--all-photos", type=str, default=None, help="Путь к полному фотоотчёту (папка).")
    parser.add_argument("--selfies", type=str, default=None, help="Путь к папке селфи (локально).")
    parser.add_argument("--from-cloud", action="store_true", help="Выбрать селфи из Я.Диска (интерактив).")
    parser.add_argument("--no-upload", action="store_true", help="Не загружать результат обратно на Я.Диск.")
    parser.add_argument("--preload", choices=["auto", "on", "off"], default="auto",
                        help="Режим прогрева кеша эмбеддингов: auto|on|off (по умолчанию auto).")
    parser.add_argument("--preload-cache", action="store_true",
                        help="(deprecated) Синоним --preload on.")    
    parser.add_argument("--resize", dest="resize", action="store_true", default=resize_enabled, help="Включить ресайз.")
    parser.add_argument("--dry-run", action="store_true", help="Не писать файлы, только логи.")
    parser.add_argument("--move", action="store_true", help="Перемещать вместо копирования.")
    parser.add_argument("--hardlink", action="store_true", help="Жёсткие ссылки вместо копирования (одна ФС).")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()), help="Кол-во процессов для матчинга/IO.")
    parser.add_argument("--chunks", type=int, default=0, help="Размер чанка для imap_unordered (0 = авто).")
    args = parser.parse_args()

    # back-compat с --preload-cache
    if args.preload_cache:
        args.preload = "on"

    if not yadisk_token:
        logger.warning("Токен Я.Диск пуст. Загрузка/скачивание работать не будут, если они нужны.")

    disk = YaDisk(token=yadisk_token) if yadisk_token else None

    # Источник отчёта
    all_photos_folder = args.all_photos or all_photos_default
    if not args.all_photos:
        all_photos_folder = choose_all_photos_source(all_photos_folder)
        if not all_photos_folder:
            logger.error("Не выбран источник полного отчёта.")
            return
    if not os.path.isdir(all_photos_folder):
        logger.error(f"Нет папки отчёта: {all_photos_folder}")
        return

    # Источник селфи
    if args.selfies:
        selfie_root = args.selfies
    elif args.from_cloud:
        if not disk:
            logger.error("Нет токена Я.Диск. Невозможно выбрать из облака.")
            return
        selfie_root = choose_selfie_source(disk, cloud_selfies_root, selfies_default, max_procs=args.workers)
    else:
        selfie_root = selfies_default
        logger.info(f"Источник селфи: {selfie_root}")
    if not selfie_root or not os.path.isdir(selfie_root):
        logger.error("Некорректный путь к селфи.")
        return

    # Список подпапок селфи
    selfie_folders = [f for f in os.listdir(selfie_root)
                      if os.path.isdir(os.path.join(selfie_root, f))]
    total_selfie_folders = len(selfie_folders)

    all_photo_paths = scan_jpgs(all_photos_folder)
    total_all_photos = len(all_photo_paths)

    logger.info(f"Папок с селфи: {total_selfie_folders}")
    logger.info(f"Фото в отчёте: {total_all_photos}")

    # Решение про прогрев кеша
    preload_mode = args.preload
    auto_should_preload = (total_selfie_folders > 20 and total_all_photos > 200)
    do_preload = (preload_mode == "on") or (preload_mode == "auto" and auto_should_preload)

    logger.info(f"Preload cache: {'ON' if do_preload else 'OFF'} "
                f"(mode={preload_mode}, guests={total_selfie_folders}, photos={total_all_photos})")

    preload = build_photo_encoding_cache(
        all_photo_paths, do_resize=args.resize, workers=args.workers, chunks=args.chunks
    ) if do_preload else None

    # Распознавание
    t0 = time.time()
    total_selfies = 0
    total_copied = 0
    no_faces: list[str] = []
    report_rows: list[tuple[str, str]] = []

    for fname in selfie_folders:
        fpath = os.path.join(selfie_root, fname)
        logger.info(f"== Обработка папки: {fpath}")
        scount, copied, found, matches = process_selfie_folder(
            fpath, all_photo_paths, preload,
            threshold_v=threshold_val, resize_flag=args.resize,
            workers=args.workers, chunks=args.chunks,
            dry_run=args.dry_run, do_move=args.move, hardlink=args.hardlink
        )
        total_selfies += scount
        total_copied += copied
        if not found:
            no_faces.append(fname)
        for mp in matches:
            report_rows.append((fname, mp))

    elapsed = round(time.time() - t0)

    # Итоги + отчёты
    print("")
    logger.info(f"ИТОГО: папок селфи={total_selfie_folders}, фото в отчёте={total_all_photos}")
    logger.info(f"Обработано селфи={total_selfies}")
    logger.info(f"Скопировано файлов={total_copied}")
    logger.info(f"Папки без лиц: {', '.join(no_faces) if no_faces else 'нет'}")
    logger.info(f"Время распознавания: {elapsed} сек.")
    print("")

    try:
        _save_report_json("report.json", {
            "selfie_folders": total_selfie_folders,
            "photos": total_all_photos,
            "selfies_processed": total_selfies,
            "copied": total_copied,
            "no_faces": no_faces,
            "elapsed_sec": elapsed,
            "dry_run": args.dry_run,
            "move": args.move,
            "hardlink": args.hardlink,
        })
        _save_report_csv("report.csv", report_rows)
        logger.info("Отчёты сохранены: report.json, report.csv")
    except Exception as e:
        logger.warning(f"Не удалось сохранить отчёты: {e}")

    # Загрузка на Я.Диск (по Enter)
    if args.no_upload:
        logger.info("Загрузка пропущена (--no-upload).")
        return

    if not yadisk_token:
        logger.warning("Нет токена Я.Диск. Пропускаю загрузку.")
        return

    user_input = input("Нажми Enter для загрузки на Яндекс.Диск, или введи что-то для выхода: ").strip()
    if user_input != "":
        logger.info("Выход без загрузки.")
        return

    t_up0 = time.time()
    token = yadisk_token
    process_partial = partial(process_folder_upload, token, selfie_root, cloud_selfies_root)
    nproc = min(8, cpu_count())
    with Pool(nproc) as pool:
        pool.map(process_partial, selfie_folders)
    t_up = round(time.time() - t_up0)

    print("")
    logger.info("Загрузка завершена.")
    logger.info(f"Время загрузки: {t_up} сек.")
    print("")

if __name__ == "__main__":
    main()
