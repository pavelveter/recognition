#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

CONFIG_FILE="config.ini"
ENTRY="recognition.py"

log() { printf '%(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

cd "$(dirname "$0")" || exit 1
[[ -f "$CONFIG_FILE" ]] || die "Файл $CONFIG_FILE не найден рядом со скриптом."
[[ -f "$ENTRY" ]]       || die "Не найден $ENTRY рядом со скриптом."

# --- uv ---
if ! command -v uv >/dev/null 2>&1; then
  log "Устанавливаю uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  hash -r
  command -v uv >/dev/null 2>&1 || die "uv установлен, но не в PATH. Добавь ~/.local/bin в PATH."
else
  log "uv уже установлен."
fi

# --- config.ini -> пути/токен ---
mapfile -t CFG < <(python3 - <<'PY'
import configparser, os
cfg = configparser.ConfigParser()
cfg.read('config.ini')
print(cfg.get('paths', 'selfies_default', fallback='images/@selfies'))
print(cfg.get('paths', 'cache',           fallback='numpy_cache'))
print(cfg.get('paths', 'all_photos',      fallback='images/@all_photos'))
print(os.getenv('YADISK_TOKEN', cfg.get('cloud','token', fallback='')))
PY
)
SELFIES_DIR_RAW="${CFG[0]}"
CACHE_DIR_RAW="${CFG[1]}"
ALL_PHOTOS_DIR_RAW="${CFG[2]}"
YATOKEN="${CFG[3]}"

normalize_py() {
  python3 - "$1" <<'PY'
import os, sys
s = sys.argv[1].strip()
if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
    s = s[1:-1]
s = s.replace(r"\ ", " ")
s = os.path.expanduser(s)
print(os.path.normpath(s))
PY
}
SELFIES_DIR="$(normalize_py "$SELFIES_DIR_RAW")"
CACHE_DIR="$(normalize_py "$CACHE_DIR_RAW")"
ALL_PHOTOS_DIR="$(normalize_py "$ALL_PHOTOS_DIR_RAW")"

# --- helpers ---
pin_python() {  # install + pin .python-version
  local ver="$1"
  log "Pin Python -> ${ver}"
  uv python install "${ver}"
  uv python pin "${ver}"
}

try_sync() {   # синхронизируем зависимости для pinned Python; НЕ валим скрипт
  set +e
  uv sync
  local rc=$?
  set -e
  return $rc
}

patch_requires_python_to_312() {
  [[ -f "pyproject.toml" ]] || return 1
  python3 - "$PWD/pyproject.toml" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text(encoding='utf-8')
s2 = re.sub(r'(requires-python\s*=\s*")[^"]*(")', r'\1>=3.12\2', s)
if s2 != s:
    p.write_text(s2, encoding='utf-8')
    print("patched")
PY
}

# --- политика версий ---
# 1) ВСЕГДА пробуем 3.13 (или FORCE_PY, если задан)
REQ_PY="${FORCE_PY:-3.13}"
pin_python "${REQ_PY}"

# 2) первая попытка sync на 3.13 (или FORCE_PY)
if try_sync; then
  log "uv sync прошёл на Python ${REQ_PY}."
else
  log "uv sync на ${REQ_PY} провалился. Пробую fallback → 3.12"
  # патчим pyproject.toml и сносим uv.lock (чтобы не держал старые требования)
  if [[ "$(patch_requires_python_to_312 || true)" == "patched" ]]; then
    log "pyproject.toml: requires-python → >=3.12"
  fi
  [[ -f uv.lock ]] && { rm -f uv.lock; log "Снес uv.lock для пересборки."; }

  # pin 3.12 и повторный sync
  REQ_PY="3.12"
  pin_python "${REQ_PY}"
  if try_sync; then
    log "uv sync прошёл на Python ${REQ_PY} (fallback 3.12)."
  else
    die "uv sync окончательно провалился. Проверь системные зависимости (cmake/boost/openblas/libjpeg) или смотри логи выше."
  fi
fi

# --- каталоги и чистка кеша ---
mkdir -p "$CACHE_DIR" "$SELFIES_DIR"
if [[ -d "$CACHE_DIR" ]]; then
  log "Чищу старые файлы в $CACHE_DIR (старше 30 дней)…"
  DELETED_COUNT=$(find "$CACHE_DIR" -type f -mtime +30 -print -delete | wc -l | tr -d ' ')
  log "Удалено: ${DELETED_COUNT:-0}"
fi

# --- откуда брать селфи ---
RUN_FLAGS=()
if [[ -n "${YATOKEN:-}" ]]; then
  if [[ -z "$(find "$SELFIES_DIR" -maxdepth 1 -type f -name '_SELFIE*.jpg' 2>/dev/null | head -n1)" ]]; then
    log "Селфи локально не найдены — возьмём из Я.Диска (--from-cloud)."
    RUN_FLAGS+=(--from-cloud)
  else
    log "Нашлись локальные селфи в $SELFIES_DIR — работаем локально."
    RUN_FLAGS+=(--selfies "$SELFIES_DIR")
  fi
else
  log "Токен Я.Диска отсутствует — работаем локально."
  RUN_FLAGS+=(--selfies "$SELFIES_DIR")
fi
[[ -d "$ALL_PHOTOS_DIR" ]] && RUN_FLAGS+=(--all-photos "$ALL_PHOTOS_DIR")

# --- запуск ---
log "Запускаю: uv run $ENTRY ${RUN_FLAGS[*]}"
echo
uv run "$ENTRY" "${RUN_FLAGS[@]}"

echo
read -r -n1 -p "Готово. Нажми любую клавишу…"
echo -e "\n\nCopyright (C) 2025 Pavel Borisov (github.com/pavelveter), licensed under GPLv3\n"
