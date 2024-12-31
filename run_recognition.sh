#!/bin/bash

# Указываем путь к config.ini
config_file="config.ini"

# 1. Установить текущую рабочую директорию в директорию запуска скрипта
cd "$(dirname "$0")" || exit 1

# Проверяем существование файла config.ini
if [[ ! -f "$config_file" ]]; then
    echo "Файл $config_file не найден. Пожалуйста, создай его."
    read -n1 -r -p "Нажми любую клавишу для выхода..."
    echo
    exit 1
fi

# 2. Установить uv, если он не найден
if ! command -v uv &>/dev/null; then
    echo "Устанавливаю uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Не удалось установить uv!"; exit 1; }
else
    echo "uv уже установлен."
fi

# 3. Проверить, активировано ли виртуальное окружение
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Виртуальное окружение не активировано."
    if [[ -d ".venv" ]]; then
        echo "Активирую существующее виртуальное окружение..."
        source .venv/bin/activate
    else
        echo "Создаю новое виртуальное окружение через uv..."
        uv venv || { echo "Не удалось создать виртуальное окружение!"; exit 1; }
        source .venv/bin/activate
    fi
else
    echo "Виртуальное окружение уже активно: $VIRTUAL_ENV"
fi

# 4. Проверить и установить нужную версию Python из .python-version
if [[ -f ".python-version" ]]; then
    required_python=$(<.python-version)
    echo "Требуемая версия Python: $required_python"
    uv python install $required_python || { echo "Не удалось установить Python $required_python!"; exit 1; }
else
    echo "Файл .python-version не найден. Пропускаю проверку версии Python."
fi

# 5. Установить необходимые зависимости
echo "Проверяю зависимости..."
uv sync || { echo "Не удалось синхронизировать зависимости!"; exit 1; }

# 6. Проверить папку numpy_cache и удалить старые файлы
cache_dir=$(grep -E '^\s*cache\s*=' "config.ini" | sed "s/^cache =//g;s/^ *//g")
if [[ -d "$cache_dir" ]]; then
    echo "Проверяю файлы в $cache_dir..."
    deleted_count=$(find "$cache_dir" -type f -mtime +30 -print0 | xargs -0 -I {} rm {} \; | wc -l)

    if [[ "$deleted_count" -gt 0 ]]; then
        echo "Удалено $deleted_count файлов старше месяца в $cache_dir."
    else
        echo "Файлов старше месяца в $cache_dir не найдено."
    fi
else
    echo "Папка $cache_dir не найдена. Да и пофиг."
fi

# 7. Запустить recognition.py через uv
echo "Запускаю recognition.py..."
echo " "
uv run recognition.py || { echo "Не удалось запустить recognition.py!"; exit 1; }

# 8. Деактивировать виртуальное окружение
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Деактивирую виртуальное окружение..."
    [[ "$VIRTUAL_ENV" != "" ]] && deactivate
fi

echo " "
read -p "Вот такие пироги..."