"""Настройки выбора LLM-бэкенда (локальная модель / Claude API) и хранение
API-ключа — по тому же принципу, что yt_accounts.json для YouTube: отдельный
локальный json, не в settings.ini и не в git (см. .gitignore).

Это "заготовка" под Claude API (см. обсуждение стоимости/архитектуры) — сам
API ключ пока не проверен реальным вызовом (нет доступа с российского IP,
см. переписку), поэтому backend по умолчанию всегда "local" и ничего не
меняется в поведении приложения, пока пользователь явно не переключит тумблер
в UI и не впишет свой ключ.
"""
import json
import os

from .config import BASE_DIR

AI_CREDENTIALS_PATH = os.path.join(BASE_DIR, "ai_credentials.json")

DEFAULT_AI_SETTINGS = {
    "backend": "local",  # "local" | "claude"
    "claude_model": "claude-haiku-4-5-20251001",
    "claude_api_key": "",
}


def load_ai_settings() -> dict:
    if not os.path.exists(AI_CREDENTIALS_PATH):
        return dict(DEFAULT_AI_SETTINGS)
    try:
        with open(AI_CREDENTIALS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {**DEFAULT_AI_SETTINGS, **data}
    except Exception:
        return dict(DEFAULT_AI_SETTINGS)


def save_ai_settings(settings: dict) -> None:
    try:
        merged = {**DEFAULT_AI_SETTINGS, **settings}
        with open(AI_CREDENTIALS_PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
