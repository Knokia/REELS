import json
import os

from ..config import ACCOUNTS_PATH, CLIENT_SECRETS_PATH, TOKEN_PATH, YOUTUBE_SCOPES


def _read_client_secrets() -> dict:
    if not os.path.exists(CLIENT_SECRETS_PATH):
        return {}
    try:
        with open(CLIENT_SECRETS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("installed") or data.get("web") or {}
    except Exception:
        return {}


def needs_setup() -> bool:
    """True if client_secrets.json is missing a usable client_id/client_secret."""
    section = _read_client_secrets()
    return not (section.get("client_id") and section.get("client_secret"))


def save_client_secrets(client_id: str, client_secret: str) -> None:
    with open(CLIENT_SECRETS_PATH, "w", encoding="utf-8") as f:
        json.dump({"installed": {
            "client_id":     client_id,
            "client_secret": client_secret,
            "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
            "token_uri":     "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }}, f)


# ── Хранилище нескольких аккаунтов (каналов) ────────────────

def _credentials_from_dict(d: dict):
    from google.oauth2.credentials import Credentials
    return Credentials(
        token=d["token"], refresh_token=d.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=d.get("client_id"), client_secret=d.get("client_secret"),
        scopes=YOUTUBE_SCOPES,
    )


def _load_accounts_raw() -> dict:
    if not os.path.exists(ACCOUNTS_PATH):
        return {}
    try:
        with open(ACCOUNTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_accounts_raw(accounts: dict) -> None:
    with open(ACCOUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(accounts, f, ensure_ascii=False)


def _migrate_legacy_token() -> None:
    """Старый формат (один yt_oauth_creds.json) -> первая запись в yt_accounts.json."""
    if os.path.exists(ACCOUNTS_PATH) or not os.path.exists(TOKEN_PATH):
        return
    try:
        with open(TOKEN_PATH, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return  # битый файл — восстанавливать нечего

    channel_id, title = "account_1", "Канал 1"
    try:
        creds = _credentials_from_dict(d)
        from . import upload as _upload  # локальный импорт — иначе циклическая зависимость
        info = _upload.get_channel_info(creds)
        channel_id = info.get("id") or channel_id
        title = info.get("title", title)
    except Exception:
        pass  # сеть/API недоступны — мигрируем токен всё равно, с именем по умолчанию

    _save_accounts_raw({channel_id: {**d, "title": title}})


def list_accounts() -> list:
    """Возвращает [{'channel_id','title','credentials'}, ...] для всех сохранённых аккаунтов."""
    _migrate_legacy_token()
    accounts = _load_accounts_raw()
    result = []
    for channel_id, data in accounts.items():
        try:
            creds = _credentials_from_dict(data)
        except Exception:
            continue
        result.append({
            "channel_id": channel_id,
            "title": data.get("title", channel_id),
            "credentials": creds,
        })
    return result


def add_account(client_id: str = None, client_secret: str = None, port: int = 9099) -> dict:
    """Запускает OAuth-флоу для ещё одного Google-аккаунта, определяет его канал
    через API и сохраняет как отдельную запись (существующие аккаунты не трогает).
    Возвращает {'channel_id', 'title', 'credentials'}."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if client_id and client_secret:
        save_client_secrets(client_id, client_secret)
    if needs_setup():
        raise RuntimeError("client_secrets.json не настроен (нет client_id/client_secret)")

    creds = (InstalledAppFlow
             .from_client_secrets_file(CLIENT_SECRETS_PATH, scopes=YOUTUBE_SCOPES)
             .run_local_server(port=port, open_browser=True))

    from . import upload as _upload
    info = _upload.get_channel_info(creds)
    accounts = _load_accounts_raw()
    channel_id = info.get("id") or f"account_{len(accounts) + 1}"
    title = info.get("title", channel_id)

    accounts[channel_id] = {
        "title": title,
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "client_id": getattr(creds, "client_id", ""),
        "client_secret": getattr(creds, "client_secret", ""),
    }
    _save_accounts_raw(accounts)
    return {"channel_id": channel_id, "title": title, "credentials": creds}


def remove_account(channel_id: str) -> None:
    accounts = _load_accounts_raw()
    if accounts.pop(channel_id, None) is not None:
        _save_accounts_raw(accounts)


def _update_stored_token(credentials) -> None:
    """После обновления просроченного токена — записываем свежий token обратно
    в запись того же аккаунта (ищем по refresh_token, он стабилен для аккаунта)."""
    accounts = _load_accounts_raw()
    for data in accounts.values():
        if data.get("refresh_token") == credentials.refresh_token:
            data["token"] = credentials.token
            _save_accounts_raw(accounts)
            return


def ensure_fresh(credentials):
    """Refreshes credentials in-place if expired, and persists the new token."""
    from google.auth.transport.requests import Request
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        _update_stored_token(credentials)
    return credentials
