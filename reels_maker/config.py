import os

BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR         = os.path.join(BASE_DIR, "models")
CLIPS_DIR          = os.path.join(BASE_DIR, "clips")
WORK_DIR           = os.path.join(BASE_DIR, "temp_processing")
CACHE_DIR          = os.path.join(BASE_DIR, "cache")
TRANSCRIPT_CACHE_DIR = os.path.join(CACHE_DIR, "transcripts")
# Фоновая "нарезка" (Subway Surfers, песок и т.п.) для режима split-screen —
# пользователь сам кладёт сюда любые mp4/mov/mkv/webm файлы.
BACKGROUND_FOOTAGE_DIR = os.path.join(BASE_DIR, "background_footage")
os.makedirs(BACKGROUND_FOOTAGE_DIR, exist_ok=True)

# Логи запусков (logs/run_*.txt) — то же, что видно в окне приложения,
# но сохраняется на диск и переживает закрытие окна.
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.ini")

def _resolve_ffmpeg() -> str:
    # Бандл-версия ffmpeg.exe в корне проекта требует системных DLL (swscale и др.),
    # которых может не быть — используем самодостаточный бинарник из imageio-ffmpeg
    # (тот же, которым уже пользуется moviepy), это надёжнее.
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    local = os.path.join(BASE_DIR, "ffmpeg.exe")
    return local if os.path.exists(local) else "ffmpeg"


FFMPEG_EXE = _resolve_ffmpeg()

CLIENT_SECRETS_PATH = os.path.join(BASE_DIR, "client_secrets.json")
TOKEN_PATH           = os.path.join(BASE_DIR, "yt_oauth_creds.json")  # legacy single-account token
ACCOUNTS_PATH        = os.path.join(BASE_DIR, "yt_accounts.json")     # multi-account store
QUOTA_PATH            = os.path.join(BASE_DIR, "yt_quota.json")

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class FontSettings:
    font_family: str    = "Arial"
    font_size: int      = 70
    bold: bool          = True
    italic: bool        = False
    text_color: tuple   = (255, 255, 255)
    stroke_color: tuple = (0, 0, 0)
    stroke_width: int   = 3
    bg_enabled: bool    = True
    bg_color: tuple     = (0, 0, 0)
    bg_alpha: int       = 140
    position: str       = "75%"
    shadow_enabled: bool  = False
    shadow_color: tuple   = (0, 0, 0)
    shadow_offset: int    = 4
    words_per_phrase: int = 4
    karaoke_enabled: bool = True
    karaoke_color: tuple  = (255, 220, 50)


FONT_SETTINGS = FontSettings()
