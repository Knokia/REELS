import json
import re
import urllib.request

API_URL = "https://sponsor.ajay.app/api/skipSegments"

# Категории, которые действительно являются рекламой/самопиаром.
# "intro"/"outro"/"music_offtopic"/"filler" по умолчанию не трогаем —
# они не всегда мешают вирусному моменту и легко зацепить полезный контент.
DEFAULT_CATEGORIES = ["sponsor", "selfpromo", "interaction"]


def extract_video_id(url: str) -> str:
    for pat in (
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([\w-]{11})',
    ):
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def fetch_segments(video_id: str, categories=None, timeout: float = 5.0) -> list:
    """Возвращает список (start, end) секунд рекламных сегментов. [] при любой ошибке/отсутствии данных."""
    if not video_id:
        return []
    categories = categories or DEFAULT_CATEGORIES
    url = f"{API_URL}?videoID={video_id}&categories={json.dumps(categories)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return sorted(
            (seg["segment"][0], seg["segment"][1]) for seg in data if seg.get("segment")
        )
    except Exception:
        return []
