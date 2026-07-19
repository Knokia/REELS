import datetime
import os

from .config import LOGS_DIR

KEEP_LAST_RUNS = 30


class SessionLog:
    """Дублирует лог приложения в файл logs/run_*.txt — окно с логом живёт
    только пока открыто приложение, а разбирать «почему клипы получились
    странными» обычно приходится уже после его закрытия."""

    def __init__(self):
        self._file = None

    def _ensure_open(self):
        if self._file is not None:
            return
        os.makedirs(LOGS_DIR, exist_ok=True)
        self._trim_old_runs()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"run_{stamp}.txt")
        self._file = open(path, "a", encoding="utf-8")
        self._file.write(f"=== AI Reels Maker PRO — запуск {stamp} ===\n")
        self._file.flush()

    @staticmethod
    def _trim_old_runs():
        try:
            runs = sorted(
                f for f in os.listdir(LOGS_DIR)
                if f.startswith("run_") and f.endswith(".txt")
            )
            for old in runs[:-KEEP_LAST_RUNS + 1] if len(runs) >= KEEP_LAST_RUNS else []:
                os.remove(os.path.join(LOGS_DIR, old))
        except Exception:
            pass

    def write(self, text: str):
        # flush на каждой строке — если приложение упадёт нативно (например,
        # краш в ffmpeg/llama), последние строки лога не должны пропасть.
        try:
            self._ensure_open()
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._file.write(f"[{ts}] {text}\n")
            self._file.flush()
        except Exception:
            pass

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
