import contextlib
import gc
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ctranslate2 (faster-whisper backend) dynamically links cublas64_12.dll,
# resolved via CUDA_PATH\bin rather than the normal Windows DLL search order.
# On this machine CUDA_PATH points at a CUDA 13.x toolkit (needed to build
# llama-cpp-python with CUDA), which only ships cublas64_13.dll, while
# ctranslate2 was built against CUDA 12.x — hence "cublas64_12.dll is not
# found". _CUBLAS_PKG_DIR is the nvidia-cublas-cu12 pip package's own dir
# (has a matching cublas64_12.dll); _cuda_path_for_ctranslate2() temporarily
# points CUDA_PATH there ONLY around the Whisper call. It can't be set
# globally at import time: llama_cpp reads the *real* CUDA_PATH\lib at
# import time to find its own CUDA build, and would crash if it were
# already overridden by then.
_CUBLAS_PKG_DIR = None


def _register_nvidia_dll_dirs():
    global _CUBLAS_PKG_DIR
    if os.name != "nt":
        return
    try:
        import nvidia.cublas
        import nvidia.cudnn
        import nvidia.cuda_nvrtc
    except ImportError:
        return
    for pkg in (nvidia.cublas, nvidia.cudnn, nvidia.cuda_nvrtc):
        pkg_dir = next(iter(pkg.__path__), None)
        if not pkg_dir:
            continue
        bin_dir = os.path.join(pkg_dir, "bin")
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)

    _CUBLAS_PKG_DIR = next(iter(nvidia.cublas.__path__), None)


@contextlib.contextmanager
def _cuda_path_for_ctranslate2():
    if not _CUBLAS_PKG_DIR:
        yield
        return
    old = os.environ.get("CUDA_PATH")
    os.environ["CUDA_PATH"] = _CUBLAS_PKG_DIR
    try:
        yield
    finally:
        if old is not None:
            os.environ["CUDA_PATH"] = old
        else:
            os.environ.pop("CUDA_PATH", None)


_register_nvidia_dll_dirs()

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from llama_cpp import Llama
from moviepy import VideoFileClip
from PyQt6.QtCore import QThread, pyqtSignal

from .analysis import MultimodalAnalyzer
from .config import (BACKGROUND_FOOTAGE_DIR, CLIPS_DIR, FFMPEG_EXE, FONT_SETTINGS,
                     MODELS_DIR, TRANSCRIPT_CACHE_DIR, WORK_DIR)
from .emotion import EmotionDetector
from .face_crop import SmartFaceCrop
from .hooks import HookOverlay
from .scenes import detect_scenes
from . import sponsorblock
from .subtitles import SubtitleRenderer
from .virality import ViralityScorer


class ProcessingThread(QThread):
    log      = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    # Испускается вместо finished, когда emit_jobs_only=True (сбор моментов для
    # компиляции из нескольких видео — см. compile_multi.py) — вместо готового
    # результата отдаёт сырые данные для дальнейшей склейки в MainWindow.
    jobs_ready = pyqtSignal(dict)

    # Кандидатов для анализа всегда берём больше, чем нужно финальных клипов —
    # иначе скорингу виральности физически не из чего отсеивать слабые.
    CANDIDATE_OVERHEAD = 8
    MAX_CANDIDATES = 25
    MIN_VIRALITY_SCORE = 0.25
    # Ничто раньше не проверяло длину момента — LLM иногда предлагает (или
    # scene-snap схлопывает до) момент в несколько секунд, особенно в быстро
    # смонтированных заставках/тизерах в начале выпуска. Такой огрызок проходит
    # дальше как валидный клип. Расширяем всё, что короче этого порога.
    MIN_CLIP_DURATION = 15

    def __init__(self, url, quality, language, clip_duration,
                 zoom_enabled, zoom_intensity,
                 face_crop_enabled=True, hook_enabled=True, virality_enabled=True,
                 multi_speaker_crop=False, clip_count=7, index_offset=0,
                 centered_layout_enabled=False, split_screen_enabled=False,
                 compilation_enabled=False, work_subdir=None, emit_jobs_only=False):
        super().__init__()
        self.url               = url
        # Изолированная рабочая папка на источник — нужна, когда несколько
        # ProcessingThread обрабатывают разные видео для одной мульти-компиляции
        # одновременно/последовательно: общий WORK_DIR/video.mp4 иначе перезаписывался
        # бы каждым следующим источником раньше, чем предыдущий успевал попасть в склейку.
        self.work_dir = os.path.join(WORK_DIR, work_subdir) if work_subdir else WORK_DIR
        # Только собрать (video_path, render_jobs, title) и отдать через jobs_ready,
        # не рендеря готовый файл и не удаляя рабочую папку — использует внешний
        # оркестратор (склейка нескольких источников в одну компиляцию).
        self.emit_jobs_only    = emit_jobs_only
        self.quality           = quality
        self.language           = language
        self.clip_duration     = clip_duration
        self.zoom_enabled      = zoom_enabled
        self.zoom_intensity    = zoom_intensity
        self.face_crop_enabled = face_crop_enabled
        self.hook_enabled      = hook_enabled
        self.virality_enabled  = virality_enabled
        self.multi_speaker_crop = multi_speaker_crop
        self.clip_count         = clip_count
        self.index_offset       = index_offset
        self._download_finished = False
        self._last_download_log = -5
        self.srt_path           = None
        self._llm               = None
        self.video_title        = None
        # Подпапка clips/<название видео>/ — вычисляется в начале run(), чтобы
        # клипы разных исходников не смешивались в одной общей папке.
        self.clip_subdir        = None
        self.sponsor_segments   = []
        # Альтернативный формат вывода: исходник вписывается в кадр целиком (без
        # кропа/зума/фокуса на лицах) и центрируется, с заголовком сверху — вместо
        # обычного full-bleed кропа с умным слежением за лицом.
        self.centered_layout_enabled = centered_layout_enabled
        # Split-screen: исходник в верхней половине, фоновая "нарезка"
        # (Subway Surfers/песок/т.п. из BACKGROUND_FOOTAGE_DIR) — в нижней.
        self.split_screen_enabled    = split_screen_enabled
        # Компиляция: вместо вертикальных Shorts-клипов найденные моменты
        # склеиваются в одно длинное горизонтальное 16:9 видео (см. compilation.py).
        self.compilation_enabled     = compilation_enabled
        # Заголовки, уже сгенерированные в этом прогоне — подсказываем модели
        # их не повторять, иначе все клипы одного видео скатываются в один шаблон.
        self._generated_titles  = []

    # ── Model loading ─────────────────────────────────────────

    def find_model(self):
        if not os.path.exists(MODELS_DIR):
            raise Exception("Создай папку 'models' и положи .gguf файл.")
        gguf = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith('.gguf'))
        if not gguf:
            raise Exception(f"Нет .gguf в {MODELS_DIR}")
        pat    = re.compile(r'-(\d+)-of-(\d+)\.gguf$', re.IGNORECASE)
        splits = [f for f in gguf if pat.search(f)]
        chosen = ([f for f in splits if pat.search(f).group(1) == '00001']
                  or splits or [None])[0] or gguf[0]
        self.log.emit(f"🔍 Модель: {chosen}")
        return os.path.join(MODELS_DIR, chosen)

    def load_llm(self):
        mp2   = self.find_model()
        pat   = re.compile(r'-(\d+)-of-(\d+)\.gguf$', re.IGNORECASE)
        match = pat.search(os.path.basename(mp2))
        # ВАЖНО: реальный обученный контекст этой модели (стандартный
        # Qwen2.5-7B-Instruct, не "-1M" вариант) — 32768 токенов; llama.cpp сам
        # об этом предупреждает ("n_ctx_seq > n_ctx_train -- possible training
        # context overflow"), если задать больше. Раньше здесь ошибочно стояло
        # 131072 (перепутал с расширенным вариантом модели) — на реальном
        # 165-минутном видео это привело не просто к замедлению (1.7 часа на
        # один вызов), а похоже и к деградации качества ответа (модель начала
        # механически перечислять контент с самого начала вместо осмысленного
        # поиска). Не поднимаем n_ctx выше того, на чём модель реально обучена.
        kwargs = dict(model_path=mp2, n_ctx=32768, n_threads=4, verbose=False)
        if match:
            kwargs['n_gpu_layers'] = 0

        def _try_load(kw):
            try:
                return Llama(**kw)
            except ValueError as e:
                err = str(e)
                if 'Failed to load model' in err and match:
                    base = pat.sub('.gguf', os.path.basename(kw['model_path']))
                    bp   = os.path.join(os.path.dirname(kw['model_path']), base)
                    if os.path.exists(bp):
                        kw = dict(kw, model_path=bp)
                        return Llama(**kw)
                raise

        # Сначала пробуем целиком выгрузить модель на GPU (Q6_K 7B ≈ 6.3 ГБ,
        # в 12 ГБ VRAM помещается вместе с KV-кэшем; Whisper к этому моменту
        # уже выгружен) — инференс на порядок быстрее CPU. Если сборка
        # llama-cpp-python без CUDA или VRAM не хватило — тихо откатываемся
        # на прежний CPU-режим, чтобы обработка не ломалась.
        if not match:
            try:
                llm = _try_load(dict(kwargs, n_gpu_layers=-1))
                self.log.emit("⚡ LLM на GPU")
                return llm
            except Exception as e:
                self.log.emit(f"⚠️ GPU для LLM недоступен ({e}) — работаю на CPU")
        return _try_load(kwargs)

    # ── Transcript cache ──────────────────────────────────────

    def _transcript_cache_key(self, video_path, is_local):
        if is_local:
            raw = f"local:{os.path.abspath(self.url)}:{os.path.getsize(self.url)}:{os.path.getmtime(self.url)}"
        else:
            raw = f"url:{self.url}:{self.language}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cached_transcript(self, key):
        path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{key}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if not d.get("words"):
                return None  # старый "отравленный" пустой кэш — считаем промахом
            return d["text"], d["words"]
        except Exception:
            return None

    def _save_cached_transcript(self, key, text, words):
        if not words:
            # Пустая транскрипция почти всегда значит, что распознавание не
            # сработало (а не что в видео реально нет речи) — не кэшируем,
            # иначе одна неудачная попытка молча "протухает" во все последующие
            # запуски с тем же файлом.
            return
        try:
            os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)
            path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{key}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"text": text, "words": words}, f)
        except Exception:
            pass

    # ── Main run ──────────────────────────────────────────────

    def run(self):
        try:
            os.makedirs(self.work_dir, exist_ok=True)
            is_local = os.path.isfile(self.url)

            if is_local:
                self.log.emit("=== ЛОКАЛЬНЫЙ ФАЙЛ ===")
                self.video_title = os.path.splitext(os.path.basename(self.url))[0]
                video_path = os.path.join(self.work_dir, "video.mp4")
                shutil.copy2(self.url, video_path)
                self.progress.emit(20)
            else:
                self.log.emit("=== СКАЧИВАЕМ С YOUTUBE ===")
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = os.path.join(self.work_dir, "video.mp4")
                    shutil.copy2(self.download_youtube(tmpdir), video_path)
                    if self.srt_path and os.path.exists(self.srt_path):
                        new_srt = os.path.join(self.work_dir, "subtitles.srt")
                        shutil.copy2(self.srt_path, new_srt)
                        self.srt_path = new_srt

                video_id = sponsorblock.extract_video_id(self.url)
                if video_id:
                    self.sponsor_segments = sponsorblock.fetch_segments(video_id)
                    if self.sponsor_segments:
                        self.log.emit(
                            f"🚫 SponsorBlock: {len(self.sponsor_segments)} рекламных сегментов "
                            f"будут исключены из нарезки"
                        )

            self.clip_subdir = self._make_video_folder(self.video_title)
            self.log.emit(f"📁 Клипы: clips/{self.clip_subdir}/")

            self.log.emit("\n=== ТРАНСКРИБАЦИЯ ==="); self.progress.emit(25)
            cache_key = self._transcript_cache_key(video_path, is_local)
            cached    = self._load_cached_transcript(cache_key)
            if cached:
                text, words = cached
                self.log.emit(f"✅ Транскрипт из кэша | Слов: {len(words)}")
            else:
                text, words = self.transcribe(video_path)
                self._save_cached_transcript(cache_key, text, words)

            try:
                with VideoFileClip(video_path) as _probe:
                    real_duration = _probe.duration or 0.0
            except Exception:
                real_duration = 0.0

            if not words:
                # Распознавание речи не дало текста (не только пустое видео —
                # часто это сбой ASR). Берём реальную длину видео, а не
                # фиктивные 60с — иначе поиск моментов схлопывается в первую
                # минуту вместо всего ролика. Субтитров в клипах не будет.
                duration = real_duration or 60.0
                self.log.emit(
                    f"⚠️ Транскрипция пуста — субтитров не будет, поиск моментов "
                    f"пойдёт по всей длине видео ({duration:.0f}с) без текста"
                )
            else:
                covered = words[-1]['end']
                if real_duration and covered < real_duration * 0.7:
                    # Распознано заметно меньше, чем реальная длина видео — скорее
                    # всего Whisper прервался раньше конца. Ищем моменты по всей
                    # длине видео (не только по покрытой части), но за пределами
                    # covered субтитров не будет — это лучше, чем схлопнуть всё
                    # в маленькое окно распознанного куска.
                    self.log.emit(
                        f"⚠️ Транскрипт покрывает только {covered:.0f}с из {real_duration:.0f}с "
                        f"видео — распознавание, похоже, прервалось раньше конца. Часть видео "
                        f"может остаться без субтитров."
                    )
                    duration = real_duration
                else:
                    duration = covered

            self.log.emit("\n=== СЦЕНЫ ==="); self.progress.emit(30)
            scenes = detect_scenes(video_path)
            self.log.emit(f"✅ Сцен: {len(scenes)}" if scenes else "⚠️ scenedetect недоступен")

            self.log.emit("\n=== МУЛЬТИМОДАЛЬНЫЙ АНАЛИЗ ==="); self.progress.emit(35)
            audio_energies, beats, silences, laugh_events = {}, [], [], []
            try:
                audio_y, audio_sr = MultimodalAnalyzer.load_audio(video_path)
                audio_energies = MultimodalAnalyzer.extract_audio_energy(audio_y, audio_sr)
                self.log.emit(f"🎵 Энергия: {len(audio_energies)} точек")
                beats = MultimodalAnalyzer.detect_beats(audio_y, audio_sr)
                self.log.emit(f"🥁 Битов: {len(beats)}")
                silences = MultimodalAnalyzer.detect_silence_gaps(audio_y, audio_sr)
                self.log.emit(f"🔕 Пауз: {len(silences)}")
                laugh_events = MultimodalAnalyzer.detect_laugh_applause(audio_y, audio_sr)
                self.log.emit(f"😂 События: {len(laugh_events)}")
                del audio_y
            except Exception as e:
                self.log.emit(f"⚠️ Аудио-анализ недоступен ({e}) — зум/скоринг будут работать без него")
            face_timeline  = MultimodalAnalyzer.detect_faces_timeline(video_path, 0.5)
            self.log.emit(f"👤 Кадров с лицами: {len(face_timeline)}")

            # Эмоции — только MediaPipe + OpenCV Haar (без сторонних ML)
            self.log.emit("😊 Определяю эмоции (MediaPipe Face Mesh)...")
            emotion_timeline = EmotionDetector.detect_timeline(video_path, sample_fps=0.33)
            if emotion_timeline:
                counts: dict = {}
                for e in emotion_timeline.values():
                    counts[e] = counts.get(e, 0) + 1
                top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
                self.log.emit(
                    f"   ✅ {len(emotion_timeline)} кадров | "
                    + ", ".join(f"{e}={c}" for e, c in top)
                )
            else:
                self.log.emit("   ⚠️ MediaPipe недоступен — субтитры белые")

            speech_rates    = MultimodalAnalyzer.speech_rate(words)
            sent_boundaries = MultimodalAnalyzer.find_sentence_boundaries(words)
            self.log.emit(f"📝 Границ предложений: {len(sent_boundaries)}")

            self.log.emit("\n=== ЛУЧШИЕ МОМЕНТЫ ==="); self.progress.emit(50)
            self._llm = self.load_llm()
            candidate_count = min(self.MAX_CANDIDATES, self.clip_count + self.CANDIDATE_OVERHEAD)
            highlights = self.find_highlights(text, duration, scenes, words, count=candidate_count)
            self.log.emit(f"🔎 Кандидатов: {len(highlights)} (нужно клипов: {self.clip_count})")
            for h in highlights:
                ws = h.get('start_time', 0); we = h.get('end_time', ws + 30)
                h['_text_snippet'] = " ".join(
                    w['word'] for w in words if ws <= w['start'] <= we
                )[:500]

            if self.virality_enabled:
                self.log.emit("\n=== СКОРИНГ ВИРАЛЬНОСТИ ==="); self.progress.emit(55)
                highlights = ViralityScorer.score_highlights(
                    highlights, words, audio_energies,
                    face_timeline, speech_rates, laugh_events
                )
                for i, h in enumerate(highlights):
                    vs  = h.get('virality_score', 0.0)
                    bar = '█' * int(vs * 10) + '░' * (10 - int(vs * 10))
                    self.log.emit(f"   #{i+1} [{bar}] {vs:.3f}")
                before = len(highlights)
                highlights = [h for h in highlights
                             if h.get('virality_score', 0.0) >= self.MIN_VIRALITY_SCORE][:self.clip_count]
                highlights.sort(key=lambda x: x['start_time'])
                self.log.emit(
                    f"🧹 Отсеяно {before - len(highlights)} слабых момент(ов) "
                    f"(порог {self.MIN_VIRALITY_SCORE}) → в работу: {len(highlights)}"
                )
            else:
                highlights = highlights[:self.clip_count]

            self.log.emit("\n🎯 Умная привязка границ...")
            cut_fixed = 0
            for h in highlights:
                for key in ('start_time', 'end_time'):
                    t2 = h[key]
                    t2 = MultimodalAnalyzer.snap_to_sentence_boundary(t2, sent_boundaries)
                    t2 = MultimodalAnalyzer.snap_to_silence(t2, silences)
                    if beats:
                        t2 = MultimodalAnalyzer.snap_to_beat(t2, beats)
                    h[key] = t2
                if h['end_time'] - h['start_time'] < 5:
                    h['end_time'] = h['start_time'] + self.clip_duration

                # Привязка выше — "мягкая" (снапает только если граница/пауза попала
                # в узкий допуск в пару секунд); если рядом ничего не нашлось, время
                # остаётся как есть и может упасть прямо внутрь слова. Это последний,
                # безусловный рубеж: если так и произошло — дотягиваем/откатываем
                # ровно до границы этого слова, чтобы клип не резал речь на полуслове.
                new_end = min(self._extend_past_word(h['end_time'], words), duration)
                new_start = max(self._retreat_to_word_start(h['start_time'], words), 0.0)
                if new_end != h['end_time'] or new_start != h['start_time']:
                    cut_fixed += 1
                h['end_time'], h['start_time'] = new_end, new_start
            if cut_fixed:
                self.log.emit(f"✂️ Поправил {cut_fixed} границ(ы), резавших слово пополам")

            # Снаппинг к паузам/предложениям/битам двигает start и end НЕЗАВИСИМО
            # друг от друга — итоговая длина клипа после этого может заметно
            # отличаться от запрошенной self.clip_duration (раньше единственная
            # проверка — "если короче 5с" — пропускала всё, что попадало в
            # диапазон 5-25с, даже если пользователь просил ровно 25с). Явно
            # подгоняем каждый клип под clip_duration±3с — сначала пробуем
            # растянуть/сжать симметрично вокруг центра момента, затем ещё раз
            # прогоняем защиту от разреза слова, т.к. новая граница может снова
            # упасть внутрь слова.
            duration_tolerance = 3.0
            duration_fixed = 0
            for h in highlights:
                cur = h['end_time'] - h['start_time']
                if abs(cur - self.clip_duration) > duration_tolerance:
                    center = (h['start_time'] + h['end_time']) / 2
                    st2 = max(0.0, center - self.clip_duration / 2)
                    et2 = min(duration, st2 + self.clip_duration)
                    st2 = max(0.0, et2 - self.clip_duration)
                    h['start_time'], h['end_time'] = st2, et2
                    h['end_time'] = min(self._extend_past_word(h['end_time'], words), duration)
                    h['start_time'] = max(self._retreat_to_word_start(h['start_time'], words), 0.0)
                    duration_fixed += 1
            if duration_fixed:
                self.log.emit(
                    f"⏱️ Подровнял длину {duration_fixed} клипа(ов) под запрошенные "
                    f"{self.clip_duration}±{duration_tolerance:.0f}с"
                )
                # Подгонка длины пересчитывает start/end вокруг центра момента и
                # обрезает по границам видео — у моментов, изначально близких друг
                # к другу (особенно у самого начала/конца видео, где обрезка "стягивает"
                # разные центры в одно и то же окно), итоговые диапазоны могут
                # схлопнуться в одинаковые или почти одинаковые. Ранний дедуп в
                # find_highlights() этого не видит, т.к. работает ДО подгонки —
                # прогоняем его ещё раз по уже финальным границам.
                before_redup = len(highlights)
                highlights = self._dedupe_highlights(highlights)
                if before_redup - len(highlights):
                    self.log.emit(
                        f"🧬 Убрано ещё {before_redup - len(highlights)} дублирующихся "
                        f"момент(ов) — совпали после подгонки длины"
                    )

            self.log.emit(f"\n=== ГЕНЕРАЦИЯ ЗАГОЛОВКОВ/ХУКОВ ({len(highlights)}) ===")
            render_jobs = []
            for i, h in enumerate(highlights, 1):
                self.progress.emit(55 + int(i / len(highlights) * 10))
                clip_title = self.generate_clip_title(text, h, words)
                self.log.emit(f"📛 {i}/{len(highlights)}: {clip_title}")
                # Хук и зум нужны только вертикальным Shorts-клипам —
                # в компиляции сегменты идут без них.
                hook_text = (self.generate_hook(text, h, words)
                             if self.hook_enabled and not self.compilation_enabled else None)
                zoom_plan = (self.analyze_zoom_points(text, h, words, audio_energies)
                             if self.zoom_enabled and not self.compilation_enabled else None)
                render_jobs.append({'highlight': h, 'title': clip_title,
                                     'hook_text': hook_text, 'zoom_plan': zoom_plan})

            # LLM больше не нужен — освобождаем память перед параллельным рендером
            del self._llm; self._llm = None; gc.collect()

            if self.compilation_enabled:
                if self.emit_jobs_only:
                    # Это видео — один из нескольких источников для мульти-
                    # компиляции; финальную склейку и очистку work_dir делает
                    # оркестратор (MainWindow) после сбора со всех источников.
                    self.log.emit(f"\n✅ Собрано моментов: {len(render_jobs)}")
                    self.progress.emit(100)
                    EmotionDetector.cleanup()
                    self.jobs_ready.emit({
                        'video_path': video_path,
                        'video_title': self.video_title or '',
                        'render_jobs': render_jobs,
                        'work_dir': self.work_dir,
                    })
                    return

                # Режим «Компиляция»: одно длинное 16:9 видео вместо вертикальных
                # клипов. Дальнейший Shorts-рендер полностью пропускается.
                from .compilation import build_compilation
                self.log.emit(f"\n=== КОМПИЛЯЦИЯ ({len(render_jobs)} сегментов) ===")
                output_dir = (os.path.join(CLIPS_DIR, self.clip_subdir)
                              if self.clip_subdir else CLIPS_DIR)
                os.makedirs(output_dir, exist_ok=True)
                comp_path = build_compilation(
                    video_path, render_jobs, output_dir,
                    self.video_title or f"video_{int(time.time())}",
                    ffmpeg_threads=max(1, os.cpu_count() or 4),
                    log_cb=self.log.emit,
                    progress_cb=lambda p: self.progress.emit(65 + int(p * 100)),
                )
                with VideoFileClip(comp_path) as _c:
                    comp_dur = _c.duration
                output_clips = [{
                    'path': comp_path,
                    'title': f"Компиляция — {self.video_title or ''}".strip(" —"),
                    'filename': os.path.basename(comp_path),
                    'start': 0.0, 'end': comp_dur, 'duration': comp_dur,
                    'reason': 'compilation',
                    'virality_score': max((j['highlight'].get('virality_score', 0.0)
                                           for j in render_jobs), default=0.0),
                    'hook': '',
                }]
                self.progress.emit(100)
                self.log.emit(f"\n✅ Компиляция готова: {comp_dur / 60:.1f} мин")
                EmotionDetector.cleanup()
                try:
                    gc.collect(); time.sleep(1)
                    shutil.rmtree(self.work_dir, ignore_errors=True)
                except Exception:
                    pass
                self.finished.emit(output_clips)
                return

            workers = max(1, min(3, (os.cpu_count() or 2) // 2))
            ffmpeg_threads = max(1, (os.cpu_count() or 4) // workers)
            self.log.emit(f"\n=== НАРЕЗКА {len(render_jobs)} КЛИПОВ (потоков: {workers}) ===")

            output_clips = [None] * len(render_jobs)

            def _render(idx, job):
                h = job['highlight']
                try:
                    path = self.cut_and_caption(
                        video_path, h['start_time'], h['end_time'], words,
                        job['title'], job['zoom_plan'], job['hook_text'],
                        emotion_timeline, ffmpeg_threads=ffmpeg_threads,
                        index=self.index_offset + idx,
                    )
                    return idx, {
                        'path':           path,
                        'title':          job['title'],
                        'filename':       os.path.basename(path),
                        'start':          h['start_time'],
                        'end':            h['end_time'],
                        'duration':       h['end_time'] - h['start_time'],
                        'reason':         h.get('reason', ''),
                        'virality_score': h.get('virality_score', 0.0),
                        'hook':           job['hook_text'] or '',
                    }
                except Exception as e:
                    self.log.emit(f"⚠️ Клип {idx + 1}: {e}")
                    return idx, None

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_render, i, job) for i, job in enumerate(render_jobs)]
                done = 0
                for fut in as_completed(futures):
                    idx, clip = fut.result()
                    output_clips[idx] = clip
                    done += 1
                    self.progress.emit(65 + int(done / len(render_jobs) * 30))

            output_clips = [c for c in output_clips if c]

            self.progress.emit(100)
            self.log.emit(f"\n✅ Клипов: {len(output_clips)}")
            if output_clips:
                self.log.emit("\n📊 Топ по виральности:")
                for c in sorted(output_clips,
                                key=lambda x: x['virality_score'], reverse=True):
                    self.log.emit(f"   {c['virality_score']:.3f} | {c['filename']}")

            EmotionDetector.cleanup()
            try:
                gc.collect(); time.sleep(1)
                shutil.rmtree(self.work_dir, ignore_errors=True)
            except Exception:
                pass
            self.finished.emit(output_clips)

        except Exception as e:
            self.log.emit(f"❌ {e}")
            import traceback; self.log.emit(traceback.format_exc())
            shutil.rmtree(self.work_dir, ignore_errors=True)
            if self.emit_jobs_only:
                self.jobs_ready.emit({})
            else:
                self.finished.emit([])

    # ── Download / transcribe ─────────────────────────────────

    def progress_hook(self, d):
        if d.get('status') == 'downloading':
            try:
                pct = float(d.get('_percent_str', '0%').replace('%', '').strip())
                self.progress.emit(int(pct * 0.25))
                if int(pct) >= self._last_download_log + 5:
                    self._last_download_log = int(pct) - (int(pct) % 5)
                    self.log.emit(f"📥 {int(pct)}% ({d.get('_speed_str','N/A')})")
            except Exception:
                pass
        elif d.get('status') == 'finished' and not self._download_finished:
            self._download_finished = True
            self.log.emit("✅ Скачано!")

    def download_youtube(self, tmpdir):
        qmap = {
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "720p":  "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "480p":  "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "360p":  "bestvideo[height<=360]+bestaudio/best[height<=360]",
        }
        import yt_dlp
        opts = {
            'format': qmap.get(self.quality, 'best'),
            'outtmpl': os.path.join(tmpdir, 'video.%(ext)s'),
            'quiet': True, 'no_warnings': True, 'merge_output_format': 'mp4',
            'ffmpeg_location': FFMPEG_EXE,
            'progress_hooks': [self.progress_hook],
            'writesubtitles': True, 'writeautomaticsub': True,
            'subtitleslangs': ['ru', 'en'], 'subtitlesformat': 'srt',
        }
        if 'youtube.com' in self.url or 'youtu.be' in self.url:
            opts['http_headers'] = {'User-Agent': 'Mozilla/5.0'}
            # Не форсируем player_client: жёсткий ['web','android'] лишал yt-dlp
            # доступа к форматам выше 360p (android сейчас режется YouTube SABR-
            # экспериментом) — стандартный автовыбор клиента у yt-dlp видит 4K.
        info = None
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
        except Exception as e:
            if '429' in str(e):
                self.log.emit("⚠️ 429 Too Many Requests — повтор без субтитров...")
                opts['writesubtitles'] = opts['writeautomaticsub'] = False
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(self.url, download=True)
            else:
                self.log.emit(f"❌ yt-dlp: {e}")
                raise
        if info:
            self.video_title = info.get('title', '')
        video_file = None
        for f in os.listdir(tmpdir):
            if f.endswith('.mp4'):
                video_file = os.path.join(tmpdir, f)
            elif f.endswith('.srt'):
                self.srt_path = os.path.join(tmpdir, f)
        if not video_file:
            raise Exception("Видео не найдено")
        return video_file

    def parse_youtube_srt(self, srt_path):
        words, full_text = [], ""
        if not os.path.exists(srt_path):
            return full_text, words
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        pat = r'(\d+:\d{2}:\d{2},\d{3}) --> (\d+:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)'
        def to_sec(t):
            h2, m, s = t.split(':'); s, ms = s.split(',')
            return int(h2)*3600 + int(m)*60 + int(s) + int(ms)/1000
        for ss, es, txt in re.findall(pat, content, re.DOTALL):
            s, e = to_sec(ss), to_sec(es)
            clean = re.sub(r'<[^>]+>', '', txt).strip()
            full_text += clean + " "
            wl = clean.split()
            if wl:
                dur2 = (e - s) / len(wl)
                for i, w in enumerate(wl):
                    words.append({"word": w, "start": s+i*dur2, "end": s+(i+1)*dur2})
        return full_text, words

    def transcribe(self, video_path):
        # Whisper даёт заметно точнее тайминг/текст, чем автосубтитры YouTube —
        # предпочитаем его, а субтитры YouTube оставляем как запасной вариант,
        # если Whisper недоступен/упал.
        with _cuda_path_for_ctranslate2():
            try:
                model = WhisperModel("large-v3", device="cuda", compute_type="float16")
                self.log.emit("🔊 Whisper large-v3 (CUDA)...")
            except Exception:
                model = None
            if model is None:
                try:
                    model = WhisperModel("small", device="cpu", compute_type="int8")
                    self.log.emit("🔊 Whisper small (CPU, GPU недоступен)...")
                except Exception:
                    model = None

            if model is not None:
                # text/words — СНАРУЖИ try: если генератор сегментов упадёт на середине
                # длинного видео (например, не хватило VRAM), то, что уже успели
                # распознать, не должно тонуть вместе с исключением.
                text, words = "", []
                try:
                    segs, _ = BatchedInferencePipeline(model=model).transcribe(
                        video_path, batch_size=16,
                        language=None if self.language == "Авто" else self.language.lower(),
                        word_timestamps=True, vad_filter=True
                    )
                    for seg in segs:
                        text += seg.text + " "
                        if seg.words:
                            for w in seg.words:
                                words.append({"word": w.word, "start": w.start, "end": w.end})
                except Exception as e:
                    self.log.emit(f"⚠️ Whisper прервался на середине ({e}) — использую то, что успел распознать")
                finally:
                    del model; gc.collect()

                if words:
                    self.log.emit(f"✅ Слов: {len(words)}")
                    return text, words

        if self.srt_path and os.path.exists(self.srt_path):
            self.log.emit("📝 Использую субтитры YouTube (запасной вариант)...")
            text, words = self.parse_youtube_srt(self.srt_path)
            if words:
                self.log.emit(f"✅ Слов: {len(words)}")
                return text, words

        return "", []

    # ── LLM-driven planning ────────────────────────────────────
    # Модель — Qwen2.5-instruct, использует формат ChatML (не Mistral [INST]),
    # см. её собственный tokenizer.chat_template и eos_token=<|im_end|>.

    CHAT_STOP = ["<|im_end|>", "<|endoftext|>"]

    @staticmethod
    def _chat_prompt(user_content: str) -> str:
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    # Раньше (на CPU, репак-квантизация Q4_K_8x8) запрос впритык к n_ctx иногда
    # валил процесс нативным крашем (0xC0000409) вместо исключения, которое можно
    # поймать — отсюда запас от края n_ctx=32768. Также пробовали полностью
    # убрать нарезку на части, подняв n_ctx до 131072 — это оказалось ошибкой
    # вдвойне: (1) реальный обученный контекст модели — только 32768 (см.
    # комментарий в load_llm), выше — неопределённое поведение; (2) даже если бы
    # контекст позволял, один гигантский промпт (120К символов) на реальном
    # 165-минутном видео обрабатывался 6088 секунд и ВСЁ РАВНО обрывался по
    # лимиту токенов — attention растёт квадратично от длины контекста, так что
    # один большой вызов кратно медленнее нескольких небольших, а обрыв
    # единственного гигантского чанка теряет большую часть видео вместо
    # маленького кусочка. MAX_PROMPT_TOKENS держим с запасом от n_ctx (32768 -
    # 8000 на ответ - safety); реальный потолок размера чанка на длинных видео
    # задаёт MAX_CHUNK_DURATION ниже.
    MAX_PROMPT_TOKENS = 20000
    # Верхняя граница длительности видео на один чанк для LLM — независимо от
    # того, сколько токенов это реально занимает. Даже если бы бюджет токенов
    # позволял впихнуть двухчасовое видео целиком, cам вызов становится
    # неприемлемо медленным и куда рискованнее (обрыв теряет половину видео).
    MAX_CHUNK_DURATION = 1800  # 30 минут

    def _fit_text_to_context(self, text: str, prompt_template: str, max_tokens: int, safety: int = 500) -> str:
        """Обрезает text по фактическому числу токенов модели (а не наугад по
        символам), но не приближается к границе n_ctx — см. MAX_PROMPT_TOKENS.
        prompt_template — тот же промпт с text="" внутри, чтобы посчитать
        накладные расходы (инструкции, список сцен и т.п.)."""
        n_ctx = self._llm.n_ctx()
        other_tokens = len(self._llm.tokenize(prompt_template.encode("utf-8"), add_bos=True))
        budget = min(n_ctx - other_tokens - max_tokens - safety, self.MAX_PROMPT_TOKENS)
        if budget <= 0:
            return ""
        tokens = self._llm.tokenize(text.encode("utf-8"), add_bos=False)
        if len(tokens) <= budget:
            return text
        return self._llm.detokenize(tokens[:budget]).decode("utf-8", errors="ignore")

    @staticmethod
    def _words_to_timestamped_text(words, marker_interval=8.0):
        """Вставляет метки времени вида [12.3s] через каждые marker_interval секунд.
        Без них LLM не имеет вообще никакой привязки текста ко времени и вынуждена
        линейно интерполировать start_time/end_time по положению в строке — это
        сильно врёт, если в видео есть паузы, музыка или молчание. С метками модель
        может просто прочитать нужное число рядом с интересующим её словом."""
        if not words:
            return ""
        parts = []
        last_marker = -marker_interval
        for w in words:
            if w['start'] - last_marker >= marker_interval:
                parts.append(f"[{w['start']:.1f}s]")
                last_marker = w['start']
            parts.append(w['word'].strip())
        return " ".join(parts)

    def _chunk_words_by_context(self, words):
        """Раньше весь транскрипт целиком обрезался до ~12000 токенов
        (_fit_text_to_context брал только начало) — для длинного видео (часовой
        эфир, стрим) LLM в принципе не видела вторую половину и не могла найти
        там хайлайты. Вместо обрезки режем транскрипт на последовательные части
        по бюджету контекста — каждая часть проходит через LLM отдельно, и в
        сумме отсматривается всё видео целиком."""
        if not words:
            return [None]
        full_text = self._words_to_timestamped_text(words)
        total_tokens = len(self._llm.tokenize(full_text.encode("utf-8"), add_bos=False))
        token_budget = self.MAX_PROMPT_TOKENS - 1500  # запас под сам промпт/инструкции/ответ
        total_span = words[-1]['end'] - words[0]['start']

        # Раньше резали только по бюджету токенов — для очень длинных видео это
        # позволяло одному чанку растянуться на многие часы контента. Из-за
        # квадратичной стоимости attention такой гигантский вызов становится
        # непропорционально медленным (на реальном 165-минутном видео — 1.7 часа
        # на один вызов) и с высокой вероятностью всё равно обрывается по лимиту
        # токенов, теряя большую часть видео разом. Дополнительно ограничиваем
        # чанк по РЕАЛЬНОЙ длительности — так каждый вызов остаётся быстрым, а
        # обрыв (если случится) теряет лишь небольшой кусок, а не половину ролика.
        if total_tokens <= token_budget and total_span <= self.MAX_CHUNK_DURATION:
            return [words]

        n_by_tokens   = -(-total_tokens // token_budget)          # ceil
        n_by_duration = -(-int(total_span) // self.MAX_CHUNK_DURATION)  # ceil
        n_chunks = max(n_by_tokens, n_by_duration, 1)
        chunk_size = -(-len(words) // n_chunks)  # ceil
        return [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    def _llm_propose_highlights(self, video_text, duration, scene_text, count):
        def build_prompt(txt):
            return self._chat_prompt(
                f"Найди лучшие моменты для Shorts — все, что заслуживают внимания "
                f"(общая длина видео {duration:.0f}с, каждый момент ~{self.clip_duration}с±3с). "
                f"Из них позже отберут {count} самых сильных, так что не сдерживай себя "
                f"числом — лучше найти больше кандидатов, чем пропустить хороший момент.\n"
                f"{'Сцены:\n'+scene_text+chr(10) if scene_text else ''}"
                f"В тексте метки вида [12.3s] стоят перед словом, произнесённым в эту "
                f"секунду — ориентируйся по ним, чтобы точно указать start_time и end_time.\n"
                f'Текст: "{txt}"\n'
                f'Ответь ТОЛЬКО валидным JSON без пояснений, reason — 2-4 слова, не '
                f'предложение: [{{"start_time":число,"end_time":число,"reason":"кратко"}},...]'
            )

        # Модель категорически не слушается запрошенного количества — просили
        # "ровно 15", в тестах на реальных транскриптах получали от 12 до 101
        # объектов на один и тот же чанк. Бороться с этим бессмысленно (не
        # похоже, что промпт может это исправить), а недостаточный max_tokens
        # обрезал JSON-массив до закрывающей ']' на границе токенов — и терялся
        # не "лишний хвост", а систематически ВСЁ, что модель не успела
        # сгенерировать, то есть кандидаты из более поздней части чанка. Из-за
        # этого хайлайты стабильно смещались к началу каждого куска транскрипта.
        # Вместо борьбы с моделью — принимаем, что кандидатов может быть много
        # (лишние всё равно отсеются скорингом ниже), и даём щедрый запас
        # токенов, чтобы генерация почти всегда успевала закончиться сама.
        # Короткий reason держит стоимость одного кандидата низкой; _parse_
        # highlights_json — подстраховка на случай, если обрыв всё же случится.
        max_tokens = 8000
        fitted_text = self._fit_text_to_context(video_text, build_prompt(""), max_tokens)
        prompt = build_prompt(fitted_text)
        self.log.emit(f"📚 Контекст LLM: {len(fitted_text)}/{len(video_text)} символов")
        out = self._llm(prompt, max_tokens=max_tokens, temperature=0.7, stop=self.CHAT_STOP, echo=False)
        raw = out['choices'][0]['text']
        finish = out['choices'][0].get('finish_reason')
        items = self._parse_highlights_json(raw)
        if finish == 'length' and items:
            self.log.emit(
                f"⚠️ Ответ LLM обрезан по лимиту токенов — спасено {len(items)} "
                f"момент(ов) из частично оборванного JSON (смещены к началу чанка)"
            )
        return items

    @staticmethod
    def _parse_highlights_json(raw: str) -> list:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        # Массив не закрылся (ответ обрезан по max_tokens до конца генерации) —
        # вместо того чтобы терять все кандидаты чанка, вытаскиваем по отдельности
        # уже полностью сформированные объекты {...} до места обрыва.
        items = []
        for obj in re.finditer(r'\{[^{}]*\}', raw, re.DOTALL):
            try:
                items.append(json.loads(obj.group()))
            except Exception:
                continue
        return items

    def find_highlights(self, text, duration, scenes, words, count=7):
        self.log.emit("🤔 LLM...")
        sc = "\n".join(
            f"Сцена {i+1}: {s:.1f}s—{e:.1f}s" for i, (s, e) in enumerate(scenes[:50])
        ) if scenes else ""

        chunks = self._chunk_words_by_context(words)
        if len(chunks) > 1:
            self.log.emit(
                f"📚 Транскрипт разбит на {len(chunks)} часте(й) — LLM пройдёт по всему "
                f"видео, а не только по началу (раньше длинные видео обрезались)"
            )

        highlights = []
        for i, chunk_words in enumerate(chunks):
            chunk_text = self._words_to_timestamped_text(chunk_words) if chunk_words else text
            raw = self._llm_propose_highlights(chunk_text, duration, sc, count)
            highlights.extend(raw)
            if len(chunks) > 1:
                self.log.emit(f"   часть {i+1}/{len(chunks)}: {len(raw)} кандидат(ов)")

        valid = []
        if scenes:
            for h in highlights:
                st, et = h.get('start_time', 0), h.get('end_time', 0)
                if 0 <= st < et <= duration:
                    # Снаппим границы момента к сцене, ТОЛЬКО если они уже и так
                    # близки к её краям (как во всех snap_to_* в analysis.py) — не
                    # если момент просто ЛЕЖИТ ВНУТРИ сцены. Раньше стояло
                    # "st >= ss-1 and et <= se+1", что верно для ЛЮБОГО короткого
                    # момента внутри сцены: для непрерывного кадра без монтажных
                    # склеек (обычное дело для влогов/подкастов) scenedetect отдаёт
                    # одну сцену на много минут, и momент схлопывался в границы ВСЕЙ
                    # сцены. Дальше шаг подгонки длины пересчитывал итоговый клип
                    # вокруг СЕРЕДИНЫ этой сцены, а не вокруг момента, который
                    # реально нашли LLM/скоринг виральности — разные кандидаты
                    # внутри одной длинной сцены схлопывались в один и тот же клип,
                    # а контент систематически смещался к той сцене, что ближе к
                    # началу видео.
                    for ss, se in scenes:
                        if abs(st - ss) <= 2.0 and abs(et - se) <= 2.0:
                            h['start_time'], h['end_time'] = ss, se; break
                    valid.append(h)
        else:
            valid = [h for h in highlights
                     if 0 <= h.get('start_time', 0) < h.get('end_time', 0) <= duration]

        before_dedup = len(valid)
        valid = self._dedupe_highlights(valid)
        if before_dedup - len(valid):
            self.log.emit(
                f"🧬 Убрано {before_dedup - len(valid)} дублирующихся момент(ов) "
                f"(привязка к одной и той же сцене/пересекающиеся тайм-коды)"
            )

        if self.sponsor_segments:
            before = len(valid)
            valid = [h for h in valid
                     if not self._overlaps_sponsor(h['start_time'], h['end_time'])]
            if before - len(valid):
                self.log.emit(f"🚫 Отфильтровано {before - len(valid)} момент(ов) с рекламой")

        if len(valid) < 2:
            seg = (duration - 20) / count
            for i in range(count):
                s = 10 + i * seg
                e = min(s + self.clip_duration, duration - 2)
                if self._overlaps_sponsor(s, e):
                    continue
                valid.append({"start_time": s, "end_time": e, "reason": f"Авто {i+1}"})
            valid = self._dedupe_highlights(valid)

        too_short = sum(1 for h in valid
                        if h['end_time'] - h['start_time'] < self.MIN_CLIP_DURATION)
        if too_short:
            valid = [self._expand_to_min_duration(h, duration) for h in valid]
            valid = self._dedupe_highlights(valid)
            self.log.emit(
                f"⏱️ Расширил {too_short} момент(ов) короче {self.MIN_CLIP_DURATION}с "
                f"(попали в короткую сцену/склейку)"
            )
        return sorted(valid[:count], key=lambda x: x['start_time'])

    def _expand_to_min_duration(self, h: dict, duration: float) -> dict:
        """Симметрично расширяет момент короче MIN_CLIP_DURATION вокруг его центра,
        не вылезая за границы видео."""
        st, et = h['start_time'], h['end_time']
        cur = et - st
        if cur >= self.MIN_CLIP_DURATION:
            return h
        need = self.MIN_CLIP_DURATION - cur
        st2 = max(0.0, st - need / 2)
        et2 = min(duration, st2 + self.MIN_CLIP_DURATION)
        st2 = max(0.0, et2 - self.MIN_CLIP_DURATION)
        h['start_time'], h['end_time'] = st2, et2
        return h

    @staticmethod
    def _extend_past_word(t: float, words: list) -> float:
        """Если t попадает строго внутрь произносимого слова — дотягивает до его
        конца, а не обрезает его на середине."""
        for w in words:
            if w['start'] < t < w['end']:
                return w['end']
        return t

    @staticmethod
    def _retreat_to_word_start(t: float, words: list) -> float:
        """Если t попадает строго внутрь произносимого слова — откатывает к его
        началу, чтобы клип не начинался с середины слова."""
        for w in words:
            if w['start'] < t < w['end']:
                return w['start']
        return t

    @staticmethod
    def _dedupe_highlights(highlights: list, overlap_ratio: float = 0.5) -> list:
        """Убирает моменты, которые более чем на overlap_ratio пересекаются с уже
        принятым — привязка к границам сцены может схлопнуть несколько разных
        предложений LLM в один и тот же диапазон, и без этой проверки они бы
        превратились в несколько клипов с одинаковым содержимым."""
        result = []
        for h in sorted(highlights, key=lambda x: x['start_time']):
            st, et = h['start_time'], h['end_time']
            is_dup = False
            for r in result:
                rst, ret = r['start_time'], r['end_time']
                overlap = max(0.0, min(et, ret) - max(st, rst))
                union   = max(et, ret) - min(st, rst)
                if union > 0 and overlap / union > overlap_ratio:
                    is_dup = True
                    break
            if not is_dup:
                result.append(h)
        return result

    def _overlaps_sponsor(self, start: float, end: float) -> bool:
        if not self.sponsor_segments:
            return False
        overlap = sum(max(0, min(end, e) - max(start, s)) for s, e in self.sponsor_segments)
        return overlap > (end - start) * 0.25

    # LLM изредка срывается в другой алфавит посреди генерации (в основном
    # китайские иероглифы — сказывается обучающие данные Qwen) — такой текст
    # попал бы прямо на экран как оверлей поверх видео. \w в Unicode-регексе
    # считает иероглифы "словом", так что вместо чёрного списка — явный
    # белый список разрешённых символов (латиница, кириллица, цифры, базовая
    # пунктуация).
    _ALLOWED_TEXT_CHARS = re.compile(r'[^a-zA-Zа-яА-ЯёЁ0-9\s.,!?:;\-—«»\'"()%№&+/]')

    @classmethod
    def _strip_foreign_script(cls, text: str) -> str:
        return cls._ALLOWED_TEXT_CHARS.sub('', text)

    def generate_clip_title(self, full_text, highlight, words):
        mw = [w['word'] for w in words
              if highlight['start_time'] <= w['start'] <= highlight['end_time']]
        mt = " ".join(mw)[:600] or full_text[:600]

        def _ask(extra_instruction=""):
            avoid = ""
            recent = self._generated_titles[-6:]
            if recent:
                avoid = (
                    "\nУже придуманные названия для других клипов этого видео — "
                    "не повторяй их структуру, первые 2-3 слова или приём, даже "
                    "если меняешь последнее слово:\n"
                    + "\n".join(f"- {t}" for t in recent)
                )
            out = self._llm(
                self._chat_prompt(
                    "Придумай короткое (4-8 слов) цепляющее название на русском для "
                    "этого фрагмента видео. Каждый раз используй РАЗНЫЙ приём: то живая "
                    "цитата/реплика персонажа, то конкретная деталь из фрагмента, то "
                    "реакция, то интрига-вопрос. НЕ начинай с шаблонных клише вроде "
                    "«Как...», «Секрет...», «Вы не поверите...», «Один трюк...» — "
                    "заголовок должен быть конкретным, про ЭТОТ фрагмент, а не общим. "
                    "Ответь только названием, без кавычек и пояснений."
                    f'{avoid}{extra_instruction}\nФрагмент: "{mt}"'
                ),
                max_tokens=25, temperature=0.95, top_p=0.92,
                stop=self.CHAT_STOP + ["\n"], echo=False
            )
            t = out['choices'][0]['text'].strip()
            t = re.sub(r'^(Название:?|Вариант\s*\d*:?)\s*', '', t, flags=re.IGNORECASE)
            t = re.sub(r'^\d+[\.\)]\s*', '', t).split('\n')[0].strip().strip('"\'«»:.-')
            t = re.sub(r'[<>:"/\\|?*]', '', t)[:70]
            return self._strip_foreign_script(t).strip()

        used = {t.lower() for t in self._generated_titles}
        title = _ask()
        if title.lower() in used:
            # Модель иногда буквально повторяет уже придуманное название слово в
            # слово, несмотря на прямой запрет в промпте (например, "Тонкий руль
            # мудрости" дважды подряд для двух разных клипов одного видео) — даём
            # ей ещё одну попытку с явным указанием, какое название под запретом,
            # прежде чем сдаться на общий фолбэк по времени.
            title = _ask(f'\nНазвание "{title}" уже использовано для другого клипа — '
                         f'придумай ДРУГОЕ, не похожее.')
        if not title or len(title) < 5 or title.lower() in used:
            title = f"Момент в {int(highlight['start_time'])}с"
        self._generated_titles.append(title)
        return title

    def generate_hook(self, full_text, highlight, words):
        mw = [w['word'] for w in words
              if highlight['start_time'] <= w['start'] <= highlight['end_time']]
        mt = " ".join(mw)[:500] or full_text[:500]
        out = self._llm(
            self._chat_prompt(
                "Создай ОДИН короткий интригующий хук (до 8 слов) для YouTube Shorts "
                f'на русском. Ответь только хуком, без кавычек и пояснений.\nФрагмент: "{mt}"'
            ),
            max_tokens=20, temperature=0.9, top_p=0.95,
            stop=self.CHAT_STOP + ["\n"], echo=False
        )
        hook = out['choices'][0]['text'].strip().strip('"\'«»').split('\n')[0].strip()[:80]
        hook = self._strip_foreign_script(hook).strip()
        if not hook or len(hook) < 5:
            hook = "Ты не поверишь что произошло..."
        self.log.emit(f"🪝 {hook}")
        return hook

    def audio_driven_zoom_plan(self, highlight, audio_energies):
        """Ищет пики аудио-энергии внутри клипа и строит по ним план зума —
        точнее совпадает с эмоциональными акцентами, чем догадки LLM по тексту."""
        st, et = highlight['start_time'], highlight['end_time']
        local = sorted((t - st, v) for t, v in audio_energies.items() if st <= t <= et)
        if len(local) < 5:
            return None
        values = [v for _, v in local]
        mean, std = float(np.mean(values)), float(np.std(values))
        if std < 1e-6:
            return None
        threshold = mean + 0.75 * std
        peaks = [
            (t, v) for i, (t, v) in enumerate(local)
            if v >= threshold and v == max(vv for _, vv in local[max(0, i - 2):i + 3])
        ]
        if not peaks:
            return None
        peaks.sort(key=lambda p: -p[1])
        selected = []
        for t, v in peaks:
            if all(abs(t - st2) > 3.0 for st2, _ in selected):
                selected.append((t, v))
            if len(selected) >= 3:
                break
        if not selected:
            return None
        vmax = max(v for _, v in selected)
        bi = self.zoom_intensity / 100 * 0.25
        return [
            {"time": t, "action": "zoom_in", "intensity": bi * (0.5 + 0.5 * (v / vmax))}
            for t, v in sorted(selected)
        ]

    def analyze_zoom_points(self, full_text, highlight, words, audio_energies=None):
        if audio_energies:
            plan = self.audio_driven_zoom_plan(highlight, audio_energies)
            if plan:
                return plan
        mw = [w['word'] for w in words
              if highlight['start_time'] <= w['start'] <= highlight['end_time']]
        mt = " ".join(mw)[:800]
        if not mt:
            return None
        dur = highlight['end_time'] - highlight['start_time']
        bi  = self.zoom_intensity / 100 * 0.25
        out = self._llm(
            self._chat_prompt(
                f"Найди 2-3 момента для зума в {dur:.0f}с клипе.\n"
                f'Текст: "{mt}"\n'
                f'Ответь ТОЛЬКО валидным JSON без пояснений: '
                f'[{{"time":сек,"action":"zoom_in/zoom_out/normal","intensity":0.05-0.2}}]'
            ),
            max_tokens=200, temperature=0.6, stop=self.CHAT_STOP, echo=False
        )
        raw = out['choices'][0]['text'].strip()
        try:
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if m:
                plan  = json.loads(m.group())
                valid = [
                    {'time': z['time'], 'action': z['action'],
                     'intensity': min(z.get('intensity', 0.1) * (self.zoom_intensity / 50), bi)}
                    for z in plan
                    if 0 <= z.get('time', -1) <= dur
                    and z.get('action') in ['zoom_in', 'zoom_out', 'normal']
                ]
                if valid:
                    return valid
        except Exception:
            pass
        return [
            {"time": dur * 0.3, "action": "zoom_in",  "intensity": bi * 0.6},
            {"time": dur * 0.7, "action": "zoom_out", "intensity": bi * 0.4},
        ]

    # Во сколько раз canvas (кадр перед зумом) больше финального разрешения.
    # Должен быть >= максимального зума ниже, чтобы окно кропа при зуме никогда
    # не оказывалось меньше финального кадра — тогда resize всегда downscale,
    # а не апскейл (апскейл — источник размытия при сильном зуме).
    ZOOM_CANVAS_MARGIN = 1.12
    ZOOM_RANGE = (0.92, 1.12)

    def apply_dynamic_zoom(self, clip, zoom_plan, out_w, out_h):
        import cv2
        src_w, src_h = clip.size
        margin = self.ZOOM_CANVAS_MARGIN
        zmin, zmax = self.ZOOM_RANGE

        def zoom_effect(get_frame, t):
            frame = get_frame(t)
            zoom  = 1.0
            if zoom_plan:
                contributions = []
                for plan in zoom_plan:
                    diff = abs(t - plan['time'])
                    if diff < 4.0:
                        smooth    = (1 + np.cos(np.pi * diff / 4.0)) / 2
                        intensity = plan['intensity'] * smooth
                        if plan['action'] == 'zoom_in':
                            contributions.append(intensity)
                        elif plan['action'] == 'zoom_out':
                            contributions.append(-intensity)
                if contributions:
                    # Берём самый сильный отдельный импульс, а не сумму всех —
                    # иначе близкие друг к другу точки зума складываются и
                    # эффект получается куда резче, чем задумывался.
                    zoom += max(contributions, key=abs)
            zoom = max(zmin, min(zmax, zoom))

            win_w = min(src_w, max(out_w, int(out_w * margin / zoom)))
            win_h = min(src_h, max(out_h, int(out_h * margin / zoom)))
            x1 = (src_w - win_w) // 2
            y1 = (src_h - win_h) // 2
            window = frame[y1:y1 + win_h, x1:x1 + win_w]

            if win_w == out_w and win_h == out_h:
                return window
            return cv2.resize(window, (out_w, out_h), interpolation=cv2.INTER_AREA)

        return clip.transform(zoom_effect)

    def _build_pil_font(self):
        from PIL import ImageFont
        fs   = FONT_SETTINGS
        dirs = [
            "C:/Windows/Fonts/", os.path.expanduser("~/.fonts/"),
            "/usr/share/fonts/truetype/", "/usr/share/fonts/",
            "/Library/Fonts/", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ]
        fam  = fs.font_family
        flow = fam.lower().replace(" ", "")
        variants = []
        if fs.bold and fs.italic:
            variants += [f"{fam} Bold Italic.ttf", f"{flow}bi.ttf", f"{flow}bolditalic.ttf"]
        if fs.bold:
            variants += [f"{fam} Bold.ttf", f"{fam}bd.ttf", f"{flow}bd.ttf",
                         f"{flow}bold.ttf", f"{flow}-bold.ttf"]
        if fs.italic:
            variants += [f"{fam} Italic.ttf", f"{flow}i.ttf", f"{flow}italic.ttf"]
        variants += [f"{fam}.ttf", f"{fam}.otf", f"{flow}.ttf",
                     "arialbd.ttf", "arial.ttf", "impact.ttf",
                     "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]
        for d in dirs:
            for v in variants:
                fp = os.path.join(d, v)
                if os.path.exists(fp):
                    try:
                        return ImageFont.truetype(fp, fs.font_size)
                    except Exception:
                        continue
        return ImageFont.load_default()

    @staticmethod
    def _resize_to_cover(clip, target_w, target_h, margin=1.6):
        """Один ресайз вместо двух последовательных (width, потом height) —
        каждый лишний проход интерполяции теряет резкость."""
        scale = max(target_w * margin / clip.w, target_h * margin / clip.h)
        return clip.resized(scale)

    @classmethod
    def _crop_to_fill(cls, clip, target_w, target_h, margin=1.5):
        resized = cls._resize_to_cover(clip, target_w, target_h, margin=margin)
        return resized.cropped(x_center=resized.w / 2, y_center=resized.h / 2,
                               width=target_w, height=target_h)

    def _pick_background_clip(self, duration):
        """Случайный ролик из BACKGROUND_FOOTAGE_DIR (Subway Surfers/песок/т.п. —
        пользователь сам кладёт файлы в эту папку), обрезанный/зациклённый под
        нужную длительность. None, если папка пуста — вызывающий код сам решает,
        что делать (лог + fallback)."""
        exts = ('.mp4', '.mov', '.mkv', '.webm')
        try:
            files = [f for f in os.listdir(BACKGROUND_FOOTAGE_DIR) if f.lower().endswith(exts)]
        except Exception:
            files = []
        if not files:
            return None
        path = os.path.join(BACKGROUND_FOOTAGE_DIR, random.choice(files))
        try:
            bg = VideoFileClip(path).without_audio()
        except Exception as e:
            self.log.emit(f"⚠️ Не смог открыть фоновое видео {path}: {e}")
            return None
        if bg.duration >= duration:
            start = random.uniform(0, bg.duration - duration)
            return bg.subclipped(start, start + duration)
        # Исходник короче нужной длины — зацикливаем.
        from moviepy import concatenate_videoclips
        loops, total = [], 0.0
        while total < duration:
            loops.append(bg.subclipped(0, bg.duration))
            total += bg.duration
        return concatenate_videoclips(loops).subclipped(0, duration)

    @staticmethod
    def _make_video_folder(title: str) -> str:
        """clips/<название видео>/ — своя папка на каждый обработанный видео-исходник,
        чтобы клипы разных видео (особенно при пакетной обработке) не мешались в одной куче."""
        # .strip(' .') ПОСЛЕ обрезки длины, не только до неё — иначе срез по [:80]
        # может оставить пробел на конце. Windows тихо обрезает такой пробел при
        # создании самой папки на диске, но Python-строка его сохраняет — и ffmpeg
        # потом не находит папку по чуть-чуть другому пути ("No such file or directory").
        safe = re.sub(r'[<>:"/\\|?*]', '', (title or '').strip())[:80].strip(' .')
        if not safe:
            safe = f"video_{int(time.time())}"
        if not os.path.exists(os.path.join(CLIPS_DIR, safe)):
            return safe
        n = 2
        while os.path.exists(os.path.join(CLIPS_DIR, f"{safe} ({n})")):
            n += 1
        return f"{safe} ({n})"

    def cut_and_caption(self, input_path, start, end, words,
                        title=None, zoom_plan=None, hook_text=None,
                        emotion_timeline=None, ffmpeg_threads=4, index=None):
        output_dir = os.path.join(CLIPS_DIR, self.clip_subdir) if self.clip_subdir else CLIPS_DIR
        os.makedirs(output_dir, exist_ok=True)
        safe = re.sub(r'[<>:"/\\|?*]', '', title or f"clip_{int(start)}")[:60].strip(' .')
        prefix = f"{index + 1:02d}_" if index is not None else ""
        out  = os.path.join(output_dir, f"{prefix}{safe}.mp4")
        self.log.emit("✂️ Нарезаю...")

        clip = None
        try:
            clip = VideoFileClip(input_path).subclipped(start, end)
            w_out, h_out = 1080, 1920

            if self.centered_layout_enabled:
                # Альтернативный формат: без кропа/зума/фокуса на лицах — исходник
                # целиком вписывается в кадр (contain-fit) и центрируется; сверху и
                # снизу — не чёрные полосы, а размытая затемнённая растяжка того же
                # кадра (cover-fit), похожая на тень/подложку, а не пустое поле.
                self.log.emit("🖼️ Формат «по центру» (тень по краям)...")
                from moviepy import CompositeVideoClip
                scale  = min(w_out / clip.w, h_out / clip.h)
                fitted = clip.resized(scale)

                def blur_shadow(get_frame, t):
                    import cv2
                    frame = get_frame(t)
                    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=30)
                    return (blurred.astype(np.float32) * 0.4).clip(0, 255).astype(np.uint8)

                bg = self._crop_to_fill(clip, w_out, h_out).transform(blur_shadow)
                clip = CompositeVideoClip([bg, fitted.with_position("center")], size=(w_out, h_out))
                if fitted.audio is not None:
                    clip = clip.with_audio(fitted.audio)
            elif self.split_screen_enabled:
                # Split-screen: исходник кропается в верхнюю половину экрана, в нижнюю —
                # случайный фрагмент фоновой "нарезки" (Subway Surfers/песок/т.п.).
                # Без зума/динамики — только статичный кроп в каждой половине.
                self.log.emit("📱 Формат Split-screen...")
                top_h = h_out // 2
                orig_audio = clip.audio
                top_clip = self._crop_to_fill(clip, w_out, top_h)
                bg_clip = self._pick_background_clip(clip.duration)
                if bg_clip is None:
                    self.log.emit(
                        f"⚠️ Папка background_footage/ пуста — фон будет чёрным. "
                        f"Положите туда mp4/mov/mkv/webm с нарезкой."
                    )
                    from moviepy import ColorClip
                    bg_clip = ColorClip(size=(w_out, top_h), color=(20, 20, 20), duration=clip.duration)
                else:
                    bg_clip = self._crop_to_fill(bg_clip, w_out, h_out - top_h)
                from moviepy import CompositeVideoClip
                clip = CompositeVideoClip(
                    [top_clip.with_position((0, 0)), bg_clip.with_position((0, top_h))],
                    size=(w_out, h_out)
                )
                if orig_audio is not None:
                    clip = clip.with_audio(orig_audio)
            else:
                # Если зум включён, кадрируем с запасом ZOOM_CANVAS_MARGIN и отдаём этот запас
                # под зум на этапе apply_dynamic_zoom — так зум всегда получается уменьшением
                # (downscale), а не апскейлом уже финального кадра, откуда раньше бралось размытие.
                zoom_margin = self.ZOOM_CANVAS_MARGIN if self.zoom_enabled else 1.0
                crop_w, crop_h = int(w_out * zoom_margin), int(h_out * zoom_margin)

                # ── Crop ──────────────────────────────────────
                # Один проход ресайза (вместо двух последовательных width→height) —
                # каждый лишний проход интерполяции размывает картинку.
                if self.face_crop_enabled:
                    self.log.emit("👤 Умный кроп...")
                    try:
                        resized = self._resize_to_cover(clip, crop_w, crop_h, margin=1.6)
                        clip = SmartFaceCrop(crop_w, crop_h, smoothing=0.97, max_step=8,
                                            multi_speaker=self.multi_speaker_crop).make_transform(resized)
                        self.log.emit("✅ Face crop!")
                    except Exception as e:
                        self.log.emit(f"⚠️ Face crop: {e} → стандартный кроп")
                        resized = self._resize_to_cover(clip, crop_w, crop_h, margin=1.5)
                        clip = resized.cropped(x_center=resized.w/2, y_center=resized.h/2,
                                              width=crop_w, height=crop_h)
                else:
                    resized = self._resize_to_cover(clip, crop_w, crop_h, margin=1.5)
                    clip = resized.cropped(x_center=resized.w/2, y_center=resized.h/2,
                                          width=crop_w, height=crop_h)

                # ── Zoom (или просто финальный ресайз canvas → w_out×h_out) ─
                if self.zoom_enabled:
                    self.log.emit("🎥 Зум...")
                    try:
                        clip = self.apply_dynamic_zoom(clip, zoom_plan, w_out, h_out)
                    except Exception as e:
                        self.log.emit(f"⚠️ Зум: {e}")
                        clip = clip.resized((w_out, h_out))

            # ── Subtitles ─────────────────────────────────────
            self.log.emit("💬 Субтитры...")
            subtitle_words = [
                (ws - start, we - start, wt)
                for ws, we, wt in [(w2['start'], w2['end'], w2['word']) for w2 in words]
                if start <= ws <= end
            ]
            if subtitle_words:
                fs2 = FONT_SETTINGS; wpf = fs2.words_per_phrase
                groups, current = [], []
                for ws, we, wt in subtitle_words:
                    current.append((ws, we, wt))
                    if len(current) >= wpf:
                        groups.append(current); current = []
                    elif we > (current[0][0] if current else 0) + 2.5 and current:
                        groups.append(current); current = []
                if current:
                    groups.append(current)
                self.log.emit(f"📝 {len(groups)} фраз | {fs2.font_family} {fs2.font_size}px")
                try:
                    font     = self._build_pil_font()
                    renderer = SubtitleRenderer(font, fs2, w_out, h_out)
                    emotion_color_map = {
                        'happy':    (255, 220,  50),
                        'surprise': (255, 165,   0),
                        'fear':     (200, 100, 255),
                        'angry':    (255,  80,  80),
                        'sad':      (100, 150, 255),
                        'disgust':  (150, 200, 100),
                        'neutral':  tuple(fs2.text_color),
                    }

                    def get_emotion_color(t_abs):
                        if not emotion_timeline:
                            return tuple(fs2.text_color)
                        closest = min(emotion_timeline,
                                      key=lambda et2: abs(et2 - t_abs), default=None)
                        return emotion_color_map.get(
                            emotion_timeline.get(closest, 'neutral'),
                            tuple(fs2.text_color)
                        )

                    groups_meta = [(g[0][0], g[-1][1], g) for g in groups]

                    def add_subtitles(get_frame, t):
                        frame = get_frame(t)
                        group = next(
                            (g for gs, ge, g in groups_meta if gs <= t <= ge), None
                        )
                        if not group:
                            return frame
                        tc = get_emotion_color(start + t)
                        if fs2.karaoke_enabled:
                            words_list = [wt for _, _, wt in group]
                            active_idx = next(
                                (i for i, (ws, we, _) in enumerate(group) if ws <= t <= we),
                                len(words_list) - 1
                            )
                            overlay = renderer.get_karaoke_overlay(words_list, active_idx, tc)
                        else:
                            phrase  = " ".join(wt for _, _, wt in group)
                            overlay = renderer.get_overlay(phrase, tc)
                        return SubtitleRenderer.blend(frame, overlay)

                    clip = clip.transform(add_subtitles)
                    self.log.emit("✅ Субтитры!")
                except Exception as e:
                    self.log.emit(f"⚠️ Субтитры: {e}")
                    import traceback; self.log.emit(traceback.format_exc())

            # ── Hook / заголовок сверху ─────────────────────────
            if self.centered_layout_enabled and title:
                # В формате «по центру» вместо мигающего 3-секундного хука —
                # постоянный заголовок сверху на весь ролик (title клипа).
                self.log.emit("🏷️ Заголовок сверху...")
                try:
                    clip = HookOverlay(title, duration=clip.duration, persistent=True).apply(clip)
                    self.log.emit("✅ Заголовок!")
                except Exception as e:
                    self.log.emit(f"⚠️ Заголовок: {e}")
            elif hook_text and self.hook_enabled:
                self.log.emit("🪝 Хук...")
                try:
                    clip = HookOverlay(hook_text, duration=3.0).apply(clip)
                    self.log.emit("✅ Хук!")
                except Exception as e:
                    self.log.emit(f"⚠️ Хук: {e}")

            # ── Export ────────────────────────────────────────
            self.log.emit("💾 Сохраняю...")
            clip.write_videofile(
                out, codec="libx264", audio_codec="aac",
                bitrate="10000k", audio_bitrate="192k",
                preset="slow", fps=30, threads=ffmpeg_threads, logger=None,
            )
            clip.close(); gc.collect()
            self.log.emit(f"✅ {safe}.mp4")
            return out

        finally:
            if clip:
                try: clip.close()
                except Exception: pass
            gc.collect()
