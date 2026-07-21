"""Сборка длинного горизонтального видео (16:9) из найденных моментов —
альтернатива вертикальным Shorts-клипам. Отдельный модуль: основной пайплайн
нарезки его не импортирует, пока пользователь не включил режим «Компиляция».
"""
import os
import re

import numpy as np
from moviepy import ColorClip, CompositeVideoClip, VideoFileClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import QThread, pyqtSignal

from .subtitles import SubtitleRenderer

TARGET_W, TARGET_H = 1920, 1080
BANNER_SECONDS = 3.5  # сколько секунд в начале сегмента висит плашка с названием


def _load_font(size: int):
    for fp in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _make_banner(text: str, w: int = TARGET_W, h: int = TARGET_H) -> np.ndarray:
    """RGBA-оверлей: тёмная плашка с названием момента в левом нижнем углу."""
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(52)
    text = text[:80]
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th = len(text) * 26, 52
    pad = 24
    x0, y1 = 60, h - 90
    y0 = y1 - th - pad * 2
    draw.rounded_rectangle(
        [(x0, y0), (x0 + tw + pad * 2, y1)], radius=14, fill=(10, 10, 10, 215)
    )
    draw.text((x0 + pad, y0 + pad), text, font=font, fill=(255, 255, 255, 255))
    return np.array(overlay)


def _fit_1080p(clip):
    """Вписывает сегмент в 1920x1080 без обрезки (contain-fit на чёрном фоне);
    если исходник и так 16:9 — это просто ресайз без полей."""
    if clip.w == TARGET_W and clip.h == TARGET_H:
        return clip
    scale = min(TARGET_W / clip.w, TARGET_H / clip.h)
    fitted = clip.resized(scale)
    if abs(fitted.w - TARGET_W) <= 2 and abs(fitted.h - TARGET_H) <= 2:
        return fitted.resized((TARGET_W, TARGET_H))
    bg = ColorClip(size=(TARGET_W, TARGET_H), color=(0, 0, 0), duration=clip.duration)
    out = CompositeVideoClip([bg, fitted.with_position("center")],
                             size=(TARGET_W, TARGET_H))
    if fitted.audio is not None:
        out = out.with_audio(fitted.audio)
    return out


def _safe_filename(comp_title: str) -> str:
    # .strip(' .') ПОСЛЕ обрезки длины, не только до неё — срез по [:70] может
    # оставить пробел на конце, а Windows тихо режет такой пробел у реального
    # файла на диске, из-за чего ffmpeg потом не находит путь, который построил
    # Python (см. тот же фикс в pipeline.py _make_video_folder/cut_and_caption).
    return re.sub(r'[<>:"/\\|?*]', '', comp_title).strip()[:70].strip(' .') or "compilation"


def build_compilation(video_path: str, render_jobs: list, output_dir: str,
                      comp_title: str, ffmpeg_threads: int = 4,
                      log_cb=None, progress_cb=None) -> str:
    """Склеивает сегменты одного видео в одно 16:9 видео с плашкой-названием
    в начале каждого сегмента. Частный случай build_multi_compilation с одним
    источником — оставлен отдельной функцией как более простой вход для
    однo-видео пайплайна в pipeline.py."""
    return build_multi_compilation(
        [{'video_path': video_path, 'video_title': '', 'render_jobs': render_jobs}],
        output_dir, comp_title, ffmpeg_threads=ffmpeg_threads,
        log_cb=log_cb, progress_cb=progress_cb,
    )


def build_multi_compilation(sources: list, output_dir: str, comp_title: str,
                            ffmpeg_threads: int = 4,
                            log_cb=None, progress_cb=None) -> str:
    """Как build_compilation, но склеивает сегменты сразу из нескольких видео
    в один файл. sources — список {'video_path', 'video_title', 'render_jobs'},
    источники идут в порядке списка, сегменты внутри каждого — хронологически."""
    log = log_cb or (lambda *_: None)
    safe = _safe_filename(comp_title)
    out_path = os.path.join(output_dir, f"Компиляция — {safe}.mp4")

    total = sum(len(src['render_jobs']) for src in sources)
    if total == 0:
        raise ValueError("Нет ни одного момента для компиляции")

    opened = []
    segments = []
    seg_i = 0
    try:
        for src in sources:
            source = VideoFileClip(src['video_path'])
            opened.append(source)
            jobs = sorted(src['render_jobs'], key=lambda j: j['highlight']['start_time'])
            for job in jobs:
                seg_i += 1
                h = job['highlight']
                seg = _fit_1080p(source.subclipped(h['start_time'], h['end_time']))
                label = f"{seg_i}/{total} · {job['title']}"
                if src.get('video_title'):
                    label = f"{seg_i}/{total} · {src['video_title']}: {job['title']}"
                banner = _make_banner(label)

                def with_banner(get_frame, t, _banner=banner):
                    frame = get_frame(t)
                    if t <= BANNER_SECONDS:
                        return SubtitleRenderer.blend(frame, _banner)
                    return frame

                segments.append(seg.transform(with_banner))
                log(f"   🧩 {label} ({h['end_time'] - h['start_time']:.0f}с)")
                if progress_cb:
                    progress_cb(seg_i / (total + 1) * 0.3)

        final = concatenate_videoclips(segments)
        dur = final.duration
        log(f"💾 Рендер компиляции: {dur / 60:.1f} мин, 1920x1080, "
            f"источников: {len(sources)}...")
        final.write_videofile(
            out_path, codec="libx264", audio_codec="aac",
            bitrate="8000k", audio_bitrate="192k",
            preset="medium", fps=30, threads=ffmpeg_threads, logger=None,
        )
        final.close()
    finally:
        for source in opened:
            source.close()
    return out_path


class CompilationMergeThread(QThread):
    """Финальная склейка нескольких уже проанализированных источников в один
    файл — вынесена в отдельный QThread, чтобы UI не подвисал во время рендера
    (сборка идёт после того как ProcessingThread каждого источника уже собрал
    свои render_jobs через emit_jobs_only=True)."""
    log      = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, sources: list, output_dir: str, comp_title: str):
        super().__init__()
        self.sources    = sources
        self.output_dir = output_dir
        self.comp_title = comp_title

    def run(self):
        try:
            path = build_multi_compilation(
                self.sources, self.output_dir, self.comp_title,
                ffmpeg_threads=max(1, os.cpu_count() or 4),
                log_cb=self.log.emit,
                progress_cb=lambda p: self.progress.emit(int(p * 100)),
            )
            with VideoFileClip(path) as c:
                dur = c.duration
            self.finished.emit({
                'path':           path,
                'title':          f"Компиляция — {self.comp_title}",
                'filename':       os.path.basename(path),
                'start': 0.0, 'end': dur, 'duration': dur,
                'reason':         'multi_compilation',
                'virality_score': 0.0,
                'hook':           '',
            })
        except Exception as e:
            self.log.emit(f"❌ {e}")
            import traceback; self.log.emit(traceback.format_exc())
            self.finished.emit(None)
        finally:
            for src in self.sources:
                try:
                    import shutil
                    shutil.rmtree(src['work_dir'], ignore_errors=True)
                except Exception:
                    pass
