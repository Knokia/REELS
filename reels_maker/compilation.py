"""Сборка длинного горизонтального видео (16:9) из найденных моментов —
альтернатива вертикальным Shorts-клипам. Отдельный модуль: основной пайплайн
нарезки его не импортирует, пока пользователь не включил режим «Компиляция».
"""
import os
import re

import numpy as np
from moviepy import ColorClip, CompositeVideoClip, VideoFileClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

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


def build_compilation(video_path: str, render_jobs: list, output_dir: str,
                      comp_title: str, ffmpeg_threads: int = 4,
                      log_cb=None, progress_cb=None) -> str:
    """Склеивает сегменты (по одному на каждый найденный момент, в хронологии)
    в одно 16:9 видео с плашкой-названием в начале каждого сегмента."""
    log = log_cb or (lambda *_: None)
    safe = re.sub(r'[<>:"/\\|?*]', '', comp_title).strip()[:70] or "compilation"
    out_path = os.path.join(output_dir, f"Компиляция — {safe}.mp4")

    source = VideoFileClip(video_path)
    segments = []
    try:
        jobs = sorted(render_jobs, key=lambda j: j['highlight']['start_time'])
        for i, job in enumerate(jobs, 1):
            h = job['highlight']
            seg = _fit_1080p(source.subclipped(h['start_time'], h['end_time']))
            banner = _make_banner(f"{i}/{len(jobs)} · {job['title']}")

            def with_banner(get_frame, t, _banner=banner):
                frame = get_frame(t)
                if t <= BANNER_SECONDS:
                    return SubtitleRenderer.blend(frame, _banner)
                return frame

            segments.append(seg.transform(with_banner))
            log(f"   🧩 Сегмент {i}/{len(jobs)}: {job['title']} "
                f"({h['end_time'] - h['start_time']:.0f}с)")
            if progress_cb:
                progress_cb(i / (len(jobs) + 1) * 0.3)

        final = concatenate_videoclips(segments)
        total = final.duration
        log(f"💾 Рендер компиляции: {total / 60:.1f} мин, 1920x1080...")
        final.write_videofile(
            out_path, codec="libx264", audio_codec="aac",
            bitrate="8000k", audio_bitrate="192k",
            preset="medium", fps=30, threads=ffmpeg_threads, logger=None,
        )
        final.close()
    finally:
        source.close()
    return out_path
