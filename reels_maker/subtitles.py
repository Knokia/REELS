import numpy as np

from .config import FontSettings


class SubtitleRenderer:
    """
    Заранее рендерит каждую уникальную фразу в RGBA numpy-массив.
    При наложении — векторный alpha-blend без PIL на горячем пути.
    Длинные фразы переносятся на несколько строк, чтобы не вылезать за края кадра.
    """

    def __init__(self, font, fs: FontSettings, frame_w: int, frame_h: int):
        self.font  = font
        self.fs    = fs
        self.w     = frame_w
        self.h     = frame_h
        self._cache: dict = {}
        try:
            self.pos_pct = int(fs.position.replace('%', '')) / 100
        except Exception:
            self.pos_pct = 0.75

    def _wrap_words(self, draw, words: list, max_width: int) -> list:
        """Разбивает список слов на строки так, чтобы каждая влезала в max_width."""
        lines, line = [], []
        for word in words:
            test = " ".join(line + [word])
            try:
                tw = draw.textlength(test, font=self.font)
            except Exception:
                tw = len(test) * self.fs.font_size * 0.5
            if line and tw > max_width:
                lines.append(line)
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(line)
        return lines or [[""]]

    def _draw_lines(self, draw, lines: list, text_color: tuple,
                    active_pos=None, karaoke_color: tuple = None):
        """Рисует уже перенесённые по строкам слова с общим фоном под всем блоком.
        active_pos=(line_idx, word_idx) — если задан, слово подсвечивается karaoke_color
        (караоке-режим), иначе строки рисуются целиком (обычный режим)."""
        fs = self.fs
        try:
            ascent, descent = self.font.getmetrics()
            line_h = ascent + descent
        except Exception:
            line_h = fs.font_size + 10
        line_spacing = int(line_h * 0.25)

        widths = []
        for line in lines:
            text = " ".join(line)
            try:
                widths.append(draw.textlength(text, font=self.font))
            except Exception:
                widths.append(len(text) * fs.font_size * 0.5)
        block_w = max(widths) if widths else 0
        block_h = line_h * len(lines) + line_spacing * (len(lines) - 1)

        pad  = 15
        bg_w = int(block_w) + pad * 2
        bg_h = int(block_h) + pad * 2
        x0   = (self.w - bg_w) // 2
        y0   = int(self.h * self.pos_pct) - bg_h // 2

        if fs.bg_enabled:
            bc = fs.bg_color
            draw.rounded_rectangle(
                [(x0, y0), (x0 + bg_w, y0 + bg_h)], radius=12,
                fill=(bc[0], bc[1], bc[2], fs.bg_alpha)
            )

        sw, sc = fs.stroke_width, fs.stroke_color

        for li, (line, lw) in enumerate(zip(lines, widths)):
            ty  = y0 + pad + li * (line_h + line_spacing)
            tx0 = x0 + pad + (int(block_w) - int(lw)) // 2

            if active_pos is None:
                text = " ".join(line)
                if fs.shadow_enabled:
                    sc_ = fs.shadow_color; so = fs.shadow_offset
                    draw.text((tx0 + so, ty + so), text, font=self.font,
                              fill=(sc_[0], sc_[1], sc_[2], 200))
                if sw > 0:
                    for dx in range(-sw, sw + 1):
                        for dy in range(-sw, sw + 1):
                            if dx * dx + dy * dy <= sw * sw:
                                draw.text((tx0 + dx, ty + dy), text, font=self.font,
                                          fill=(sc[0], sc[1], sc[2], 255))
                draw.text((tx0, ty), text, font=self.font,
                          fill=(text_color[0], text_color[1], text_color[2], 255))
            else:
                active_line, active_word = active_pos
                try:
                    space_w = draw.textlength(" ", font=self.font)
                except Exception:
                    space_w = 15
                cursor = tx0
                for wi, word in enumerate(line):
                    color = karaoke_color if (li == active_line and wi == active_word) else text_color
                    if fs.shadow_enabled:
                        sc_ = fs.shadow_color; so = fs.shadow_offset
                        draw.text((cursor + so, ty + so), word, font=self.font,
                                  fill=(sc_[0], sc_[1], sc_[2], 200))
                    if sw > 0:
                        for dx in range(-sw, sw + 1):
                            for dy in range(-sw, sw + 1):
                                if dx * dx + dy * dy <= sw * sw:
                                    draw.text((cursor + dx, ty + dy), word, font=self.font,
                                              fill=(sc[0], sc[1], sc[2], 255))
                    draw.text((cursor, ty), word, font=self.font,
                              fill=(color[0], color[1], color[2], 255))
                    try:
                        ww = draw.textlength(word, font=self.font)
                    except Exception:
                        ww = len(word) * 30
                    cursor += ww + space_w

    def _render_phrase(self, phrase: str, text_color: tuple) -> np.ndarray:
        from PIL import Image, ImageDraw
        overlay = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        img  = Image.fromarray(overlay, mode='RGBA')
        draw = ImageDraw.Draw(img)
        max_width = int(self.w * 0.9)
        words = phrase.split() or [phrase]
        lines = self._wrap_words(draw, words, max_width)
        self._draw_lines(draw, lines, text_color)
        return np.array(img)

    def get_overlay(self, phrase: str, text_color: tuple = None) -> np.ndarray:
        tc  = text_color or tuple(self.fs.text_color)
        key = f"{phrase}_{tc}"
        if key not in self._cache:
            self._cache[key] = self._render_phrase(phrase, tc)
        return self._cache[key]

    def _render_words(self, words: list, active_idx: int,
                      text_color: tuple, karaoke_color: tuple) -> np.ndarray:
        """Как _render_phrase, но подсвечивает слово с индексом active_idx отдельным цветом."""
        from PIL import Image, ImageDraw
        overlay = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        img  = Image.fromarray(overlay, mode='RGBA')
        draw = ImageDraw.Draw(img)
        max_width = int(self.w * 0.9)
        lines = self._wrap_words(draw, words, max_width)

        active_pos = (0, active_idx)
        count = 0
        for li, line in enumerate(lines):
            if count + len(line) > active_idx:
                active_pos = (li, active_idx - count)
                break
            count += len(line)

        self._draw_lines(draw, lines, text_color, active_pos=active_pos, karaoke_color=karaoke_color)
        return np.array(img)

    def get_karaoke_overlay(self, words: list, active_idx: int,
                            text_color: tuple = None, karaoke_color: tuple = None) -> np.ndarray:
        tc  = text_color or tuple(self.fs.text_color)
        kc  = karaoke_color or tuple(self.fs.karaoke_color)
        key = ("kw", tuple(words), active_idx, tc, kc)
        if key not in self._cache:
            self._cache[key] = self._render_words(words, active_idx, tc, kc)
        return self._cache[key]

    @staticmethod
    def blend(frame_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        alpha   = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
        fg      = overlay_rgba[:, :, :3].astype(np.float32)
        bg      = frame_rgb.astype(np.float32)
        blended = bg * (1.0 - alpha) + fg * alpha
        return blended.clip(0, 255).astype(np.uint8)
