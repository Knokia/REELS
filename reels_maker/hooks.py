import os

import numpy as np

from .subtitles import SubtitleRenderer


class HookOverlay:
    ALPHA_LEVELS = 32

    def __init__(self, hook_text: str, duration: float = 3.0, persistent: bool = False):
        self.hook_text  = hook_text
        self.duration   = duration
        # persistent=True — заголовок держится весь ролик (только fade-in в начале,
        # без fade-out), в отличие от обычного хука, который гаснет через duration сек.
        self.persistent = persistent
        self._cache: dict = {}

    def _alpha_at(self, t: float) -> float:
        if t < 0.5:
            return t / 0.5
        elif not self.persistent and t > self.duration - 0.5:
            return (self.duration - t) / 0.5
        return 1.0

    def _render(self, alpha: float, w: int, h: int) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont
        font_paths = [
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, 72)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        overlay    = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw       = ImageDraw.Draw(overlay)
        words_list = self.hook_text.split()
        lines, line = [], []
        for word in words_list:
            test = " ".join(line + [word])
            try:
                tw = draw.textbbox((0, 0), test, font=font)[2]
            except Exception:
                tw = len(test) * 30
            if tw > w * 0.85 and line:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))

        y_start = int(h * 0.12)
        pad     = 18
        bg_a    = int(180 * alpha)
        text_a  = int(255 * alpha)

        for li, line_text in enumerate(lines):
            try:
                bb = draw.textbbox((0, 0), line_text, font=font)
                lw, lh = bb[2] - bb[0], bb[3] - bb[1]
            except Exception:
                lw, lh = 400, 72
            lx = (w - lw) // 2
            ly = y_start + li * 90
            draw.rounded_rectangle(
                [(lx - pad, ly - pad // 2), (lx + lw + pad, ly + lh + pad // 2)],
                radius=10, fill=(0, 0, 0, bg_a)
            )
            stroke = 3
            for dx in range(-stroke, stroke + 1):
                for dy in range(-stroke, stroke + 1):
                    if dx * dx + dy * dy <= stroke * stroke:
                        draw.text((lx + dx, ly + dy), line_text, font=font,
                                  fill=(0, 0, 0, text_a))
            draw.text((lx, ly), line_text, font=font, fill=(255, 230, 50, text_a))

        return np.array(overlay)

    def apply(self, clip):
        w, h   = clip.size
        levels = self.ALPHA_LEVELS

        def add_hook(get_frame, t):
            if not self.persistent and t > self.duration:
                return get_frame(t)
            alpha = max(0.0, min(1.0, self._alpha_at(t)))
            level = int(alpha * (levels - 1))
            if level not in self._cache:
                self._cache[level] = self._render(level / (levels - 1), w, h)
            return SubtitleRenderer.blend(get_frame(t), self._cache[level])

        return clip.transform(add_hook)
