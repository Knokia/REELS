from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QFontComboBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout,
)

from ..config import FONT_SETTINGS


class FontSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔤 Настройки шрифта субтитров")
        self.setMinimumWidth(520)
        self.setStyleSheet("""
            QDialog{background:#1e1e1e;color:#ddd;}
            QLabel{color:#ddd;}
            QGroupBox{color:#aaa;border:1px solid #444;border-radius:6px;
                margin-top:8px;padding:8px;font-weight:bold;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;}
            QSpinBox,QComboBox,QFontComboBox{background:#2a2a2a;color:#ddd;
                border:1px solid #555;border-radius:4px;padding:4px;}
            QCheckBox{color:#ddd;}
            QPushButton{background:#333;color:#ddd;border:1px solid #555;
                border-radius:5px;padding:6px 14px;}
            QPushButton:hover{background:#444;}
        """)
        self._setup()
        self._load_from_settings()

    def _setup(self):
        layout = QVBoxLayout(self)
        header = QLabel("🔤 Настройки шрифта субтитров")
        header.setStyleSheet("font-size:15px;font-weight:bold;color:#4CAF50;padding:6px;")
        layout.addWidget(header)

        fg = QGroupBox("Шрифт"); fl = QFormLayout()
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(FONT_SETTINGS.font_family))
        fl.addRow("Семейство:", self.font_combo)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(20, 200)
        self.size_spin.setValue(FONT_SETTINGS.font_size)
        self.size_spin.setSuffix(" px")
        fl.addRow("Размер:", self.size_spin)
        sr2 = QHBoxLayout()
        self.bold_cb   = QCheckBox("Жирный"); self.bold_cb.setChecked(FONT_SETTINGS.bold)
        self.italic_cb = QCheckBox("Курсив"); self.italic_cb.setChecked(FONT_SETTINGS.italic)
        sr2.addWidget(self.bold_cb); sr2.addWidget(self.italic_cb); sr2.addStretch()
        fl.addRow("Стиль:", sr2)
        self.words_spin = QSpinBox()
        self.words_spin.setRange(1, 10)
        self.words_spin.setValue(FONT_SETTINGS.words_per_phrase)
        self.words_spin.setSuffix(" слов")
        fl.addRow("Слов в фразе:", self.words_spin)
        fg.setLayout(fl); layout.addWidget(fg)

        cg = QGroupBox("Цвета текста"); cl = QFormLayout()
        self.text_color_btn   = self._color_btn(FONT_SETTINGS.text_color,   "text")
        self.stroke_color_btn = self._color_btn(FONT_SETTINGS.stroke_color, "stroke")
        self.stroke_spin = QSpinBox()
        self.stroke_spin.setRange(0, 20)
        self.stroke_spin.setValue(FONT_SETTINGS.stroke_width)
        self.stroke_spin.setSuffix(" px")
        cl.addRow("Цвет текста:",   self.text_color_btn)
        cl.addRow("Цвет обводки:", self.stroke_color_btn)
        cl.addRow("Толщина:",       self.stroke_spin)
        cg.setLayout(cl); layout.addWidget(cg)

        sg2 = QGroupBox("Тень"); sl3 = QFormLayout()
        self.shadow_cb = QCheckBox("Включить")
        self.shadow_cb.setChecked(FONT_SETTINGS.shadow_enabled)
        self.shadow_color_btn = self._color_btn(FONT_SETTINGS.shadow_color, "shadow")
        self.shadow_offset_spin = QSpinBox()
        self.shadow_offset_spin.setRange(1, 20)
        self.shadow_offset_spin.setValue(FONT_SETTINGS.shadow_offset)
        self.shadow_offset_spin.setSuffix(" px")
        sl3.addRow("", self.shadow_cb)
        sl3.addRow("Цвет:", self.shadow_color_btn)
        sl3.addRow("Смещение:", self.shadow_offset_spin)
        sg2.setLayout(sl3); layout.addWidget(sg2)

        bg2 = QGroupBox("Фон"); bl2 = QFormLayout()
        self.bg_cb = QCheckBox("Включить")
        self.bg_cb.setChecked(FONT_SETTINGS.bg_enabled)
        self.bg_color_btn    = self._color_btn(FONT_SETTINGS.bg_color, "bg")
        self.bg_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.bg_alpha_slider.setRange(0, 255)
        self.bg_alpha_slider.setValue(FONT_SETTINGS.bg_alpha)
        self.bg_alpha_label  = QLabel(str(FONT_SETTINGS.bg_alpha))
        self.bg_alpha_slider.valueChanged.connect(
            lambda v: self.bg_alpha_label.setText(str(v))
        )
        ar2 = QHBoxLayout()
        ar2.addWidget(self.bg_alpha_slider)
        ar2.addWidget(self.bg_alpha_label)
        bl2.addRow("", self.bg_cb)
        bl2.addRow("Цвет:", self.bg_color_btn)
        bl2.addRow("Прозрачность:", ar2)
        bg2.setLayout(bl2); layout.addWidget(bg2)

        kg = QGroupBox("Караоке-подсветка слов"); kl = QFormLayout()
        self.karaoke_cb = QCheckBox("Подсвечивать произносимое слово")
        self.karaoke_cb.setChecked(FONT_SETTINGS.karaoke_enabled)
        self.karaoke_color_btn = self._color_btn(FONT_SETTINGS.karaoke_color, "karaoke")
        kl.addRow("", self.karaoke_cb)
        kl.addRow("Цвет подсветки:", self.karaoke_color_btn)
        kg.setLayout(kl); layout.addWidget(kg)

        pg = QGroupBox("Позиция"); pl2 = QFormLayout()
        self.pos_combo = QComboBox()
        self.pos_combo.addItems(["25%", "50%", "75%", "90%"])
        self.pos_combo.setCurrentText(FONT_SETTINGS.position)
        pl2.addRow("Вертикаль:", self.pos_combo)
        pg.setLayout(pl2); layout.addWidget(pg)

        self.preview_label = QLabel("Предпросмотр: Текст субтитров")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(60)
        self.preview_label.setStyleSheet(
            "background:#111;border:1px solid #333;border-radius:6px;padding:8px;"
        )
        layout.addWidget(self.preview_label)
        self.font_combo.currentFontChanged.connect(self._update_preview)
        self.size_spin.valueChanged.connect(self._update_preview)
        self.bold_cb.stateChanged.connect(self._update_preview)
        self.italic_cb.stateChanged.connect(self._update_preview)
        self._update_preview()

        br2 = QHBoxLayout(); br2.addStretch()
        rb = QPushButton("↩ Сбросить"); rb.clicked.connect(self._reset)
        cb2 = QPushButton("Отмена");   cb2.clicked.connect(self.reject)
        ob  = QPushButton("✅ Применить")
        ob.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;font-weight:bold;"
            "padding:8px 20px;border-radius:6px;}"
            "QPushButton:hover{background:#45a049;}"
        )
        ob.clicked.connect(self._apply)
        br2.addWidget(rb); br2.addWidget(cb2); br2.addWidget(ob)
        layout.addLayout(br2)

    def _color_btn(self, rgb, tag):
        btn = QPushButton(); btn.setFixedWidth(80)
        btn.setProperty("color_tag", tag)
        btn.setProperty("rgb", list(rgb))
        self._refresh_btn_color(btn, rgb)
        btn.clicked.connect(lambda _, b=btn: self._pick_color(b))
        return btn

    def _refresh_btn_color(self, btn, rgb):
        r, g, b = rgb
        fg = "#000" if 0.299*r + 0.587*g + 0.114*b > 128 else "#fff"
        btn.setStyleSheet(
            f"QPushButton{{background:rgb({r},{g},{b});color:{fg};"
            f"border:2px solid #888;border-radius:4px;}}"
        )
        btn.setText(f"#{r:02X}{g:02X}{b:02X}")

    def _pick_color(self, btn):
        color = QColorDialog.getColor(QColor(*btn.property("rgb")), self, "Цвет")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            btn.setProperty("rgb", list(rgb))
            self._refresh_btn_color(btn, rgb)

    def _update_preview(self):
        f = self.font_combo.currentFont().family()
        s = max(8, self.size_spin.value() // 5)
        w = "bold"   if self.bold_cb.isChecked()   else "normal"
        i = "italic" if self.italic_cb.isChecked() else "normal"
        self.preview_label.setStyleSheet(
            f"background:#111;border:1px solid #333;border-radius:6px;padding:8px;"
            f"font-family:'{f}';font-size:{s}pt;font-weight:{w};font-style:{i};color:white;"
        )

    def _load_from_settings(self):
        self.font_combo.setCurrentFont(QFont(FONT_SETTINGS.font_family))
        self.size_spin.setValue(FONT_SETTINGS.font_size)
        self.bold_cb.setChecked(FONT_SETTINGS.bold)
        self.italic_cb.setChecked(FONT_SETTINGS.italic)
        self.stroke_spin.setValue(FONT_SETTINGS.stroke_width)
        self.shadow_cb.setChecked(FONT_SETTINGS.shadow_enabled)
        self.shadow_offset_spin.setValue(FONT_SETTINGS.shadow_offset)
        self.bg_cb.setChecked(FONT_SETTINGS.bg_enabled)
        self.bg_alpha_slider.setValue(FONT_SETTINGS.bg_alpha)
        self.pos_combo.setCurrentText(FONT_SETTINGS.position)
        self.words_spin.setValue(FONT_SETTINGS.words_per_phrase)
        self.karaoke_cb.setChecked(FONT_SETTINGS.karaoke_enabled)

    def _reset(self):
        d = FONT_SETTINGS.__class__()
        for attr in vars(d):
            setattr(FONT_SETTINGS, attr, getattr(d, attr))
        self._load_from_settings()
        for btn, key in [
            (self.text_color_btn,   'text_color'),
            (self.stroke_color_btn, 'stroke_color'),
            (self.shadow_color_btn, 'shadow_color'),
            (self.bg_color_btn,     'bg_color'),
            (self.karaoke_color_btn, 'karaoke_color'),
        ]:
            self._refresh_btn_color(btn, getattr(FONT_SETTINGS, key))

    def _apply(self):
        FONT_SETTINGS.font_family      = self.font_combo.currentFont().family()
        FONT_SETTINGS.font_size        = self.size_spin.value()
        FONT_SETTINGS.bold             = self.bold_cb.isChecked()
        FONT_SETTINGS.italic           = self.italic_cb.isChecked()
        FONT_SETTINGS.text_color       = tuple(self.text_color_btn.property("rgb"))
        FONT_SETTINGS.stroke_color     = tuple(self.stroke_color_btn.property("rgb"))
        FONT_SETTINGS.stroke_width     = self.stroke_spin.value()
        FONT_SETTINGS.shadow_enabled   = self.shadow_cb.isChecked()
        FONT_SETTINGS.shadow_color     = tuple(self.shadow_color_btn.property("rgb"))
        FONT_SETTINGS.shadow_offset    = self.shadow_offset_spin.value()
        FONT_SETTINGS.bg_enabled       = self.bg_cb.isChecked()
        FONT_SETTINGS.bg_color         = tuple(self.bg_color_btn.property("rgb"))
        FONT_SETTINGS.bg_alpha         = self.bg_alpha_slider.value()
        FONT_SETTINGS.position         = self.pos_combo.currentText()
        FONT_SETTINGS.words_per_phrase = self.words_spin.value()
        FONT_SETTINGS.karaoke_enabled  = self.karaoke_cb.isChecked()
        FONT_SETTINGS.karaoke_color    = tuple(self.karaoke_color_btn.property("rgb"))
        self.accept()
