import os
import re

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QSlider, QSpinBox, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget, QInputDialog,
)

from .. import youtube
from ..compilation import CompilationMergeThread
from ..config import CLIPS_DIR, FONT_SETTINGS, SETTINGS_PATH
from ..pipeline import ProcessingThread
from ..session_log import SessionLog
from .font_dialog import FontSettingsDialog
from .multi_source_dialog import MultiSourceDialog
from .upload_dialog import UploadScheduleDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Reels Maker PRO")
        self.setMinimumSize(900, 760)
        self._clips_result = []
        self._accounts      = []
        self._credentials   = None
        self._batch_queue  = []
        self._batch_total  = 0
        self._batch_index  = 0
        self.session_log = SessionLog()
        self.setup_ui()
        self._load_settings()

    def _log(self, text: str):
        """Единая точка логирования: в окно и в файл logs/run_*.txt."""
        self.log_view.append(text)
        self.session_log.write(text)

    def closeEvent(self, event):
        self._save_settings()
        self.session_log.close()
        super().closeEvent(event)

    # ── Сохранение настроек между запусками ────────────────────

    def _settings(self):
        return QSettings(SETTINGS_PATH, QSettings.Format.IniFormat)

    def _save_settings(self):
        s = self._settings()
        s.setValue("quality",       self.quality_combo.currentText())
        s.setValue("language",      self.lang_combo.currentText())
        s.setValue("duration",      self.duration_spin.value())
        s.setValue("clip_count",    self.clip_count_spin.value())
        s.setValue("zoom_enabled",  self.zoom_enabled.isChecked())
        s.setValue("zoom_value",    self.zoom_slider.value())
        s.setValue("face_crop",     self.face_crop_cb.isChecked())
        s.setValue("hook",          self.hook_cb.isChecked())
        s.setValue("virality",      self.virality_cb.isChecked())
        s.setValue("multi_speaker", self.multi_speaker_cb.isChecked())
        if self.format_centered_radio.isChecked():
            fmt = "centered"
        elif self.format_split_radio.isChecked():
            fmt = "split"
        else:
            fmt = "normal"
        s.setValue("frame_format", fmt)
        s.setValue("output_type",
                   "compilation" if self.output_compilation_radio.isChecked() else "shorts")
        s.setValue("tab_index",    self.tabs.currentIndex())
        s.setValue("geometry",     self.saveGeometry())
        if 0 <= self.channel_combo.currentIndex() < len(self._accounts):
            s.setValue("channel_id", self._accounts[self.channel_combo.currentIndex()]["channel_id"])

    def _load_settings(self):
        s = self._settings()
        if s.value("quality"):
            self.quality_combo.setCurrentText(s.value("quality"))
        if s.value("language"):
            self.lang_combo.setCurrentText(s.value("language"))
        self.duration_spin.setValue(s.value("duration", 30, int))
        self.clip_count_spin.setValue(s.value("clip_count", 7, int))
        self.zoom_enabled.setChecked(s.value("zoom_enabled", True, bool))
        self.zoom_slider.setValue(s.value("zoom_value", 40, int))
        self.face_crop_cb.setChecked(s.value("face_crop", True, bool))
        self.hook_cb.setChecked(s.value("hook", True, bool))
        self.virality_cb.setChecked(s.value("virality", True, bool))
        self.multi_speaker_cb.setChecked(s.value("multi_speaker", False, bool))
        fmt = s.value("frame_format", "normal")
        {"centered": self.format_centered_radio,
         "split":    self.format_split_radio}.get(fmt, self.format_normal_radio).setChecked(True)
        if s.value("output_type", "shorts") == "compilation":
            self.output_compilation_radio.setChecked(True)
        self.tabs.setCurrentIndex(s.value("tab_index", 0, int))
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        channel_id = s.value("channel_id")
        if channel_id:
            self._refresh_channel_combo(select_channel_id=channel_id)

    def setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout  = QVBoxLayout(central)

        title = QLabel("🎬 AI Reels Maker PRO")
        title.setStyleSheet("font-size:20px;font-weight:bold;color:#4CAF50;padding:12px;")
        layout.addWidget(title)

        ar = QHBoxLayout()
        ar.addWidget(QLabel("Канал:"))
        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumWidth(200)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_selected)
        ar.addWidget(self.channel_combo)
        self.remove_channel_btn = QPushButton("🗑")
        self.remove_channel_btn.setFixedWidth(32)
        self.remove_channel_btn.setToolTip("Убрать выбранный канал из приложения")
        self.remove_channel_btn.clicked.connect(self.remove_selected_channel)
        ar.addWidget(self.remove_channel_btn)
        ar.addStretch()
        self.auth_btn = QPushButton("➕ Добавить канал")
        self.auth_btn.setStyleSheet(
            "QPushButton{background:#4285F4;color:white;padding:6px 16px;"
            "border-radius:6px;font-weight:bold;}"
        )
        self.auth_btn.clicked.connect(self.do_auth)
        ar.addWidget(self.auth_btn); layout.addLayout(ar)
        self._refresh_channel_combo()

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab{padding:8px 16px;font-weight:bold;}
            QTabWidget::pane{border:1px solid #444;border-radius:4px;}
        """)
        self.tabs.addTab(self._build_source_tab(), "🎬 Источник")
        self.tabs.addTab(self._build_ai_format_tab(), "🤖 AI и формат")
        self.tabs.addTab(self._build_subtitles_tab(), "🔤 Субтитры")
        layout.addWidget(self.tabs)

        self.start_btn = QPushButton("🎬 Создать клипы с AI обработкой")
        self.start_btn.setStyleSheet("""
            QPushButton{background:#4CAF50;color:white;padding:12px;font-size:14px;
                border-radius:5px;font-weight:bold;}
            QPushButton:hover{background:#45a049;}
        """)
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)

        self.upload_btn = QPushButton("📤 Загрузить клипы на YouTube с расписанием")
        self.upload_btn.setStyleSheet("""
            QPushButton{background:#f44336;color:white;padding:12px;font-size:14px;
                border-radius:5px;font-weight:bold;}
            QPushButton:hover{background:#d32f2f;}
        """)
        self.upload_btn.clicked.connect(self.open_upload_dialog)
        self.upload_btn.setVisible(False); layout.addWidget(self.upload_btn)

        self.upload_from_folder_btn = QPushButton("📂 Загрузить готовые клипы из папки...")
        self.upload_from_folder_btn.setToolTip(
            "Выбрать любую папку с готовыми .mp4 (по умолчанию открывается clips) — "
            "ищет и во вложенных подпапках"
        )
        self.upload_from_folder_btn.setStyleSheet("""
            QPushButton{background:#7B1FA2;color:white;padding:10px;font-size:13px;
                border-radius:5px;font-weight:bold;}
            QPushButton:hover{background:#6A1B9A;}
        """)
        self.upload_from_folder_btn.clicked.connect(self.open_clips_folder_dialog)
        layout.addWidget(self.upload_from_folder_btn)

        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar{border:2px solid #ccc;border-radius:5px;
                text-align:center;height:25px;}
            QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #4CAF50,stop:1 #45a049);}
        """)
        layout.addWidget(self.progress_bar)

        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(160)
        self.log_view.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#ddd;font-family:Consolas;"
            "font-size:11px;padding:10px;border-radius:5px;}"
        )
        layout.addWidget(self.log_view)

    def _build_source_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("🔗 YouTube URL или локальный файл:"))
        ur = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText(
            "https://www.youtube.com/watch?v=... или путь к файлу"
        )
        ur.addWidget(self.url_input)
        br4 = QPushButton("📁 Файл")
        br4.setStyleSheet(
            "QPushButton{background:#2196F3;color:white;padding:8px 14px;border-radius:5px;}"
        )
        br4.clicked.connect(self.browse_file)
        ur.addWidget(br4)
        br5 = QPushButton("📚 Несколько файлов")
        br5.setToolTip("Выбрать несколько локальных видео — каждое обработается отдельно, по очереди")
        br5.setStyleSheet(
            "QPushButton{background:#00838F;color:white;padding:8px 14px;border-radius:5px;}"
        )
        br5.clicked.connect(self.browse_files_batch)
        ur.addWidget(br5)
        layout.addLayout(ur)

        self.multi_comp_btn = QPushButton("🎬 Компиляция из нескольких видео...")
        self.multi_comp_btn.setToolTip(
            "Только для типа результата «Компиляция»: выбрать несколько источников "
            "(YouTube-ссылки и/или файлы) — моменты из всех склеятся в один ролик 16:9."
        )
        self.multi_comp_btn.setStyleSheet(
            "QPushButton{background:#5E35B1;color:white;padding:8px 14px;border-radius:5px;font-weight:bold;}"
            "QPushButton:hover{background:#4527A0;}"
        )
        self.multi_comp_btn.clicked.connect(self.browse_multi_compilation_sources)
        self.multi_comp_btn.setVisible(False)  # видно только в режиме «Компиляция»
        layout.addWidget(self.multi_comp_btn)

        sr3 = QHBoxLayout()
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["1080p", "720p", "480p", "360p"])
        qg = QGroupBox("Качество"); ql2 = QHBoxLayout()
        ql2.addWidget(self.quality_combo); qg.setLayout(ql2)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Авто", "RU", "EN"])
        lg2 = QGroupBox("Язык"); ll2 = QHBoxLayout()
        ll2.addWidget(self.lang_combo); lg2.setLayout(ll2)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(15, 90); self.duration_spin.setValue(30)
        dg2 = QGroupBox("Длина (сек)"); dl2 = QHBoxLayout()
        dl2.addWidget(self.duration_spin); dg2.setLayout(dl2)
        self.clip_count_spin = QSpinBox()
        self.clip_count_spin.setRange(3, 20); self.clip_count_spin.setValue(7)
        self.clip_count_spin.setToolTip(
            "LLM рассматривает больше кандидатов, чем нужно клипов, и по скорингу\n"
            "виральности отсеивает слабые — итоговых клипов может быть меньше,\n"
            "если сильных моментов не набралось."
        )
        cg2 = QGroupBox("Кол-во клипов"); cl2 = QHBoxLayout()
        cl2.addWidget(self.clip_count_spin); cg2.setLayout(cl2)
        sr3.addWidget(qg); sr3.addWidget(lg2); sr3.addWidget(dg2); sr3.addWidget(cg2)
        layout.addLayout(sr3)
        layout.addStretch()
        return tab

    def _build_ai_format_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)

        og = QGroupBox("🎞️ Тип результата"); ol = QVBoxLayout()
        self.output_shorts_radio = QRadioButton("📱 Shorts — вертикальные клипы 9:16 (как обычно)")
        self.output_shorts_radio.setChecked(True)
        self.output_compilation_radio = QRadioButton(
            "🎬 Компиляция — одно длинное горизонтальное видео 16:9"
        )
        self.output_compilation_radio.setToolTip(
            "Найденные моменты склеиваются в хронологическом порядке в одно\n"
            "видео 1920x1080 с плашкой-названием в начале каждого сегмента.\n"
            "Кроп/зум/хук/субтитры Shorts не применяются. Для 8+ минут\n"
            "на YouTube доступны mid-roll рекламные вставки."
        )
        for rb in (self.output_shorts_radio, self.output_compilation_radio):
            rb.setStyleSheet("font-weight:bold;padding:3px;")
            rb.toggled.connect(self._on_output_type_changed)
            ol.addWidget(rb)
        self.output_type_group = QButtonGroup(self)
        self.output_type_group.addButton(self.output_shorts_radio)
        self.output_type_group.addButton(self.output_compilation_radio)
        og.setLayout(ol); layout.addWidget(og)

        fg = QGroupBox("🖼️ Формат кадра"); fl = QVBoxLayout()
        self.format_normal_radio = QRadioButton("🎯 Обычный — умный кроп по лицу + зум")
        self.format_normal_radio.setChecked(True)
        self.format_centered_radio = QRadioButton("🖼️ По центру")
        self.format_centered_radio.setToolTip(
            "Видео вписывается в кадр целиком, без обрезки (contain-fit), и\n"
            "центрируется на чёрном фоне. Кроп/зум/фокус на лицах не применяются.\n"
            "Заголовок клипа держится сверху весь ролик."
        )
        self.format_split_radio = QRadioButton("📱 Split-screen")
        self.format_split_radio.setToolTip(
            "Видео кропается в верхнюю половину экрана, в нижнюю — случайный\n"
            "отрезок фоновой нарезки (Subway Surfers/песок/т.п.) из папки\n"
            "background_footage/. Положите туда свои файлы — .mp4/.mov/.mkv/.webm."
        )
        for rb in (self.format_normal_radio, self.format_centered_radio, self.format_split_radio):
            rb.setStyleSheet("font-weight:bold;padding:3px;")
            rb.toggled.connect(self._on_format_changed)
            fl.addWidget(rb)
        self.format_group = QButtonGroup(self)
        for rb in (self.format_normal_radio, self.format_centered_radio, self.format_split_radio):
            self.format_group.addButton(rb)
        fg.setLayout(fl); layout.addWidget(fg)
        self.format_group_box = fg

        zg2 = QGroupBox("⚙️ Динамический зум (только формат «Обычный»)"); zl2 = QVBoxLayout()
        self.zoom_enabled = QCheckBox("🎥 Включить зум")
        self.zoom_enabled.setChecked(True)
        self.zoom_enabled.setStyleSheet("font-weight:bold;")
        zl2.addWidget(self.zoom_enabled)
        ir2 = QHBoxLayout(); ir2.addWidget(QLabel("Интенсивность:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, 100); self.zoom_slider.setValue(40)
        ir2.addWidget(self.zoom_slider)
        self.zoom_val = QLabel("40%")
        self.zoom_val.setStyleSheet("font-weight:bold;min-width:40px;")
        ir2.addWidget(self.zoom_val); zl2.addLayout(ir2)
        zg2.setLayout(zl2); layout.addWidget(zg2)
        self.zoom_slider.valueChanged.connect(lambda v: self.zoom_val.setText(f"{v}%"))
        self.zoom_group_box = zg2

        ag2 = QGroupBox("🤖 AI Улучшения"); al2 = QVBoxLayout()
        self.face_crop_cb = QCheckBox("👤 Умный кроп (MediaPipe) + плавное слежение")
        self.face_crop_cb.setChecked(True)
        self.face_crop_cb.setStyleSheet("font-weight:bold;")
        self.hook_cb = QCheckBox("🪝 Автогенерация хука (первые 3 сек)")
        self.hook_cb.setChecked(True)
        self.hook_cb.setStyleSheet("font-weight:bold;")
        self.virality_cb = QCheckBox("📊 Скоринг виральности")
        self.virality_cb.setChecked(True)
        self.virality_cb.setStyleSheet("font-weight:bold;")
        self.multi_speaker_cb = QCheckBox("🗣️ Мультиспикерный кроп (переключение на говорящего)")
        self.multi_speaker_cb.setChecked(False)
        self.multi_speaker_cb.setStyleSheet("font-weight:bold;")
        emo_info = QLabel(
            "😊 Эмоции: MediaPipe Face Mesh (геометрия лица) + OpenCV Haar fallback\n"
            "   Зависимости: mediapipe, opencv-python — уже установлены"
        )
        emo_info.setStyleSheet("color:#888;font-size:10px;padding:2px;")
        emo_info.setWordWrap(True)
        info2 = QLabel(
            "ℹ️  Плавность: EMA 0.97 • max_step 8px • zoom 4s • субтитры кэш • hook 32 уровня\n"
            "   Транскрипт кэшируется • рендер клипов идёт параллельно в несколько потоков"
        )
        info2.setStyleSheet("color:#888;font-size:10px;padding:2px;")
        info2.setWordWrap(True)
        al2.addWidget(self.face_crop_cb); al2.addWidget(self.hook_cb)
        al2.addWidget(self.virality_cb); al2.addWidget(self.multi_speaker_cb)
        al2.addWidget(emo_info); al2.addWidget(info2)
        ag2.setLayout(al2); layout.addWidget(ag2)
        layout.addStretch()
        return tab

    def _build_subtitles_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        sg3 = QGroupBox("🔤 Субтитры"); sl4 = QHBoxLayout()
        self.font_preview = QLabel(); self._refresh_font_preview()
        self.font_preview.setMinimumWidth(220)
        self.font_preview.setStyleSheet(
            "background:#111;border:1px solid #333;border-radius:5px;"
            "padding:6px 12px;color:white;"
        )
        sl4.addWidget(self.font_preview); sl4.addStretch()
        fb2 = QPushButton("🔤 Настроить шрифт...")
        fb2.setStyleSheet(
            "QPushButton{background:#7B1FA2;color:white;padding:9px 18px;"
            "border-radius:6px;font-weight:bold;font-size:12px;}"
            "QPushButton:hover{background:#6A1B9A;}"
        )
        fb2.clicked.connect(self.open_font_dialog)
        sl4.addWidget(fb2); sg3.setLayout(sl4); layout.addWidget(sg3)
        layout.addStretch()
        return tab

    def _on_format_changed(self):
        """Кроп по лицу, мультиспикер и зум имеют смысл только в обычном формате —
        в «по центру»/split-screen кадр строится иначе, эти настройки не участвуют."""
        if self.output_compilation_radio.isChecked():
            return  # в режиме компиляции всё уже заблокировано _on_output_type_changed
        normal = self.format_normal_radio.isChecked()
        for w in (self.face_crop_cb, self.multi_speaker_cb, self.zoom_group_box):
            w.setEnabled(normal)
        # Хук (мигающий 3 сек) заменяется постоянным заголовком только в «по центру».
        self.hook_cb.setEnabled(not self.format_centered_radio.isChecked())

    def _on_output_type_changed(self):
        """В режиме «Компиляция» кадр не перекраивается в вертикаль — формат кадра,
        кроп, зум и хук не участвуют вовсе."""
        shorts = self.output_shorts_radio.isChecked()
        self.format_group_box.setEnabled(shorts)
        self.multi_comp_btn.setVisible(not shorts)
        if shorts:
            self._on_format_changed()
        else:
            for w in (self.face_crop_cb, self.multi_speaker_cb,
                      self.zoom_group_box, self.hook_cb):
                w.setEnabled(False)

    def _refresh_font_preview(self):
        fs3 = FONT_SETTINGS; r, g, b = fs3.text_color
        parts = [fs3.font_family, f"{fs3.font_size}px"]
        if fs3.bold:   parts.append("Bold")
        if fs3.italic: parts.append("Italic")
        parts += [f"| {fs3.position}", f"| {fs3.words_per_phrase} сл."]
        self.font_preview.setText("  ".join(parts))
        self.font_preview.setStyleSheet(
            f"background:#111;border:1px solid #333;border-radius:5px;padding:6px 12px;"
            f"color:rgb({r},{g},{b});font-family:'{fs3.font_family}';"
            f"font-weight:{'bold' if fs3.bold else 'normal'};"
            f"font-style:{'italic' if fs3.italic else 'normal'};"
        )

    def open_font_dialog(self):
        dlg = FontSettingsDialog(self)
        if dlg.exec():
            self._refresh_font_preview()
            self._log(
                f"✅ Шрифт: {FONT_SETTINGS.font_family} {FONT_SETTINGS.font_size}px"
                + (" Bold" if FONT_SETTINGS.bold else "")
                + (" Italic" if FONT_SETTINGS.italic else "")
            )

    def browse_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "",
            "Видео (*.mp4 *.avi *.mov *.mkv *.webm *.flv);;Все (*.*)"
        )
        if fp:
            self.url_input.setText(fp)
            self._log(f"✅ {os.path.basename(fp)}")

    def browse_files_batch(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите видео (можно несколько)", "",
            "Видео (*.mp4 *.avi *.mov *.mkv *.webm *.flv);;Все (*.*)"
        )
        if files:
            self.start_batch_processing(files)

    def do_auth(self):
        """Авторизует ЕЩЁ ОДИН Google-аккаунт/канал — не заменяет уже добавленные,
        а добавляет рядом. В браузере можно войти под другим аккаунтом."""
        try:
            client_id = client_secret = None
            if youtube.needs_setup():
                cid,  ok1 = QInputDialog.getText(self, "Client ID",     "OAuth Client ID:")
                csec, ok2 = QInputDialog.getText(self, "Client Secret", "Client Secret:")
                if not (ok1 and ok2 and cid and csec):
                    return
                client_id, client_secret = cid, csec
            result = youtube.add_account(client_id, client_secret)
            self._refresh_channel_combo(select_channel_id=result["channel_id"])
            self._log(f"✅ Канал добавлен: {result['title']}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка авторизации", str(e))

    def _refresh_channel_combo(self, select_channel_id=None):
        self._accounts = youtube.list_accounts()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        for acc in self._accounts:
            self.channel_combo.addItem(acc["title"], acc["channel_id"])
        self.channel_combo.blockSignals(False)
        if self._accounts:
            idx = 0
            if select_channel_id:
                for i, acc in enumerate(self._accounts):
                    if acc["channel_id"] == select_channel_id:
                        idx = i; break
            self.channel_combo.setCurrentIndex(idx)
            self._credentials = self._accounts[idx]["credentials"]
        else:
            self._credentials = None

    def _on_channel_selected(self, index):
        if 0 <= index < len(self._accounts):
            self._credentials = self._accounts[index]["credentials"]

    def remove_selected_channel(self):
        idx = self.channel_combo.currentIndex()
        if idx < 0 or idx >= len(self._accounts):
            return
        acc = self._accounts[idx]
        if QMessageBox.question(
            self, "Удалить канал", f"Убрать «{acc['title']}» из приложения?"
        ) == QMessageBox.StandardButton.Yes:
            youtube.remove_account(acc["channel_id"])
            self._refresh_channel_combo()

    def _build_thread(self, url, index_offset=0):
        return ProcessingThread(
            url,
            self.quality_combo.currentText(),
            self.lang_combo.currentText(),
            self.duration_spin.value(),
            self.zoom_enabled.isChecked(),
            self.zoom_slider.value(),
            face_crop_enabled=self.face_crop_cb.isChecked(),
            hook_enabled=self.hook_cb.isChecked(),
            virality_enabled=self.virality_cb.isChecked(),
            multi_speaker_crop=self.multi_speaker_cb.isChecked(),
            clip_count=self.clip_count_spin.value(),
            index_offset=index_offset,
            centered_layout_enabled=self.format_centered_radio.isChecked(),
            split_screen_enabled=self.format_split_radio.isChecked(),
            compilation_enabled=self.output_compilation_radio.isChecked(),
        )

    def start_processing(self):
        url = self.url_input.text().strip()
        if not url:
            self._log("❌ Вставьте URL или выберите файл"); return
        if not url.startswith('http') and not os.path.isfile(url):
            self._log(f"❌ Файл не найден: {url}"); return
        self.log_view.clear()
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False); self.upload_btn.setVisible(False)
        self.thread = self._build_thread(url)
        self.thread.log.connect(self._log)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, clips):
        self.start_btn.setEnabled(True); self._clips_result = clips
        if clips:
            self._log(f"\n🎉 Клипов: {len(clips)}")
            self.upload_btn.setVisible(True)
            try: os.startfile(CLIPS_DIR)
            except Exception: pass

    # ── Пакетная обработка нескольких локальных файлов ─────────

    def start_batch_processing(self, files):
        self._batch_queue  = list(files)
        self._batch_total  = len(files)
        self._batch_index  = 0
        self._clips_result = []
        self.log_view.clear()
        self._log(f"📦 Пакетная обработка: {self._batch_total} файл(ов)")
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False); self.upload_btn.setVisible(False)
        self._run_next_in_batch()

    def _run_next_in_batch(self):
        if self._batch_index >= self._batch_total:
            self.start_btn.setEnabled(True)
            if self._clips_result:
                self._log(f"\n🎉🎉 Пакет завершён. Всего клипов: {len(self._clips_result)}")
                self.upload_btn.setVisible(True)
                try: os.startfile(CLIPS_DIR)
                except Exception: pass
            else:
                self._log("\n⚠️ Пакет завершён, но клипов не получилось.")
            return

        fp = self._batch_queue[self._batch_index]
        self._batch_index += 1
        self._log(
            f"\n\n=== 📁 Файл {self._batch_index}/{self._batch_total}: {os.path.basename(fp)} ==="
        )
        self.thread = self._build_thread(fp)
        self.thread.log.connect(self._log)
        self.thread.progress.connect(self._on_batch_progress)
        self.thread.finished.connect(self._on_batch_file_finished)
        self.thread.start()

    def _on_batch_progress(self, pct):
        base = (self._batch_index - 1) / self._batch_total * 100
        span = 100 / self._batch_total
        self.progress_bar.setValue(int(base + pct / 100 * span))

    def _on_batch_file_finished(self, clips):
        self._clips_result.extend(clips)
        self._run_next_in_batch()

    # ── Компиляция из нескольких источников в одно видео 16:9 ──

    def browse_multi_compilation_sources(self):
        dlg = MultiSourceDialog(self)
        if dlg.exec():
            sources = dlg.sources()
            if sources:
                self.start_multi_compilation(sources)

    def start_multi_compilation(self, sources: list):
        self._multi_comp_queue = list(sources)
        self._multi_comp_total = len(sources)
        self._multi_comp_index = 0
        self._multi_comp_jobs  = []
        self.log_view.clear()
        self._log(f"🎬 Компиляция из {self._multi_comp_total} источник(ов)")
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False); self.upload_btn.setVisible(False)
        self._run_next_multi_comp_source()

    def _run_next_multi_comp_source(self):
        if self._multi_comp_index >= self._multi_comp_total:
            self._finish_multi_comp_collection()
            return
        src = self._multi_comp_queue[self._multi_comp_index]
        self._multi_comp_index += 1
        label = os.path.basename(src) if os.path.isfile(src) else src
        self._log(
            f"\n\n=== 📁 Источник {self._multi_comp_index}/{self._multi_comp_total}: {label} ==="
        )
        # work_subdir даёт каждому источнику свою рабочую папку — иначе видео
        # следующего источника перезаписало бы ещё не склеенное видео предыдущего.
        self.thread = ProcessingThread(
            src,
            self.quality_combo.currentText(),
            self.lang_combo.currentText(),
            self.duration_spin.value(),
            self.zoom_enabled.isChecked(),
            self.zoom_slider.value(),
            clip_count=self.clip_count_spin.value(),
            compilation_enabled=True,
            work_subdir=f"multi_comp_{self._multi_comp_index}",
            emit_jobs_only=True,
        )
        self.thread.log.connect(self._log)
        self.thread.progress.connect(self._on_multi_comp_source_progress)
        self.thread.jobs_ready.connect(self._on_multi_comp_source_ready)
        self.thread.start()

    def _on_multi_comp_source_progress(self, pct):
        base = (self._multi_comp_index - 1) / self._multi_comp_total * 100
        span = 100 / self._multi_comp_total
        # Последние ~10% прогресс-бара оставляем под финальную склейку.
        self.progress_bar.setValue(int((base + pct / 100 * span) * 0.9))

    def _on_multi_comp_source_ready(self, data):
        if data and data.get('render_jobs'):
            self._multi_comp_jobs.append(data)
        self._run_next_multi_comp_source()

    def _finish_multi_comp_collection(self):
        if not self._multi_comp_jobs:
            self._log("\n⚠️ Не удалось собрать моменты ни из одного источника.")
            self.start_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            return
        total_moments = sum(len(d['render_jobs']) for d in self._multi_comp_jobs)
        self._log(
            f"\n=== СКЛЕЙКА: {len(self._multi_comp_jobs)} источник(ов), "
            f"{total_moments} момент(ов) ==="
        )
        title = " + ".join(d['video_title'] or 'видео' for d in self._multi_comp_jobs)[:80]
        self.merge_thread = CompilationMergeThread(self._multi_comp_jobs, CLIPS_DIR, title)
        self.merge_thread.log.connect(self._log)
        self.merge_thread.progress.connect(
            lambda p: self.progress_bar.setValue(90 + int(p * 0.1)))
        self.merge_thread.finished.connect(self._on_multi_comp_finished)
        self.merge_thread.start()

    def _on_multi_comp_finished(self, result):
        self.start_btn.setEnabled(True)
        if result:
            self.progress_bar.setValue(100)
            self._clips_result = [result]
            self._log(f"\n🎉 Компиляция из {len(self._multi_comp_jobs)} видео готова!")
            self.upload_btn.setVisible(True)
            try: os.startfile(CLIPS_DIR)
            except Exception: pass
        else:
            self._log("\n⚠️ Склейка не удалась.")

    def open_upload_dialog(self):
        if not self._credentials:
            QMessageBox.warning(self, "Авторизация", "Сначала авторизуйтесь."); return
        if not self._clips_result:
            QMessageBox.warning(self, "Клипы", "Сначала создайте клипы."); return
        UploadScheduleDialog(self._clips_result, self._credentials, self).exec()

    def open_clips_folder_dialog(self):
        """Позволяет загрузить на YouTube любые готовые .mp4 из выбранной папки
        (по умолчанию — clips, но можно выбрать любую другую) — без повторной
        генерации, в любой момент запуска приложения. Ищет и во вложенных
        папках (клипы теперь раскладываются по подпапкам на каждое видео)."""
        if not self._credentials:
            QMessageBox.warning(self, "Авторизация", "Сначала авторизуйтесь."); return

        start_dir = CLIPS_DIR if os.path.isdir(CLIPS_DIR) else ""
        folder = QFileDialog.getExistingDirectory(
            self, "Папка с готовыми клипами", start_dir
        )
        if not folder:
            return

        paths = []
        for root, _dirs, filenames in os.walk(folder):
            for fname in filenames:
                if fname.lower().endswith('.mp4'):
                    paths.append(os.path.join(root, fname))
        paths.sort()
        if not paths:
            QMessageBox.information(self, "Клипы", f"В папке {folder} нет .mp4 файлов."); return

        from moviepy import VideoFileClip
        clips = []
        for path in paths:
            fname = os.path.basename(path)
            title = re.sub(r'^\d+_', '', os.path.splitext(fname)[0])
            duration = 0.0
            try:
                with VideoFileClip(path) as vc:
                    duration = vc.duration
            except Exception:
                pass
            clips.append({
                'path': path, 'filename': fname, 'title': title,
                'duration': duration, 'virality_score': 0.0,
            })
        UploadScheduleDialog(clips, self._credentials, self).exec()
