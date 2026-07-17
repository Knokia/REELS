import os
import re

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSlider,
    QSpinBox, QTextEdit, QVBoxLayout, QWidget, QInputDialog,
)

from .. import youtube
from ..config import CLIPS_DIR, FONT_SETTINGS
from ..pipeline import ProcessingThread
from .font_dialog import FontSettingsDialog
from .upload_dialog import UploadScheduleDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Reels Maker PRO")
        self.setMinimumSize(860, 820)
        self._clips_result = []
        self._accounts      = []
        self._credentials   = None
        self._batch_queue  = []
        self._batch_total  = 0
        self._batch_index  = 0
        self.setup_ui()

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

        zg2 = QGroupBox("⚙️ Динамический зум"); zl2 = QVBoxLayout()
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
        self.log_view.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#ddd;font-family:Consolas;"
            "font-size:11px;padding:10px;border-radius:5px;}"
        )
        layout.addWidget(self.log_view)

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
            self.log_view.append(
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
            self.log_view.append(f"✅ {os.path.basename(fp)}")

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
            self.log_view.append(f"✅ Канал добавлен: {result['title']}")
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
        )

    def start_processing(self):
        url = self.url_input.text().strip()
        if not url:
            self.log_view.append("❌ Вставьте URL или выберите файл"); return
        if not url.startswith('http') and not os.path.isfile(url):
            self.log_view.append(f"❌ Файл не найден: {url}"); return
        self.log_view.clear()
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False); self.upload_btn.setVisible(False)
        self.thread = self._build_thread(url)
        self.thread.log.connect(self.log_view.append)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, clips):
        self.start_btn.setEnabled(True); self._clips_result = clips
        if clips:
            self.log_view.append(f"\n🎉 Клипов: {len(clips)}")
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
        self.log_view.append(f"📦 Пакетная обработка: {self._batch_total} файл(ов)")
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False); self.upload_btn.setVisible(False)
        self._run_next_in_batch()

    def _run_next_in_batch(self):
        if self._batch_index >= self._batch_total:
            self.start_btn.setEnabled(True)
            if self._clips_result:
                self.log_view.append(f"\n🎉🎉 Пакет завершён. Всего клипов: {len(self._clips_result)}")
                self.upload_btn.setVisible(True)
                try: os.startfile(CLIPS_DIR)
                except Exception: pass
            else:
                self.log_view.append("\n⚠️ Пакет завершён, но клипов не получилось.")
            return

        fp = self._batch_queue[self._batch_index]
        self._batch_index += 1
        self.log_view.append(
            f"\n\n=== 📁 Файл {self._batch_index}/{self._batch_total}: {os.path.basename(fp)} ==="
        )
        self.thread = self._build_thread(fp)
        self.thread.log.connect(self.log_view.append)
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
