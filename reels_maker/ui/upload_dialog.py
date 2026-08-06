from PyQt6.QtCore import QDate, QDateTime, Qt, QTime, QTimer, QUrl
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QTableWidget,
    QTableWidgetItem, QTextEdit, QTimeEdit, QVBoxLayout, QDialog,
)

from .. import youtube

CATEGORIES = {
    "1": "Фильмы", "10": "Музыка", "17": "Спорт", "20": "Игры",
    "22": "Блоги", "23": "Юмор", "24": "Развлечения", "27": "Образование", "28": "Наука"
}

# Категория, с которой стартуют все клипы. Раньше её никто не задавал, и список
# вставал на первый пункт словаря — «Фильмы», что для коротких нарезок неверно
# и уезжало в YouTube молча.
DEFAULT_CATEGORY = "24"  # Развлечения

COL_FILE, COL_TITLE, COL_DUR, COL_VIRAL, COL_UPLOAD, COL_SCHEDULE, \
    COL_DATE, COL_TIME, COL_PRIVACY, COL_CATEGORY, COL_PLAYLIST, \
    COL_THUMB, COL_DESC, COL_PREVIEW = range(14)


class UploadScheduleDialog(QDialog):
    def __init__(self, clips, credentials, parent=None):
        super().__init__(parent)
        self.clips        = clips
        self.credentials  = credentials
        self._thumbnails  = {}
        self._descriptions = {}
        self._playlists   = []
        self.setWindowTitle("📤 Загрузка на YouTube")
        self.setMinimumSize(1280, 680)
        self._setup()

    def _setup(self):
        layout = QVBoxLayout(self)
        hdr = QLabel("📤 Настройка загрузки клипов на YouTube")
        hdr.setStyleSheet("font-size:16px;font-weight:bold;color:#f44336;padding:8px;")
        layout.addWidget(hdr)

        self.channel_label = QLabel("✅ Авторизованы в Google")
        layout.addWidget(self.channel_label)
        self._load_channel_and_quota()

        for lbl, attr, default in [
            ("🏷️ Теги:", 'global_tags', "shorts, reels, ai, viral"),
            ("📝 Описание (по умолчанию):", 'global_desc', "Создано с помощью AI Reels Maker PRO 🎬"),
        ]:
            row = QHBoxLayout(); row.addWidget(QLabel(lbl))
            edit = QLineEdit(default); setattr(self, attr, edit)
            row.addWidget(edit); layout.addLayout(row)
        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("📂 Категория:"))
        self.global_category = QComboBox()
        for vid, label in CATEGORIES.items():
            self.global_category.addItem(label, vid)
        self.global_category.setCurrentIndex(self.global_category.findData(DEFAULT_CATEGORY))
        self.global_category.currentIndexChanged.connect(self._apply_category_to_all)
        cat_row.addWidget(self.global_category); cat_row.addStretch()
        layout.addLayout(cat_row)

        desc_hint = QLabel(
            "Описание и категория применяются ко всем клипам сразу — "
            "у каждого можно задать своё: описание через кнопку «📝», "
            "категорию — в её колонке таблицы."
        )
        desc_hint.setStyleSheet("color:#888;font-size:10px;padding:0 0 4px 2px;")
        layout.addWidget(desc_hint)

        self.table = QTableWidget()
        self.table.setColumnCount(14)
        self.table.setHorizontalHeaderLabels([
            "Файл", "Название", "Длит.", "Viral",
            "Загрузить?", "Расписание?", "Дата", "Время", "Приватность",
            "Категория", "Плейлист", "Превью", "Описание", "▶",
        ])
        self.table.horizontalHeader().setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(COL_TITLE, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().resizeSection(COL_PLAYLIST, 140)
        self.table.horizontalHeader().resizeSection(COL_THUMB, 110)
        self.table.horizontalHeader().resizeSection(COL_DESC, 110)
        self.table.horizontalHeader().resizeSection(COL_PREVIEW, 36)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget{background:#1e1e1e;color:#ddd;gridline-color:#333;
                border:1px solid #444;border-radius:8px;}
            QHeaderView::section{background:#2a2a2a;color:#aaa;border:none;
                padding:6px;font-weight:bold;}
        """)

        self._playlists = self._safe_list_playlists()
        for idx, clip in enumerate(self.clips):
            self._add_row(idx, clip)
        # Строки созданы — только теперь есть куда разложить стартовую категорию.
        self._apply_category_to_all()
        layout.addWidget(self.table)

        br3 = QHBoxLayout(); br3.addStretch()
        cb3 = QPushButton("Отмена"); cb3.clicked.connect(self.reject)
        br3.addWidget(cb3)
        ub = QPushButton("🚀 Загрузить на YouTube")
        ub.setStyleSheet(
            "QPushButton{background:#f44336;color:white;padding:10px 24px;"
            "border-radius:8px;font-weight:bold;font-size:14px;}"
            "QPushButton:hover{background:#d32f2f;}"
        )
        ub.clicked.connect(self._start_upload); br3.addWidget(ub)
        layout.addLayout(br3)

        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(160)
        self.log_view.setStyleSheet(
            "QTextEdit{background:#111;color:#ddd;font-family:Consolas;"
            "font-size:11px;padding:8px;border-radius:8px;}"
        )
        layout.addWidget(self.log_view)

    # ── Setup helpers ─────────────────────────────────────────

    def _load_channel_and_quota(self):
        try:
            info = youtube.get_channel_info(self.credentials)
            title = info.get("title", "?")
        except Exception as e:
            title = f"не удалось получить ({e})"
        remaining = youtube.quota_remaining()
        max_uploads = remaining // youtube.QUOTA_COST_VIDEO_INSERT
        color = "#4CAF50" if max_uploads >= len(self.clips) else "#FFC107" if max_uploads > 0 else "#f44336"
        self.channel_label.setText(
            f"✅ Канал: {title}  |  Квота на сегодня: ~{max_uploads} загрузок доступно "
            f"({remaining}/{youtube.DEFAULT_DAILY_QUOTA} units)"
        )
        self.channel_label.setStyleSheet(f"color:{color};font-size:12px;padding:4px;")

    def _safe_list_playlists(self):
        try:
            return youtube.list_playlists(self.credentials)
        except Exception:
            return []

    def _apply_category_to_all(self):
        """Раскладывает глобальную категорию по всем строкам таблицы.

        Выпадающий список в строке при этом остаётся рабочим: после смены
        общего значения категорию отдельного клипа можно переопределить — так
        же, как описание. В API уходит именно значение из строки
        (см. currentData на COL_CATEGORY в _start_upload).
        """
        vid = self.global_category.currentData()
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, COL_CATEGORY)
            if combo is None:
                continue
            index = combo.findData(vid)
            if index >= 0:
                combo.setCurrentIndex(index)

    def _add_row(self, idx, clip):
        self.table.insertRow(idx)
        self.table.setItem(idx, COL_FILE, QTableWidgetItem(clip['filename']))
        ti = QTableWidgetItem(clip['title'])
        ti.setFlags(ti.flags() | Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(idx, COL_TITLE, ti)
        self.table.setItem(idx, COL_DUR, QTableWidgetItem(f"{clip['duration']:.0f}с"))
        vs  = clip.get('virality_score', 0.0)
        vsi = QTableWidgetItem(f"{vs:.2f}")
        vsi.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        vsi.setForeground(
            QColor('#4CAF50' if vs >= 0.7 else '#FFC107' if vs >= 0.4 else '#f44336')
        )
        self.table.setItem(idx, COL_VIRAL, vsi)
        ecb = QCheckBox(); ecb.setChecked(True)
        self.table.setCellWidget(idx, COL_UPLOAD, ecb)
        scb = QCheckBox()
        scb.stateChanged.connect(lambda _, r=idx: self._toggle_schedule(r))
        self.table.setCellWidget(idx, COL_SCHEDULE, scb)
        de = QDateEdit()
        de.setDate(QDate.currentDate().addDays(1))
        de.setDisplayFormat("yyyy-MM-dd")
        de.setToolTip("Используется только если включено «Расписание?»")
        self.table.setCellWidget(idx, COL_DATE, de)
        te = QTimeEdit()
        te.setTime(QTime.fromString("10:00", "HH:mm"))
        te.setToolTip("Используется только если включено «Расписание?»")
        self.table.setCellWidget(idx, COL_TIME, te)
        pr = QComboBox(); pr.addItems(["public", "unlisted", "private"])
        self.table.setCellWidget(idx, COL_PRIVACY, pr)
        cat = QComboBox()
        for vid, label in CATEGORIES.items():
            cat.addItem(label, vid)
        self.table.setCellWidget(idx, COL_CATEGORY, cat)

        pl = QComboBox()
        pl.addItem("— без плейлиста —", None)
        for p in self._playlists:
            pl.addItem(p['title'], p['id'])
        self.table.setCellWidget(idx, COL_PLAYLIST, pl)

        thumb_btn = QPushButton("Выбрать...")
        thumb_btn.clicked.connect(lambda _, r=idx: self._pick_thumbnail(r))
        self.table.setCellWidget(idx, COL_THUMB, thumb_btn)

        desc_btn = QPushButton("📝")
        desc_btn.setToolTip("Своё описание для этого клипа (по умолчанию — общее сверху)")
        desc_btn.clicked.connect(lambda _, r=idx: self._edit_description(r))
        self.table.setCellWidget(idx, COL_DESC, desc_btn)

        preview_btn = QPushButton("▶")
        preview_btn.clicked.connect(lambda _, c=clip: self._preview_clip(c))
        self.table.setCellWidget(idx, COL_PREVIEW, preview_btn)

    def _pick_thumbnail(self, row):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Превью для видео", "", "Изображения (*.jpg *.jpeg *.png)"
        )
        if fp:
            self._thumbnails[row] = fp
            self.table.cellWidget(row, COL_THUMB).setText(
                "✅ " + fp.split('/')[-1].split('\\')[-1]
            )

    def _preview_clip(self, clip):
        QDesktopServices.openUrl(QUrl.fromLocalFile(clip['path']))

    def _edit_description(self, row):
        current = self._descriptions.get(row, self.global_desc.text())
        dlg = QDialog(self)
        dlg.setWindowTitle(f"📝 Описание — {self.clips[row]['filename']}")
        dlg.setMinimumSize(520, 320)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Описание для этого клипа (пусто = использовать общее по умолчанию):"))
        text_edit = QTextEdit(); text_edit.setPlainText(current)
        v.addWidget(text_edit)
        br = QHBoxLayout(); br.addStretch()
        reset_btn = QPushButton("↩ Сбросить на общее")
        reset_btn.clicked.connect(lambda: text_edit.setPlainText(self.global_desc.text()))
        cancel_btn = QPushButton("Отмена"); cancel_btn.clicked.connect(dlg.reject)
        save_btn = QPushButton("Сохранить"); save_btn.clicked.connect(dlg.accept)
        br.addWidget(reset_btn); br.addWidget(cancel_btn); br.addWidget(save_btn)
        v.addLayout(br)
        if dlg.exec():
            new_text = text_edit.toPlainText()
            if new_text.strip() and new_text != self.global_desc.text():
                self._descriptions[row] = new_text
                self.table.cellWidget(row, COL_DESC).setText("📝✅")
                self.table.cellWidget(row, COL_DESC).setToolTip("Своё описание задано — нажмите, чтобы изменить")
            else:
                self._descriptions.pop(row, None)
                self.table.cellWidget(row, COL_DESC).setText("📝")
                self.table.cellWidget(row, COL_DESC).setToolTip(
                    "Своё описание для этого клипа (по умолчанию — общее сверху)"
                )

    def _toggle_schedule(self, row):
        # Дата/время всегда доступны для выбора — но используются только если
        # включено «Расписание?» (см. _start_upload). YouTube требует приватность
        # "private" для отложенной публикации.
        if self.table.cellWidget(row, COL_SCHEDULE).isChecked():
            self.table.cellWidget(row, COL_PRIVACY).setCurrentText("private")

    # ── Upload ────────────────────────────────────────────────

    def _log(self, text):
        self.log_view.append(text)
        # Дублируем в файловый лог главного окна, чтобы история загрузок
        # на YouTube тоже сохранялась в logs/run_*.txt.
        parent = self.parent()
        if parent is not None and hasattr(parent, "session_log"):
            parent.session_log.write(f"[upload] {text}")

    def _start_upload(self):
        configs = []
        for row in range(self.table.rowCount()):
            if not self.table.cellWidget(row, COL_UPLOAD).isChecked():
                continue
            is_sched   = self.table.cellWidget(row, COL_SCHEDULE).isChecked()
            if is_sched:
                # Дата/время в таблице — локальное время пользователя, а YouTube
                # (publishAt) ожидает UTC. Раньше к введённому времени просто
                # приклеивался "+00:00", то есть 10:00 по Москве уходило в API
                # как "10:00 UTC" и YouTube показывал/публиковал в 13:00 по
                # Москве. Явно конвертируем локальное время в UTC через Qt.
                local_dt = QDateTime(
                    self.table.cellWidget(row, COL_DATE).date(),
                    self.table.cellWidget(row, COL_TIME).time(),
                    Qt.TimeSpec.LocalTime,
                )
                publish_at = local_dt.toUTC().toString("yyyy-MM-ddTHH:mm:ss") + "Z"
            else:
                publish_at = None
            privacy    = "private" if is_sched else self.table.cellWidget(row, COL_PRIVACY).currentText()
            configs.append({
                "clip":          self.clips[row],
                "title":         self.table.item(row, COL_TITLE).text(),
                "description":   self._descriptions.get(row, self.global_desc.text()),
                "tags":          [t.strip() for t in self.global_tags.text().split(",") if t.strip()],
                "category":      self.table.cellWidget(row, COL_CATEGORY).currentData(),
                "privacy":       privacy,
                "publish_at":    publish_at,
                "playlist_id":   self.table.cellWidget(row, COL_PLAYLIST).currentData(),
                "thumbnail_path": self._thumbnails.get(row),
            })
        if not configs:
            QMessageBox.information(self, "Инфо", "Ничего не выбрано.")
            return

        est_units = len(configs) * youtube.QUOTA_COST_VIDEO_INSERT
        est_units += sum(youtube.QUOTA_COST_THUMBNAIL_SET for c in configs if c["thumbnail_path"])
        est_units += sum(youtube.QUOTA_COST_PLAYLIST_INSERT for c in configs if c["playlist_id"])
        if est_units > youtube.quota_remaining():
            if QMessageBox.question(
                self, "Квота",
                f"Похоже, этой загрузки ({est_units} units) не хватит дневной квоты "
                f"YouTube API (~{youtube.quota_remaining()} units осталось). "
                f"Часть загрузок может завершиться ошибкой. Продолжить?"
            ) != QMessageBox.StandardButton.Yes:
                return

        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self._upload_next(configs, 0)

    def _upload_next(self, configs, idx):
        if idx >= len(configs):
            self.progress_bar.setValue(100)
            self._log(f"\n✅ Загружено {idx} видео.")
            QMessageBox.information(self, "Готово", "✅ Все видео загружены!")
            self.accept(); return
        cfg = configs[idx]; total = len(configs)
        self.progress_bar.setValue(int(idx / total * 100))
        self._log(f"🎬 {idx+1}/{total}: {cfg['title']}")
        try:
            result = youtube.upload_video(
                self.credentials, cfg['clip']['path'], cfg['title'], cfg['description'],
                cfg['tags'], cfg['category'], cfg['privacy'], publish_at=cfg['publish_at'],
                thumbnail_path=cfg['thumbnail_path'], playlist_id=cfg['playlist_id'],
            )
            self._log(f"   ✅ {result['url']}")
        except Exception as e:
            self._log(f"   ❌ {e}")
        QTimer.singleShot(500, lambda: self._upload_next(configs, idx + 1))
