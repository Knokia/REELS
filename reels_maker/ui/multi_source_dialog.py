from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout,
)


class MultiSourceDialog(QDialog):
    """Собирает список источников (YouTube-ссылки и/или локальные файлы, по
    одному на строку) для режима «Компиляция из нескольких видео» — порядок
    строк определяет порядок сегментов в итоговом ролике."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Источники для компиляции")
        self.resize(560, 380)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Вставьте ссылки YouTube и/или пути к локальным файлам — по одной\n"
            "на строку. Порядок строк = порядок сегментов в итоговом видео."
        ))

        self.text = QTextEdit()
        self.text.setPlaceholderText(
            "https://www.youtube.com/watch?v=...\n"
            "https://www.youtube.com/watch?v=...\n"
            "D:\\видео\\мой_файл.mp4"
        )
        layout.addWidget(self.text)

        row = QHBoxLayout()
        add_files_btn = QPushButton("📁 Добавить файлы...")
        add_files_btn.clicked.connect(self._add_files)
        row.addWidget(add_files_btn)
        row.addStretch()
        layout.addLayout(row)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        ok_btn = QPushButton("🎬 Начать")
        ok_btn.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;padding:8px 20px;"
            "border-radius:5px;font-weight:bold;}"
        )
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите видео (можно несколько)", "",
            "Видео (*.mp4 *.avi *.mov *.mkv *.webm *.flv);;Все (*.*)"
        )
        if not files:
            return
        current = self.text.toPlainText()
        addition = "\n".join(files)
        self.text.setPlainText((current + "\n" + addition).strip() if current else addition)

    def sources(self) -> list:
        lines = [line.strip() for line in self.text.toPlainText().splitlines()]
        return [line for line in lines if line]
