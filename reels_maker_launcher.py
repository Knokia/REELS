# reels_maker_launcher.py
"""
Лаунчер для .exe сборки.
Перехватывает необработанные исключения и показывает их в диалоге.
"""
import sys
import os
import traceback

# Фикс для PyInstaller — добавляем папку exe в PATH
if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
    os.environ['PATH'] = app_dir + os.pathsep + os.environ.get('PATH', '')

    # Фикс для llama_cpp DLL
    llama_dir = os.path.join(app_dir, '_internal', 'llama_cpp')
    if os.path.exists(llama_dir):
        os.add_dll_directory(llama_dir)

    # Фикс для mediapipe
    mp_dir = os.path.join(app_dir, '_internal', 'mediapipe')
    if os.path.exists(mp_dir):
        os.environ['MEDIAPIPE_RESOURCE_DIR'] = mp_dir


def handle_exception(exc_type, exc_value, exc_tb):
    """Глобальный обработчик необработанных исключений."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    error_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))

    # Сохраняем лог ошибки
    log_path = os.path.join(os.path.expanduser('~'), 'reels_maker_error.log')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(error_text)
    except Exception:
        pass

    # Показываем диалог
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        if not QApplication.instance():
            app = QApplication(sys.argv)
        msg = QMessageBox()
        msg.setWindowTitle("Критическая ошибка")
        msg.setText(f"Произошла ошибка:\n\n{str(exc_value)}")
        msg.setDetailedText(error_text)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.exec()
    except Exception:
        print(error_text)


sys.excepthook = handle_exception

# Запускаем основное приложение
try:
    from reels_maker import MainWindow
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
except Exception as e:
    handle_exception(type(e), e, e.__traceback__)