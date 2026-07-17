import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from PyQt6.QtWidgets import QApplication

from reels_maker import MainWindow

if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
