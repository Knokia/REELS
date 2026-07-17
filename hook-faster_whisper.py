# hook-faster_whisper.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas          = collect_data_files('faster_whisper')
hiddenimports  = collect_submodules('faster_whisper')
hiddenimports += collect_submodules('ctranslate2')
datas         += collect_data_files('ctranslate2')