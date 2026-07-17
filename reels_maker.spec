# reels_maker.spec
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Собираем все данные нужных библиотек
datas = []
datas += collect_data_files('faster_whisper')
datas += collect_data_files('whisper')
datas += collect_data_files('mediapipe')
datas += collect_data_files('cv2')
datas += collect_data_files('llama_cpp')
datas += collect_data_files('yt_dlp')
datas += collect_data_files('moviepy')
datas += collect_data_files('certifi')

# Скрытые импорты
hiddenimports = []
hiddenimports += collect_submodules('mediapipe')
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('llama_cpp')
hiddenimports += collect_submodules('faster_whisper')
hiddenimports += collect_submodules('yt_dlp')
hiddenimports += collect_submodules('moviepy')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('google')
hiddenimports += collect_submodules('googleapiclient')
hiddenimports += [
    'PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtCore', 'PyQt6.QtGui',
    'numpy', 'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont',
    'librosa', 'soundfile', 'audioread',
    'google.auth', 'google.auth.transport.requests',
    'google.oauth2.credentials',
    'google_auth_oauthlib', 'google_auth_oauthlib.flow',
    'googleapiclient', 'googleapiclient.discovery', 'googleapiclient.http',
    'scenedetect', 'scenedetect.detectors',
    'huggingface_hub', 'tokenizers',
    'ctypes', 'ctypes.util',
    'pkg_resources', 'importlib_metadata',
    'charset_normalizer', 'certifi', 'urllib3', 'requests',
    'av', 'tqdm', 'regex', 'filelock',
]

a = Analysis(
    ['reels_maker_launcher.py'],  # ← имя твоего скрипта
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.'],              # ищем хуки в текущей папке
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'scipy',
        'pandas', 'jupyter', 'IPython',
        'test', 'tests', 'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI_Reels_Maker_PRO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # False = без чёрного окна консоли
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',         # путь к иконке (опционально)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI_Reels_Maker_PRO',
)