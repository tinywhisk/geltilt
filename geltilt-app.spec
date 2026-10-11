# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('geltilt')
hiddenimports += collect_submodules('cv2')


a = Analysis(
    ['C:/Users/tinywhisk/Desktop/geltilt_codearts/run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/tinywhisk/Desktop/geltilt_codearts/LICENSE', '.'), ('C:/Users/tinywhisk/Desktop/geltilt_codearts/THIRD_PARTY_NOTICES.md', '.'), ('C:/Users/tinywhisk/Desktop/geltilt_codearts/licenses', 'licenses'), ('C:/Users/tinywhisk/.geltilt/gelgenie_universal_1024.onnx', '.')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['geltilt.smears', 'scipy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='geltilt-app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
