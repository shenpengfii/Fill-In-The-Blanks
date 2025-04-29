# -*- mode: python ; coding: utf-8 -*-


# 注：全部使用相对路径
a = Analysis(
    # 项目源代码文件列表
    ['src/main.py', 'src/data.py', 'src/blankFiller.py', 
        'src/bilstm_crf.py', 'src/bilstm.py', 'src/crf.py'],
    # 项目源代码目录列表
    pathex=['src'],
    datas=[],
    binaries=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # 不弹出控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
