# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# cheshm discovers its detectors at runtime via importlib walks, so
# PyInstaller's static analysis can't see them. Pull every Python
# submodule, every compiled .dylib/.so/.dll next to a detector core,
# and any non-Python data files (icons, etc.) the package ships.
hiddenimports = collect_submodules("cheshm")
cheshm_binaries = collect_dynamic_libs("cheshm")
cheshm_datas = collect_data_files("cheshm")


a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=cheshm_binaries,
    datas=cheshm_datas,
    hiddenimports=hiddenimports,
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
    name="Cheshm",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["bindings/python/cheshm/gui/icon.icns"],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Cheshm",
)
app = BUNDLE(
    coll,
    name="Cheshm.app",
    icon="bindings/python/cheshm/gui/icon.icns",
    bundle_identifier="ir.mh-salari.cheshm",
)
