"""Load Torch once, early, on the process' main thread.

On Windows ``torch/__init__.py`` loads ``c10.dll`` and friends with
``ctypes.CDLL``.  That load fails with ``[WinError 1114] 动态链接库(DLL)初始化
例程失败`` when it happens late in the life of a process that already loaded
OpenCV/EasyOCR/ONNX, or when it happens on a worker thread instead of the main
thread.  ``examples/run_vsi.py`` never hits this because it imports the VSI
adapters at module scope; the web backend only touches Torch inside the session
and job worker threads (``asyncio.to_thread``), which is exactly the failing
pattern.  Importing Torch while the ``app`` package itself is being imported
restores the script's import order for the server too.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_LOCK = threading.RLock()
_REGISTERED_DLL_DIRS: set[str] = set()
_PRELOADED = False
_PRELOAD_ERROR: str | None = None

_WINDOWS_DLL_HINT = (
    "Torch 的 Windows 动态库加载失败。请确认后端与 run_vsi.py 使用同一个虚拟环境，"
    "并在该环境中重新安装匹配的 torch 版本（pip install --force-reinstall torch），"
    "同时安装 Microsoft Visual C++ 运行库。"
)


def torch_lib_dir() -> Path | None:
    """Locate ``torch/lib`` without importing Torch."""
    try:
        from importlib.util import find_spec

        spec = find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    candidates: list[Path] = []
    if spec is not None and spec.submodule_search_locations:
        candidates.append(Path(next(iter(spec.submodule_search_locations))) / "lib")
    candidates.append(Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def register_dll_directory() -> None:
    """Add ``torch/lib`` to the DLL search path (Windows, Python 3.8+)."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    lib_dir = torch_lib_dir()
    if lib_dir is None:
        return
    key = str(lib_dir)
    with _LOCK:
        if key in _REGISTERED_DLL_DIRS:
            return
        try:
            os.add_dll_directory(key)
        except OSError:
            return
        _REGISTERED_DLL_DIRS.add(key)


def _describe(error: BaseException) -> str:
    detail = f"{type(error).__name__}: {error}"
    if os.name == "nt" and getattr(error, "winerror", None) in {126, 127, 1114}:
        return f"{_WINDOWS_DLL_HINT} 原始错误：{detail}"
    return f"无法加载 Torch。原始错误：{detail}"


def preload_torch() -> str | None:
    """Import Torch once; return ``None`` on success or a message on failure."""
    global _PRELOADED, _PRELOAD_ERROR
    with _LOCK:
        if _PRELOADED:
            return _PRELOAD_ERROR
        register_dll_directory()
        try:
            import torch  # noqa: F401
        except Exception as error:  # noqa: BLE001 - reported to the caller verbatim
            _PRELOADED = True
            _PRELOAD_ERROR = _describe(error)
        else:
            _PRELOADED = True
            _PRELOAD_ERROR = None
        return _PRELOAD_ERROR


def require_torch() -> None:
    """Raise a readable error if Torch cannot be loaded in this process."""
    error = preload_torch()
    if error is not None:
        raise RuntimeError(error)


def _preload_requested() -> bool:
    override = os.environ.get("KFV_PRELOAD_TORCH", "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
    # Only Windows needs the eager main-thread import; elsewhere the lazy
    # adapter imports keep startup and the test suite fast.
    return os.name == "nt"


def preload_torch_on_import() -> None:
    if not _preload_requested():
        return
    error = preload_torch()
    if error is not None:
        print(f"[keyframe-visualizer] Torch 预加载失败：{error}", file=sys.stderr)
