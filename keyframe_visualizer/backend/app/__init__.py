"""Keyframe Visualizer backend package."""

from .torch_runtime import preload_torch_on_import

# Must run before OpenCV/EasyOCR/Ultralytics are imported and before any worker
# thread starts, otherwise Torch's Windows DLLs fail to initialise.
preload_torch_on_import()
