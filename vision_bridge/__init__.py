"""
Vision Bridge Module
~~~~~~~~~~~~~~~~~~~~

A highly robust, plug-and-play middleware that captures frames from a camera 
feed, optimizes them, passes them to a Vision-Language Model (VLM), and emits 
structured text events to downstream reasoning loops via Generators or Callbacks.

Basic Usage:
    >>> from vision_bridge import VisionBridgeModule
    >>> bridge = VisionBridgeModule(vlm_model="mock")
    >>> bridge.start("Is there a person in the frame?")
    >>> for event in bridge.get_events():
    ...     print(event)
"""

from vision_bridge.types import (
    VisionBridgeError,
    CameraError,
    VLMClientError,
    VisionEvent
)
from vision_bridge.core import VisionBridgeModule

__version__ = "1.0.0"
__all__ = [
    "VisionBridgeModule",
    "VisionEvent",
    "VisionBridgeError",
    "CameraError",
    "VLMClientError"
]
