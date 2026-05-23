"""
vision_bridge.types
~~~~~~~~~~~~~~~~~~~

Defines core data structures, type annotations, and custom exceptions used
throughout the Vision Bridge Module. These types ensure strong contracts between
the VLM engine, the camera capture system, and consumer applications.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


class VisionBridgeError(Exception):
    """Base exception for all Vision Bridge Module related errors."""
    pass


class CameraError(VisionBridgeError):
    """Raised when camera initialization, frame capture, or configuration fails."""
    pass


class VLMClientError(VisionBridgeError):
    """Raised when VLM inference, authentication, or API communication fails."""
    pass


@dataclass(frozen=True)
class VisionEvent:
    """
    Represents a structured visual event emitted by the Vision Bridge.
    
    This immutable data structure captures both the VLM analysis and relevant 
    system telemetry at the time of frame processing.
    """
    
    frame_id: int
    """Unique sequential identifier for the processed frame."""
    
    raw_text: str
    """The raw text description or response returned by the VLM."""
    
    timestamp: float
    """The POSIX timestamp (time.time()) when the frame was captured."""
    
    response_time_ms: float
    """The inference latency in milliseconds for the VLM API call."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Additional context information (e.g., model name, resolution, 
    camera index, prompt parameters).
    """

    @property
    def datetime(self) -> datetime:
        """Helper property to retrieve the event timestamp as a datetime object."""
        return datetime.fromtimestamp(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the event data to a clean dictionary representation for logging/serialisation."""
        return {
            "frame_id": self.frame_id,
            "raw_text": self.raw_text,
            "timestamp": self.timestamp,
            "datetime": self.datetime.isoformat(),
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        return f"[Event #{self.frame_id} | {self.datetime.strftime('%H:%M:%S.%f')[:-3]}]: {self.raw_text}"
