"""
vision_bridge.camera
~~~~~~~~~~~~~~~~~~~~

Manages background thread camera ingestion. It separates camera hardware 
bottlenecks from the application thread, avoids frame buffer lag, and provides 
robust connection recovery and offline mock pattern fallback.
"""

import threading
import time
from typing import Tuple, Union, Optional
from vision_bridge.types import CameraError


class CameraCapture(threading.Thread):
    """
    Background worker thread that ingests frames from a video source or 
    simulated camera feed. It keeps only the latest frame in memory, 
    eliminating OpenCV's internal frame buffering latency.
    """

    def __init__(
        self,
        camera_index: Union[int, str] = 0,
        resolution: Tuple[int, int] = (448, 448),
        max_reconnect_attempts: int = 3,
        reconnect_delay: float = 2.0
    ):
        """
        Initializes the CameraCapture thread.
        
        Args:
            camera_index: Standard camera index (int, e.g., 0) or RTSP stream link (str)
                         or "mock" to generate moving diagnostic test patterns.
            resolution: Tuple representing output image dimensions (width, height).
            max_reconnect_attempts: Retries if the hardware stream drops.
            reconnect_delay: Sleep time in seconds before attempting stream reset.
        """
        super().__init__(name="CameraCaptureThread", daemon=True)
        self.camera_index = camera_index
        self.resolution = resolution
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.frame_id = 0
        self.cap = None

    def run(self) -> None:
        """Core camera reading loop running in a background thread."""
        try:
            import cv2
        except ImportError as e:
            raise CameraError(
                "OpenCV ('opencv-python') is required. Please install it using: "
                "pip install opencv-python"
            ) from e

        self.is_running = True
        reconnect_count = 0
        
        while self.is_running:
            # Handle mock diagnostic camera generator
            if self.camera_index == "mock":
                try:
                    import numpy as np
                except ImportError as e:
                    raise CameraError(
                        "The 'numpy' package is required to run mock test pattern generator. "
                        "Please run: pip install numpy"
                    ) from e
                
                # Render diagnostic canvas
                frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
                
                # Draw grid patterns
                for i in range(0, self.resolution[0], 40):
                    cv2.line(frame, (i, 0), (i, self.resolution[1]), (40, 40, 40), 1)
                for j in range(0, self.resolution[1], 40):
                    cv2.line(frame, (0, j), (self.resolution[0], j), (40, 40, 40), 1)
                
                # Draw text indicator with moving timestamp
                current_time = time.strftime('%H:%M:%S')
                cv2.putText(
                    frame,
                    f"TEST PATTERN FEED - frame #{self.frame_id}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f"Time: {current_time}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                
                # Draw dynamic moving target to simulate activity
                pos_x = int((time.time() * 50) % (self.resolution[0] - 60)) + 30
                cv2.circle(frame, (pos_x, 250), 30, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    "MOVING OBJECT",
                    (pos_x - 40, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA
                )

                with self.frame_lock:
                    self.latest_frame = frame
                    self.frame_id += 1
                
                # Lock frame rate of simulated stream to ~10 FPS
                time.sleep(0.1)
                continue

            # Hardware Capture Initialization & Recovery
            if self.cap is None or not self.cap.isOpened():
                print(f"[CameraCapture]: Connecting to source index/URL: {self.camera_index}...")
                self.cap = cv2.VideoCapture(self.camera_index)
                
                if not self.cap.isOpened():
                    reconnect_count += 1
                    print(
                        f"[CameraCapture Warning]: Unable to open camera source {self.camera_index} "
                        f"(Attempt {reconnect_count}/{self.max_reconnect_attempts})"
                    )
                    
                    if reconnect_count >= self.max_reconnect_attempts:
                        print("[CameraCapture Error]: Exceeded maximum reconnection attempts. Thread halting.")
                        self.is_running = False
                        break
                    
                    time.sleep(self.reconnect_delay)
                    continue
                else:
                    reconnect_count = 0
                    print(f"[CameraCapture]: Successful connection to source {self.camera_index}.")

            # Ingest physical frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[CameraCapture Warning]: Empty frame received from stream. Attempting camera reset...")
                self.release_cap()
                time.sleep(self.reconnect_delay)
                continue

            # Resize frame for efficient VLM bandwidth
            try:
                resized = cv2.resize(frame, self.resolution)
                with self.frame_lock:
                    self.latest_frame = resized
                    self.frame_id += 1
            except Exception as e:
                print(f"[CameraCapture Error]: Frame resize processing failure: {e}")

            # Yield thread to prevent CPU thread starvation
            time.sleep(0.005)

        self.release_cap()

    def get_latest_frame(self) -> Tuple[Optional[object], int]:
        """
        Retrieves the most recent frame and its serial ID in a thread-safe manner.
        
        Returns:
            A tuple of (frame, frame_id), where frame is a numpy ndarray (cv2 image)
            or None if no frame has been captured yet.
        """
        with self.frame_lock:
            return self.latest_frame, self.frame_id

    def stop(self) -> None:
        """Signals the loop to stop and blocks until the thread terminates cleanly."""
        self.is_running = False
        if self.is_alive():
            self.join(timeout=3.0)

    def release_cap(self) -> None:
        """Releases the OpenCV video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[CameraCapture]: Hardware camera device handle released.")
