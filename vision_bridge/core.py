"""
vision_bridge.core
~~~~~~~~~~~~~~~~~~

The primary orchestration module. Houses the VisionBridgeModule class, 
coordinating background frame grabbers, encoding, VLM analysis, queueing, 
and dual-pattern event distribution.
"""

import threading
import queue
import time
from typing import Tuple, Union, Callable, Set, Generator, Optional, Any

from vision_bridge.types import VisionEvent, VisionBridgeError
from vision_bridge.camera import CameraCapture
from vision_bridge.clients.base import VLMClient
from vision_bridge.clients import get_vlm_client


class VisionBridgeModule:
    """
    Main orchestrator that connects camera streams to Vision-Language Models (VLMs) 
    in a non-blocking background thread. It publishes structured events using 
    both the Iterator Pattern (Python Generators) and the Observer Pattern (Callbacks).
    """

    def __init__(
        self,
        vlm_model: Union[str, VLMClient] = "llama3.2-vision:1b",
        camera_index: Union[int, str] = 0,
        frame_rate: float = 2.0,
        resolution: Tuple[int, int] = (448, 448),
        provider: str = "ollama",
        **client_kwargs: Any
    ):
        """
        Initializes the VisionBridgeModule.
        
        Args:
            vlm_model: The target VLM. Can be a string identifier (e.g. "llama3.2-vision:1b", 
                       "gpt-4o-mini", "mock") or an instance of VLMClient.
            camera_index: Index of physical system camera (int, e.g., 0) or RTSP stream (str),
                         or "mock" to run offline simulations.
            frame_rate: Target frames to analyze per second (e.g., 1.0 = 1 frame/sec).
            resolution: Image dimensions (width, height) to resize before sending to VLM.
            provider: VLM client provider name ('ollama', 'openai', 'gemini', 'mock'). 
                      Only used if `vlm_model` is passed as a string.
            **client_kwargs: Additional parameters forwarded to the client factory 
                             (e.g., api_key, simulated_latency, etc.).
        """
        self.camera_index = camera_index
        self.delay = 1.0 / frame_rate if frame_rate > 0 else 0.5
        self.resolution = resolution

        # Standardize the VLMClient implementation
        if isinstance(vlm_model, VLMClient):
            self.vlm_client = vlm_model
        else:
            # If the user sets model to "mock", force the mock provider for convenience
            target_provider = "mock" if vlm_model == "mock" else provider
            self.vlm_client = get_vlm_client(target_provider, model_name=vlm_model, **client_kwargs)

        # Queues for event processing
        self.event_queue: queue.Queue = queue.Queue()
        self.callback_queue: queue.Queue = queue.Queue()

        # Observer callback storage
        self.callbacks: Set[Callable[[VisionEvent], None]] = set()
        self.callback_lock = threading.Lock()

        # Lifecycle flags & background handles
        self.is_running = False
        self.analysis_thread: Optional[threading.Thread] = None
        self.dispatcher_thread: Optional[threading.Thread] = None
        self.camera_capture: Optional[CameraCapture] = None

    def _capture_and_analyze_loop(self, prompt: str) -> None:
        """Internal analysis loop. Runs in a dedicated background thread."""
        import cv2
        import base64

        # Start the background frame ingestion thread
        self.camera_capture = CameraCapture(
            camera_index=self.camera_index,
            resolution=self.resolution
        )
        self.camera_capture.start()

        last_processed_frame_id = -1

        while self.is_running:
            iteration_start = time.time()
            
            # Fetch latest frame safely from capture thread
            frame, frame_id = self.camera_capture.get_latest_frame()

            # Skip iteration if no frame captured, or frame has already been analyzed
            if frame is None or frame_id == last_processed_frame_id:
                time.sleep(0.01)
                continue

            last_processed_frame_id = frame_id

            # Compress and encode frame as Base64 JPEG
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    raise VisionBridgeError("OpenCV image compression failed.")
                b64_image = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"[VisionBridge Error]: Base64 encoding failure: {e}")
                time.sleep(0.05)
                continue

            # Submit frame to VLM Client
            vlm_start = time.time()
            try:
                analysis = self.vlm_client.analyze_image(b64_image, prompt)
                vlm_end = time.time()
                latency_ms = (vlm_end - vlm_start) * 1000.0

                # Form structured VisionEvent object
                event = VisionEvent(
                    frame_id=frame_id,
                    raw_text=analysis,
                    timestamp=iteration_start,
                    response_time_ms=latency_ms,
                    metadata={
                        "provider": self.vlm_client.__class__.__name__,
                        "model": getattr(self.vlm_client, "model_name", "custom"),
                        "resolution": self.resolution,
                        "camera_index": self.camera_index
                    }
                )

                # Push to generator stream queue
                self.event_queue.put(event)

                # Push to callback dispatch queue if observers exist
                with self.callback_lock:
                    if self.callbacks:
                        self.callback_queue.put(event)

            except Exception as e:
                print(f"[VisionBridge Engine Error]: VLM Analysis failed: {e}")

            # Enforce target framerate delay relative to loop start
            elapsed = time.time() - iteration_start
            sleep_time = max(0.005, self.delay - elapsed)
            time.sleep(sleep_time)

        # Cleanup camera handle
        if self.camera_capture:
            self.camera_capture.stop()
            self.camera_capture = None

    def _callback_dispatcher_loop(self) -> None:
        """Internal dispatch loop. Expose observers to async event workers."""
        while self.is_running or not self.callback_queue.empty():
            try:
                # 200ms timeout prevents CPU pinning during idle phases
                event = self.callback_queue.get(timeout=0.2)
                
                with self.callback_lock:
                    active_callbacks = list(self.callbacks)
                
                # Execute registered listeners
                for cb in active_callbacks:
                    try:
                        cb(event)
                    except Exception as e:
                        # Defensive: catch observer errors to protect core dispatch integrity
                        cb_name = getattr(cb, "__name__", str(cb))
                        print(f"[VisionBridge Dispatch Error]: Callback '{cb_name}' raised: {e}")
            except queue.Empty:
                continue

    def start(
        self,
        prompt: str = "Describe any distinct action occurring in one brief sentence. If nothing is happening, say 'Idle'."
    ) -> None:
        """
        Starts the background processing and VLM analysis threads.
        
        Args:
            prompt: The text instruction / prompt passed to the visual engine 
                    for frame filtering/descriptions.
        """
        if self.is_running:
            return

        self.is_running = True

        # Clear active queues
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
        while not self.callback_queue.empty():
            try:
                self.callback_queue.get_nowait()
            except queue.Empty:
                break

        # Start primary analysis thread
        self.analysis_thread = threading.Thread(
            target=self._capture_and_analyze_loop,
            args=(prompt,),
            name="VisionBridgeAnalysisThread",
            daemon=True
        )
        self.analysis_thread.start()

        # Start secondary callback distribution thread
        self.dispatcher_thread = threading.Thread(
            target=self._callback_dispatcher_loop,
            name="VisionBridgeDispatcherThread",
            daemon=True
        )
        self.dispatcher_thread.start()

        model_name = getattr(self.vlm_client, "model_name", self.vlm_client.__class__.__name__)
        print(f"[VisionBridge]: Module active using client model: {model_name}.")

    def stop(self) -> None:
        """Stops all background camera reading, analysis, and dispatching threads."""
        if not self.is_running:
            return

        self.is_running = False

        # Signal threads to finish by stopping dependent streams
        if self.camera_capture:
            self.camera_capture.stop()

        # Join workers to gracefully free resources
        if self.analysis_thread:
            self.analysis_thread.join(timeout=3.0)
            self.analysis_thread = None

        if self.dispatcher_thread:
            self.dispatcher_thread.join(timeout=3.0)
            self.dispatcher_thread = None

        print("[VisionBridge]: Module shutdown successfully completed.")

    def add_callback(self, callback: Callable[[VisionEvent], None]) -> None:
        """
        Registers an event listener (Observer Pattern) to be called 
        asynchronously when a new frame event is analyzed.
        
        Args:
            callback: Callable that accepts a single `VisionEvent` parameter.
        """
        with self.callback_lock:
            self.callbacks.add(callback)

    def remove_callback(self, callback: Callable[[VisionEvent], None]) -> None:
        """
        Unregisters an event listener.
        
        Args:
            callback: The Callable to remove from the observer set.
        """
        with self.callback_lock:
            self.callbacks.discard(callback)

    def get_events(self, block: bool = True, timeout: Optional[float] = 0.5) -> Generator[VisionEvent, None, None]:
        """
        Generates/yields analysis results as they arrive (Iterator Pattern).
        
        Args:
            block: If True, blocks waiting for events.
            timeout: Blocks up to `timeout` seconds before checking module state 
                     to avoid holding threads forever.
                     
        Yields:
            A stream of VisionEvent objects.
        """
        while self.is_running or not self.event_queue.empty():
            try:
                yield self.event_queue.get(block=block, timeout=timeout)
            except queue.Empty:
                continue
