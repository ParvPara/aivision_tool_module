"""
examples.main_observer
~~~~~~~~~~~~~~~~~~~~~~

Demonstrates how to integrate the Vision Bridge using the Observer Pattern 
(Callbacks). The main application thread remains fully unblocked while 
asynchronous callbacks handle visual telemetry events.
"""

import sys
import os
import time

# Ensure parent package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision_bridge import VisionBridgeModule, VisionEvent


def safety_monitoring_callback(event: VisionEvent) -> None:
    """
    An event observer callback. Runs asynchronously in a dedicated dispatch 
    thread whenever a frame analysis is completed by the VLM.
    
    Args:
        event: The structured VisionEvent data payload.
    """
    text = event.raw_text
    print(f"\n🔔 [Asynchronous Callback Dispatcher] Received Event #{event.frame_id}:")
    print(f"   Timestamp : {event.datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"   Observation: {text}")
    print(f"   VLM Latency: {event.response_time_ms:.1f}ms")

    # Evaluate business rules asynchronously
    text_lower = text.lower()
    if "person" in text_lower or "technician" in text_lower or "worker" in text_lower:
        print("   🚨 [ALARM - SECURITY]: Person detected on active camera stream!")
    if "wrench" in text_lower or "tool" in text_lower or "calibration" in text_lower:
        print("   🔧 [AUDIT - INVENTORY]: Operational hand tool event logged.")


def main():
    # 1. Initialize the module (low rate of 0.5 FPS, camera index 0)
    print("[VisionBridge Demo]: Initializing Vision Bridge Module...")
    vision_layer = VisionBridgeModule(
        vlm_model="llama3.2-vision:1b",
        camera_index=0,
        frame_rate=0.5,
        provider="ollama"
    )

    # 2. Register the listener (Observer Pattern)
    vision_layer.add_callback(safety_monitoring_callback)

    # 3. Define the filtering prompt and start the engine
    prompt = "Is there a person or hand tool visible? Describe the scene in a brief sentence."
    vision_layer.start(prompt=prompt)

    try:
        print("\n[System]: Listening via Callbacks. The main execution thread is completely FREE.")
        print("[System]: Main thread will now sleep in a simple heartbeat loop.")
        print("[System]: Press Ctrl+C to stop.\n")

        heartbeat_count = 0
        while True:
            heartbeat_count += 1
            print(f"[Main App Loop Heartbeat #{heartbeat_count}] Thread active, doing other work...")
            time.sleep(3.0)

    except KeyboardInterrupt:
        print("\n[System]: Keyboard interrupt detected. Shutting down cleanly...")
    finally:
        # 4. Clean up resources and unregister callbacks
        vision_layer.remove_callback(safety_monitoring_callback)
        vision_layer.stop()


if __name__ == "__main__":
    main()
