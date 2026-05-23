"""
tests.test_core
~~~~~~~~~~~~~~~

Unit tests for the main VisionBridgeModule orchestrator, validating its generator 
iteration streams and callback notification systems.
"""

import unittest
import time
import sys
import os

# Ensure parent package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision_bridge import VisionBridgeModule, VisionEvent


class TestCoreBridgeSuite(unittest.TestCase):
    """Verifies dual pattern event delivery and lifecycle thread cleanup."""

    def test_iterator_pattern_generator(self):
        """Tests that get_events() yields valid VisionEvent objects correctly."""
        # 1. Initialize orchestrator (high frame rate 5 FPS for rapid testing)
        bridge = VisionBridgeModule(
            vlm_model="mock",
            camera_index="mock",
            frame_rate=5.0,
            simulated_latency=0.01
        )
        
        # 2. Start core
        bridge.start(prompt="Describe.")

        captured_events = []
        try:
            # 3. Pull first 3 events via the Iterator Generator
            for event in bridge.get_events():
                captured_events.append(event)
                if len(captured_events) >= 3:
                    break
        finally:
            # 4. Stop core cleanly
            bridge.stop()
            
        # 5. Asset check
        self.assertEqual(len(captured_events), 3)
        for event in captured_events:
            self.assertIsInstance(event, VisionEvent)
            self.assertGreater(event.frame_id, 0)
            self.assertIsNotNone(event.raw_text)
            self.assertGreaterEqual(event.response_time_ms, 0.0)
            self.assertEqual(event.metadata["camera_index"], "mock")
            self.assertEqual(event.metadata["provider"], "MockVLMClient")

    def test_observer_pattern_callbacks(self):
        """Tests that registered callbacks receive visual updates asynchronously."""
        bridge = VisionBridgeModule(
            vlm_model="mock",
            camera_index="mock",
            frame_rate=5.0,
            simulated_latency=0.01
        )

        callback_events = []

        def custom_listener(event: VisionEvent) -> None:
            callback_events.append(event)

        # 1. Register observer callback
        bridge.add_callback(custom_listener)
        
        # 2. Start
        bridge.start(prompt="Observe tools.")

        try:
            # Wait a brief moment to allow dispatcher thread to process events
            time.sleep(0.8)
        finally:
            # 3. Shutdown and unregister
            bridge.stop()
            bridge.remove_callback(custom_listener)

        # 4. Assertions
        self.assertGreater(len(callback_events), 0)
        for event in callback_events:
            self.assertIsInstance(event, VisionEvent)
            self.assertGreater(event.frame_id, 0)
            self.assertIsNotNone(event.raw_text)


if __name__ == "__main__":
    unittest.main()
