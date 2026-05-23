"""
tests.test_camera
~~~~~~~~~~~~~~~~~

Unit tests for thread-safe CameraCapture operations using the mock test pattern stream.
"""

import unittest
import time
import sys
import os

# Ensure parent package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision_bridge.camera import CameraCapture


class TestCameraSuite(unittest.TestCase):
    """Verifies that the camera reader can start, capture frames, resize, and close cleanly."""

    def test_mock_camera_stream(self):
        """Tests that the mock test pattern camera thread starts and generates valid image objects."""
        # 1. Setup mock camera capture thread
        res = (320, 240)
        camera = CameraCapture(camera_index="mock", resolution=res)
        
        # Must start with empty state
        frame, frame_id = camera.get_latest_frame()
        self.assertIsNone(frame)
        self.assertEqual(frame_id, 0)
        
        # 2. Run the thread
        camera.start()
        
        # Wait a brief moment for the thread to generate at least one frame
        time.sleep(0.3)
        
        # 3. Retrieve results
        frame, frame_id = camera.get_latest_frame()
        
        try:
            self.assertIsNotNone(frame)
            self.assertGreater(frame_id, 0)
            
            # Verify shape: frame should be height x width x channels
            self.assertEqual(frame.shape[0], res[1]) # Height
            self.assertEqual(frame.shape[1], res[0]) # Width
            self.assertEqual(frame.shape[2], 3)      # BGR Channels
            
        finally:
            # 4. Stop thread
            camera.stop()
            self.assertFalse(camera.is_alive())


if __name__ == "__main__":
    unittest.main()
