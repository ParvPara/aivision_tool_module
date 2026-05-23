"""
tests.test_clients
~~~~~~~~~~~~~~~~~~

Unit tests for VLM client creation factory and mock client simulated behavior.
"""

import unittest
import sys
import os

# Ensure parent package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision_bridge.clients import get_vlm_client, MockVLMClient
from vision_bridge.types import VLMClientError


class TestVLMClientSuite(unittest.TestCase):
    """Verifies client construction logic and simulated VLM interactions."""

    def test_factory_invalid_provider(self):
        """Verifies that the factory raises ValueError on unknown client types."""
        with self.assertRaises(ValueError):
            get_vlm_client("unknown_provider")

    def test_factory_mock_client_creation(self):
        """Verifies that the factory successfully loads a mock provider client."""
        client = get_vlm_client("mock")
        self.assertIsInstance(client, MockVLMClient)

    def test_mock_client_response_cycle(self):
        """Verifies that the mock client cycles through its simulated text library."""
        custom_responses = ["A", "B", "C"]
        client = MockVLMClient(responses=custom_responses, simulated_latency=0.0)
        
        # Test cycle
        self.assertEqual(client.analyze_image("dummy_b64", "What do you see?"), "A")
        self.assertEqual(client.analyze_image("dummy_b64", "What do you see?"), "B")
        self.assertEqual(client.analyze_image("dummy_b64", "What do you see?"), "C")
        # Should wrap around
        self.assertEqual(client.analyze_image("dummy_b64", "What do you see?"), "A")

    def test_mock_client_prompt_awareness(self):
        """Verifies that the mock client matches filter prompt patterns."""
        client = MockVLMClient(simulated_latency=0.0)
        
        # Ask filter question involving 'none'
        response_1 = client.analyze_image("dummy_b64", "Look for objects. Reply 'None' otherwise.")
        # Mock client should match none instruction and cycle options
        self.assertIn(response_1, ["Technician holding a socket wrench", "None", "Person standing at desk", "Hand tool calibration completed"])


if __name__ == "__main__":
    unittest.main()
