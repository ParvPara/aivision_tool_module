"""
vision_bridge.clients.mock_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides a local simulator client that implements the VLMClient interface. 
It requires no external APIs or local hardware, allowing developer testing, 
CI automation, and mock dry-runs.
"""

import time
from typing import List, Optional
from vision_bridge.clients.base import VLMClient


class MockVLMClient(VLMClient):
    """
    A simulated VLM Client that mimics api call delays and returns realistic 
    text responses. Cycle through actions based on prompts.
    """

    def __init__(self, responses: Optional[List[str]] = None, simulated_latency: float = 0.3):
        """
        Initializes the Mock VLM Client.
        
        Args:
            responses: A list of string responses to cycle through. If None, 
                       uses a high-quality list of industrial/office descriptions.
            simulated_latency: Sleep delay in seconds to mimic web API overhead.
        """
        self.simulated_latency = simulated_latency
        self.responses = responses or [
            "A technician wearing safety glasses enters the scene.",
            "Idle",
            "A blue socket wrench is picked up from the tool tray.",
            "Idle",
            "A worker waving at the security camera.",
            "A laptop displaying an open terminal on the desk.",
            "Idle"
        ]
        self._index = 0

    def analyze_image(self, b64_image: str, prompt: str) -> str:
        """
        Simulates an API request, sleeping for the configured latency and 
        returning a mock visual update.
        
        Args:
            b64_image: Unused base64 string.
            prompt: Text prompt (influences mock selection to maintain consistency).
            
        Returns:
            A simulated text response.
        """
        if self.simulated_latency > 0:
            time.sleep(self.simulated_latency)

        prompt_lower = prompt.lower()
        
        # If the prompt explicitly asks for "None" if nothing happens (e.g. filter prompts)
        if "none" in prompt_lower:
            choices = [
                "Technician holding a socket wrench",
                "None",
                "Person standing at desk",
                "None",
                "Hand tool calibration completed",
                "None"
            ]
            response = choices[self._index % len(choices)]
            self._index += 1
            return response

        # General context prompt descriptions
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        return response
