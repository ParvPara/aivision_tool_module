"""
vision_bridge.clients.base
~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines the abstract interface for all VLM integration providers. All specific 
vendors (Ollama, OpenAI, Gemini, Mock) must implement this interface to be 
interchangeable within the Vision Bridge Module.
"""

from abc import ABC, abstractmethod


class VLMClient(ABC):
    """
    Abstract Base Class defining the contract for a Vision-Language Model client.
    
    Any client wrapper (Ollama, OpenAI, Gemini, etc.) must subclass this and 
    implement the `analyze_image` method to ensure seamless drop-in swapability.
    """

    @abstractmethod
    def analyze_image(self, b64_image: str, prompt: str) -> str:
        """
        Sends an image (as a Base64-encoded JPEG/PNG string) and a text prompt 
        to the target Vision-Language Model.
        
        Args:
            b64_image: The image encoded as a base64 string (without data URI prefix).
            prompt: The query or instruction for the VLM (e.g., 'Describe the scene').
            
        Returns:
            The raw text response from the model.
            
        Raises:
            VLMClientError: If there's an API error, transport error, or validation failure.
        """
        pass
