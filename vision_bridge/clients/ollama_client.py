"""
vision_bridge.clients.ollama_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements the VLMClient interface for local Ollama-hosted vision models 
(e.g., Llama 3.2 Vision).
"""

from vision_bridge.clients.base import VLMClient
from vision_bridge.types import VLMClientError


class OllamaVLMClient(VLMClient):
    """
    Client for interacting with local Vision-Language Models served by Ollama.
    
    Ensures that local VLM pipelines are completely self-contained and run 
    offline on local hardware.
    """

    def __init__(self, model_name: str = "llama3.2-vision:1b"):
        """
        Initializes the Ollama vision client.
        
        Args:
            model_name: The name of the model installed in Ollama. Defaults 
                        to Llama 3.2 Vision (1.1B parameters).
        """
        self.model_name = model_name

    def analyze_image(self, b64_image: str, prompt: str) -> str:
        """
        Sends the base64 encoded image to the locally running Ollama instance.
        
        Args:
            b64_image: Base64-encoded image bytes.
            prompt: Text prompt filter/query.
            
        Returns:
            The raw text response from the local model.
        """
        try:
            import ollama
        except ImportError as e:
            raise VLMClientError(
                "The 'ollama' Python library is required. Please install it using: "
                "pip install ollama"
            ) from e

        try:
            # We communicate with the local Ollama daemon.
            # The images field accepts a list of base64 encoded image strings.
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [b64_image]
                }]
            )
            
            # Extract and clean up the result
            if 'message' in response and 'content' in response['message']:
                return response['message']['content'].strip()
            
            raise VLMClientError("Malformed response structure received from Ollama API.")
            
        except Exception as e:
            raise VLMClientError(f"Ollama local inference failed: {e}") from e
