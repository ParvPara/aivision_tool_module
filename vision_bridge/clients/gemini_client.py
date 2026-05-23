"""
vision_bridge.clients.gemini_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements the VLMClient interface for cloud-based Google Gemini models 
(e.g., gemini-1.5-flash) using the `google-generativeai` client library.
"""

import os
import base64
from typing import Optional
from vision_bridge.clients.base import VLMClient
from vision_bridge.types import VLMClientError


class GeminiVLMClient(VLMClient):
    """
    Client for interacting with Google's Gemini Vision models.
    
    Expects `google-generativeai` to be installed and `GEMINI_API_KEY` (or 
    `GOOGLE_API_KEY`) to be set in the system environment.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        """
        Initializes the Gemini vision client.
        
        Args:
            model_name: Gemini model to target. Defaults to "gemini-1.5-flash" 
                        (highly responsive and optimized for visual tasks).
            api_key: Optional explicit API key. If omitted, falls back to 
                     the `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variables.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def analyze_image(self, b64_image: str, prompt: str) -> str:
        """
        Sends the base64 encoded image to the Google Gemini model.
        
        Args:
            b64_image: Base64-encoded image bytes.
            prompt: Text instructions or question for Gemini.
            
        Returns:
            The raw text response from Gemini.
        """
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise VLMClientError(
                "The 'google-generativeai' library is required. Please install it using: "
                "pip install google-generativeai"
            ) from e

        if not self.api_key:
            raise VLMClientError(
                "No API key provided. Please pass an api_key to the initializer or "
                "set the 'GEMINI_API_KEY' or 'GOOGLE_API_KEY' environment variable."
            )

        try:
            # Configure API key
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            
            # Decode the base64 data to binary bytes for the Gemini SDK payload
            raw_bytes = base64.b64decode(b64_image)
            image_part = {
                "mime_type": "image/jpeg",
                "data": raw_bytes
            }
            
            # Gemini's generate_content accepts mixed modalities natively
            response = model.generate_content([prompt, image_part])
            
            if response.text:
                return response.text.strip()
            
            raise VLMClientError("Empty response returned from Gemini model.")

        except Exception as e:
            raise VLMClientError(f"Gemini cloud inference failed: {e}") from e
