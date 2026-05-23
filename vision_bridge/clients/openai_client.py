"""
vision_bridge.clients.openai_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements the VLMClient interface for cloud-based OpenAI models (e.g., GPT-4o, 
GPT-4o-mini) using the official `openai` Python client library.
"""

import os
from typing import Optional
from vision_bridge.clients.base import VLMClient
from vision_bridge.types import VLMClientError


class OpenAIVLMClient(VLMClient):
    """
    Client for interacting with cloud-based OpenAI GPT Vision models.
    
    Expects `openai` package to be installed and `OPENAI_API_KEY` to be configured.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initializes the OpenAI vision client.
        
        Args:
            model_name: OpenAI model to use. Defaults to "gpt-4o-mini" (highly 
                        cost-effective and fast).
            api_key: Optional explicit API key. If omitted, falls back to 
                     the `OPENAI_API_KEY` environment variable.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def analyze_image(self, b64_image: str, prompt: str) -> str:
        """
        Sends the base64 encoded image to OpenAI's completion API.
        
        Args:
            b64_image: Base64-encoded image bytes.
            prompt: Text instructions for the model.
            
        Returns:
            The raw text response from GPT.
        """
        try:
            import openai
        except ImportError as e:
            raise VLMClientError(
                "The 'openai' Python library is required. Please install it using: "
                "pip install openai"
            ) from e

        if not self.api_key:
            raise VLMClientError(
                "No API key provided. Please pass an api_key to the initializer or "
                "set the 'OPENAI_API_KEY' environment variable."
            )

        try:
            # Instantiate client (uses environment API key, or explicit key)
            client = openai.OpenAI(api_key=self.api_key)
            
            # Format using OpenAI's system message structure for Vision
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150  # Keep descriptions tight for vision triggers
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            
            raise VLMClientError("Empty response returned from OpenAI GPT model.")

        except Exception as e:
            raise VLMClientError(f"OpenAI API cloud inference failed: {e}") from e
