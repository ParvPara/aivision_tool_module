"""
vision_bridge.clients
~~~~~~~~~~~~~~~~~~~~~

Exposes interchangeable VLM client providers and a unified builder factory.
"""

from typing import Optional
from vision_bridge.clients.base import VLMClient
from vision_bridge.clients.ollama_client import OllamaVLMClient
from vision_bridge.clients.openai_client import OpenAIVLMClient
from vision_bridge.clients.gemini_client import GeminiVLMClient
from vision_bridge.clients.mock_client import MockVLMClient


def get_vlm_client(provider: str, model_name: Optional[str] = None, **kwargs) -> VLMClient:
    """
    Unified factory method to instantiate a VLMClient by provider name.
    
    Args:
        provider: Name of the VLM provider ('ollama', 'openai', 'gemini', 'mock').
        model_name: Optional target model string (e.g., 'gpt-4o-mini', 'llama3.2-vision:1b').
        **kwargs: Additional parameters passed directly to the client initializer 
                 (e.g., api_key, simulated_latency, responses).
                 
    Returns:
        An instance of VLMClient.
        
    Raises:
        ValueError: If an unsupported provider name is specified.
    """
    prov_clean = provider.strip().lower()
    
    if prov_clean == "ollama":
        return OllamaVLMClient(model_name=model_name or "llama3.2-vision:1b")
    elif prov_clean == "openai":
        return OpenAIVLMClient(model_name=model_name or "gpt-4o-mini", **kwargs)
    elif prov_clean == "gemini":
        return GeminiVLMClient(model_name=model_name or "gemini-1.5-flash", **kwargs)
    elif prov_clean == "mock":
        return MockVLMClient(**kwargs)
    else:
        raise ValueError(
            f"Unsupported VLM provider '{provider}'. "
            "Supported values are: 'ollama', 'openai', 'gemini', 'mock'."
        )


__all__ = [
    "VLMClient",
    "OllamaVLMClient",
    "OpenAIVLMClient",
    "GeminiVLMClient",
    "MockVLMClient",
    "get_vlm_client"
]
