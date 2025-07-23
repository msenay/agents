"""
LLM Factory for creating different LLM instances

This factory provides a centralized way to create LLM instances for all agents.
Supports multiple providers: Azure OpenAI, OpenAI, Anthropic, etc.
"""

import os
from typing import Optional, Dict, Any, Literal
from enum import Enum

# Import LLM providers
try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class LLMProvider(Enum):
    """Supported LLM providers"""
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MOCK = "mock"  # For testing


class LLMFactory:
    """Factory for creating LLM instances"""
    
    # Default configurations for different use cases
    CONFIGS = {
        "coder": {
            "temperature": 0.1,  # Low for consistent code generation
            "max_tokens": 4000,
            "top_p": 0.95
        },
        "tester": {
            "temperature": 0.2,  # Slightly higher for creative test cases
            "max_tokens": 3000,
            "top_p": 0.9
        },
        "executor": {
            "temperature": 0.0,  # Deterministic for execution
            "max_tokens": 2000,
            "top_p": 1.0
        },
        "orchestrator": {
            "temperature": 0.3,  # Balanced for coordination
            "max_tokens": 4000,
            "top_p": 0.9
        },
        "default": {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.95
        }
    }
    
    @classmethod
    def create(
        cls,
        provider: LLMProvider = None,
        agent_type: Literal["coder", "tester", "executor", "orchestrator", "default"] = "default",
        **kwargs
    ):
        """
        Create an LLM instance
        
        Args:
            provider: LLM provider to use (auto-detects if not specified)
            agent_type: Type of agent to optimize settings for
            **kwargs: Additional parameters to override defaults
            
        Returns:
            LLM instance
        """
        # Auto-detect provider if not specified
        if provider is None:
            provider = cls._detect_provider()
            
        # Get default config for agent type
        config = cls.CONFIGS.get(agent_type, cls.CONFIGS["default"]).copy()
        
        # Override with any provided kwargs
        config.update(kwargs)
        
        # Create LLM based on provider
        if provider == LLMProvider.AZURE_OPENAI:
            return cls._create_azure_openai(**config)
        elif provider == LLMProvider.OPENAI:
            return cls._create_openai(**config)
        elif provider == LLMProvider.ANTHROPIC:
            return cls._create_anthropic(**config)
        elif provider == LLMProvider.GOOGLE:
            return cls._create_google(**config)
        elif provider == LLMProvider.MOCK:
            return cls._create_mock(**config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def _detect_provider() -> LLMProvider:
        """Auto-detect which provider to use based on environment variables"""
        if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("OPENAI_API_KEY"):
            return LLMProvider.AZURE_OPENAI
        elif os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_ENDPOINT"):
            return LLMProvider.OPENAI
        elif os.getenv("ANTHROPIC_API_KEY"):
            return LLMProvider.ANTHROPIC
        elif os.getenv("GOOGLE_API_KEY"):
            return LLMProvider.GOOGLE
        else:
            # Default to mock for testing
            return LLMProvider.MOCK
    
    @staticmethod
    def _create_azure_openai(**kwargs):
        """Create Azure OpenAI instance"""
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
            
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_version=os.getenv("OPENAI_API_VERSION", "2023-12-01-preview"),
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt4o"),
            **kwargs
        )
    
    @staticmethod
    def _create_openai(**kwargs):
        """Create OpenAI instance"""
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
            
        return ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=kwargs.pop("model_name", "gpt-4-turbo-preview"),
            **kwargs
        )
    
    @staticmethod
    def _create_anthropic(**kwargs):
        """Create Anthropic instance"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("langchain-anthropic not installed. Run: pip install langchain-anthropic")
            
        return ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name=kwargs.pop("model_name", "claude-3-opus-20240229"),
            **kwargs
        )
    
    @staticmethod
    def _create_google(**kwargs):
        """Create Google instance"""
        if not GOOGLE_AVAILABLE:
            raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
            
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model=kwargs.pop("model", "gemini-pro"),
            **kwargs
        )
    
    @staticmethod
    def _create_mock(**kwargs):
        """Create mock LLM for testing"""
        from unittest.mock import Mock
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Mock response"))
        mock_llm.stream = Mock(return_value=[Mock(content="Mock"), Mock(content=" response")])
        mock_llm.temperature = kwargs.get("temperature", 0.7)
        mock_llm.max_tokens = kwargs.get("max_tokens", 2000)
        
        return mock_llm
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Get list of available providers"""
        return {
            "azure_openai": OPENAI_AVAILABLE and bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
            "openai": OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": ANTHROPIC_AVAILABLE and bool(os.getenv("ANTHROPIC_API_KEY")),
            "google": GOOGLE_AVAILABLE and bool(os.getenv("GOOGLE_API_KEY")),
            "mock": True  # Always available
        }
    
    @classmethod
    def get_provider_info(cls) -> str:
        """Get information about available providers"""
        info = ["Available LLM Providers:"]
        providers = cls.get_available_providers()
        
        for provider, available in providers.items():
            status = "✅" if available else "❌"
            info.append(f"  {status} {provider}")
            
        return "\n".join(info)


# Convenience functions
def create_llm(agent_type: str = "default", **kwargs):
    """Convenience function to create an LLM"""
    return LLMFactory.create(agent_type=agent_type, **kwargs)


def get_coder_llm(**kwargs):
    """Get LLM optimized for code generation"""
    return LLMFactory.create(agent_type="coder", **kwargs)


def get_tester_llm(**kwargs):
    """Get LLM optimized for test generation"""
    return LLMFactory.create(agent_type="tester", **kwargs)


def get_executor_llm(**kwargs):
    """Get LLM optimized for code execution"""
    return LLMFactory.create(agent_type="executor", **kwargs)


def get_orchestrator_llm(**kwargs):
    """Get LLM optimized for orchestration"""
    return LLMFactory.create(agent_type="orchestrator", **kwargs)


if __name__ == "__main__":
    # Test the factory
    print(LLMFactory.get_provider_info())
    
    # Example usage
    try:
        llm = create_llm("coder")
        print(f"\nCreated LLM: {type(llm).__name__}")
        print(f"Temperature: {getattr(llm, 'temperature', 'N/A')}")
        print(f"Max tokens: {getattr(llm, 'max_tokens', 'N/A')}")
    except Exception as e:
        print(f"\nError creating LLM: {e}")