"""
Core Agent Framework

A comprehensive agent framework with optional features for building sophisticated LangGraph agents.
"""

from ai_factory.agents.core.config import AgentConfig
from ai_factory.agents.core.model import CoreAgentState
from ai_factory.agents.core.core_agent import CoreAgent
from ai_factory.agents.core.managers import (
    SubgraphManager,
    MemoryManager,
    SupervisorManager,
    MCPManager,
    EvaluationManager,
    RateLimiterManager
)

__all__ = [
    "AgentConfig",
    "CoreAgentState", 
    "CoreAgent",
    "SubgraphManager",
    "MemoryManager",
    "SupervisorManager",
    "MCPManager",
    "EvaluationManager",
    "RateLimiterManager"
]

__version__ = "1.0.0"