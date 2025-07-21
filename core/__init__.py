"""
Core Agent Framework

A comprehensive agent framework with optional features for building sophisticated LangGraph agents.
"""

from core.config import AgentConfig
from core.model import CoreAgentState
from core.core_agent import CoreAgent
from core.managers import (
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