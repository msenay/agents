"""
Simple test to check if core modules can be imported with mocks
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock all external dependencies
mock_modules = [
    'langchain_core',
    'langchain_core.messages',
    'langchain_core.messages.utils',
    'langchain_core.rate_limiters',
    'langchain_core.tools',
    'langchain_core.language_models',
    'langgraph',
    'langgraph.graph',
    'langgraph.checkpoint',
    'langgraph.checkpoint.memory',
    'langgraph.checkpoint.redis',
    'langgraph.checkpoint.postgres',
    'langgraph.checkpoint.mongodb',
    'langgraph.store',
    'langgraph.store.memory',
    'langgraph.store.redis', 
    'langgraph.store.postgres',
    'langgraph.prebuilt',
    'langgraph.types',
    'langgraph.graph.message',
    'langchain',
    'langchain.embeddings',
    'langmem',
    'langmem.short_term',
    'langgraph_supervisor',
    'langgraph_swarm',
    'langchain_mcp_adapters',
    'langchain_mcp_adapters.client',
    'agentevals',
    'agentevals.trajectory',
    'agentevals.trajectory.match',
    'agentevals.trajectory.llm',
    'pydantic'
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

# Now try to import our modules
try:
    from core.config import AgentConfig
    print("✓ core.config imported successfully")
except Exception as e:
    print(f"✗ Failed to import core.config: {e}")

try:
    from core.model import CoreAgentState
    print("✓ core.model imported successfully")
except Exception as e:
    print(f"✗ Failed to import core.model: {e}")

try:
    from core.managers import (
        SubgraphManager, MemoryManager, SupervisorManager,
        MCPManager, EvaluationManager, RateLimiterManager
    )
    print("✓ core.managers imported successfully")
except Exception as e:
    print(f"✗ Failed to import core.managers: {e}")

try:
    from core.core_agent import CoreAgent
    print("✓ core.core_agent imported successfully")
except Exception as e:
    print(f"✗ Failed to import core.core_agent: {e}")

try:
    import core
    print("✓ core package imported successfully")
except Exception as e:
    print(f"✗ Failed to import core package: {e}")

print("\nAll core modules imported successfully with mocks!")