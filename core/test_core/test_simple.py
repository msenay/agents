#!/usr/bin/env python3
"""
Simple Core Agent Test
====================

Basic functionality test without mocking dependencies.
This tests the actual imports and basic functionality.
"""

import sys
import os

# Add workspace directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_imports():
    """Test that all core modules can be imported"""
    print("Testing imports...")
    
    try:
        from core.config import AgentConfig
        print("✓ AgentConfig imported successfully")
    except Exception as e:
        print(f"✗ AgentConfig import failed: {e}")
        return False
    
    try:
        from core.model import CoreAgentState
        print("✓ CoreAgentState imported successfully")
    except Exception as e:
        print(f"✗ CoreAgentState import failed: {e}")
        return False
    
    try:
        from core.managers import MemoryManager, RateLimiterManager
        print("✓ Managers imported successfully")
    except Exception as e:
        print(f"✗ Managers import failed: {e}")
        return False
    
    try:
        from core.core_agent import CoreAgent
        print("✓ CoreAgent imported successfully")
    except Exception as e:
        print(f"✗ CoreAgent import failed: {e}")
        return False
    
    return True

def test_basic_config():
    """Test basic AgentConfig creation"""
    print("\nTesting AgentConfig...")
    
    try:
        from core.config import AgentConfig
        
        # Test minimal config
        config = AgentConfig()
        print(f"✓ Minimal config created: {config.name}")
        
        # Test config with memory
        config_memory = AgentConfig(
            name="TestAgent",
            enable_memory=True,
            memory_types=["short_term"]
        )
        print(f"✓ Memory config created: {config_memory.name}")
        
        return True
    except Exception as e:
        print(f"✗ AgentConfig test failed: {e}")
        return False

def test_basic_agent():
    """Test basic CoreAgent creation"""
    print("\nTesting CoreAgent...")
    
    try:
        from core.config import AgentConfig
        from core.core_agent import CoreAgent
        
        # Test minimal agent
        config = AgentConfig(name="SimpleAgent")
        agent = CoreAgent(config)
        print(f"✓ Basic agent created: {agent.config.name}")
        
        # Test agent status
        status = agent.get_status()
        print(f"✓ Agent status retrieved: {status['name']}")
        
        return True
    except Exception as e:
        print(f"✗ CoreAgent test failed: {e}")
        return False

def test_managers():
    """Test manager classes"""
    print("\nTesting Managers...")
    
    try:
        from core.config import AgentConfig
        from core.managers import MemoryManager, RateLimiterManager
        
        config = AgentConfig()
        
        # Test MemoryManager
        memory_manager = MemoryManager(config)
        print("✓ MemoryManager created")
        
        # Test RateLimiterManager
        rate_manager = RateLimiterManager(config)
        print("✓ RateLimiterManager created")
        
        return True
    except Exception as e:
        print(f"✗ Managers test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Core Agent Simple Test ===\n")
    
    tests = [
        test_imports,
        test_basic_config,
        test_basic_agent,
        test_managers
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("\nTest failed! Stopping...")
            break
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)