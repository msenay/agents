#!/usr/bin/env python3
"""
🎯 New AgentConfig System Demo
Smart, controlled configuration with intelligent validation

This example shows how the new AgentConfig system prevents impossible configurations
and provides user-friendly error messages.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import AgentConfig, CoreAgent


def test_memory_disabled():
    """Test 1: Memory disabled - should block memory features"""
    print("🧪 TEST 1: Memory Disabled - Blocking Invalid Configs")
    print("=" * 60)
    
    # ✅ Valid: Memory disabled, no memory features
    try:
        config = AgentConfig(
            name="SimpleAgent",
            enable_memory=False  # Memory disabled
        )
        print(f"✅ Valid config: {config.name}, memory={config.enable_memory}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # ❌ Invalid: Memory disabled but trying to use summarization
    try:
        config = AgentConfig(
            name="InvalidAgent",
            enable_memory=False,
            enable_summarization=True  # Should fail!
        )
        print(f"❌ This should not work!")
    except ValueError as e:
        print(f"✅ Correctly blocked: {e}")
    
    # ❌ Invalid: Memory disabled but trying to use session_id
    try:
        config = AgentConfig(
            name="InvalidAgent",
            enable_memory=False,
            session_id="test_session"  # Should fail!
        )
        print(f"❌ This should not work!")
    except ValueError as e:
        print(f"✅ Correctly blocked: {e}")
    
    print()


def test_memory_enabled_smart_validation():
    """Test 2: Memory enabled with smart backend validation"""
    print("🧪 TEST 2: Memory Enabled - Smart Backend Validation")
    print("=" * 60)
    
    # ✅ Valid: InMemory backend
    try:
        config = AgentConfig(
            name="InMemoryAgent",
            enable_memory=True,
            memory_types=["short_term", "long_term"],
            memory_backend="inmemory"  # No URL needed
        )
        print(f"✅ InMemory backend: {config.memory_backend}")
        print(f"   Short-term: {config.enable_short_term_memory}")
        print(f"   Long-term: {config.enable_long_term_memory}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # ❌ Invalid: Redis backend without URL
    try:
        config = AgentConfig(
            name="InvalidRedis",
            enable_memory=True,
            memory_backend="redis"  # No redis_url provided!
        )
        print(f"❌ This should not work!")
    except ValueError as e:
        print(f"✅ Correctly blocked: {e}")
    
    # ✅ Valid: Redis backend with URL
    try:
        config = AgentConfig(
            name="RedisAgent",
            enable_memory=True,
            memory_types=["short_term", "session"],
            memory_backend="redis",
            redis_url="redis://localhost:6379"
        )
        print(f"✅ Redis backend: {config.memory_backend}")
        print(f"   Session memory: {config.enable_shared_memory}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print()


def test_ttl_validation():
    """Test 3: TTL validation - only for compatible backends"""
    print("🧪 TEST 3: TTL Validation - Backend Compatibility")
    print("=" * 60)
    
    # ❌ Invalid: TTL with InMemory backend
    try:
        config = AgentConfig(
            name="InvalidTTL",
            enable_memory=True,
            memory_backend="inmemory",
            enable_ttl=True  # TTL not supported on inmemory!
        )
        print(f"❌ This should not work!")
    except ValueError as e:
        print(f"✅ Correctly blocked: {e}")
    
    # ✅ Valid: TTL with Redis backend
    try:
        config = AgentConfig(
            name="RedisTTL",
            enable_memory=True,
            memory_backend="redis",
            redis_url="redis://localhost:6379",
            enable_ttl=True,
            default_ttl_minutes=60
        )
        print(f"✅ TTL with Redis: {config.enable_ttl}, TTL={config.default_ttl_minutes}min")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print()


def test_memory_tools_validation():
    """Test 4: Memory tools require long-term memory"""
    print("🧪 TEST 4: Memory Tools - Requires Long-term Memory")
    print("=" * 60)
    
    # ❌ Invalid: Memory tools without long-term memory
    try:
        config = AgentConfig(
            name="InvalidTools",
            enable_memory=True,
            memory_types=["short_term"],  # No long_term!
            enable_memory_tools=True
        )
        print(f"❌ This should not work!")
    except ValueError as e:
        print(f"✅ Correctly blocked: {e}")
    
    # ✅ Valid: Memory tools with long-term memory
    try:
        config = AgentConfig(
            name="ToolsAgent",
            enable_memory=True,
            memory_types=["short_term", "long_term"],
            enable_memory_tools=True
        )
        print(f"✅ Memory tools enabled: {config.enable_memory_tools}")
        print(f"   Long-term memory: {config.enable_long_term_memory}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print()


def test_multi_agent_validation():
    """Test 5: Multi-agent pattern conflicts"""
    print("🧪 TEST 5: Multi-agent Pattern - Conflict Detection")
    print("=" * 60)
    
    # ❌ Invalid: Multiple patterns enabled
    try:
        config = AgentConfig(
            name="ConflictAgent",
            enable_supervisor=True,
            enable_swarm=True  # Conflict!
        )
        print(f"❌ This should not work!")
    except ValueError as e:
        print(f"✅ Correctly blocked: {e}")
    
    # ✅ Valid: Single pattern
    try:
        config = AgentConfig(
            name="SupervisorAgent",
            enable_supervisor=True
        )
        print(f"✅ Supervisor pattern: {config.enable_supervisor}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # ❌ Invalid: Handoff without agents
    try:
        config = AgentConfig(
            name="EmptyHandoff",
            enable_handoff=True,
            handoff_agents=[]  # Empty list!
        )
        print(f"❌ This should not work!")
    except ValueError as e:
        print(f"✅ Correctly blocked: {e}")
    
    print()


def test_comprehensive_valid_config():
    """Test 6: Comprehensive valid configuration"""
    print("🧪 TEST 6: Comprehensive Valid Configuration")
    print("=" * 60)
    
    try:
        config = AgentConfig(
            name="ComprehensiveAgent",
            
            # Memory system
            enable_memory=True,
            memory_types=["short_term", "long_term", "session", "semantic"],
            memory_backend="redis",
            redis_url="redis://localhost:6379",
            session_id="demo_session",
            
            # TTL (compatible with Redis)
            enable_ttl=True,
            default_ttl_minutes=120,
            
            # Context management
            enable_message_trimming=True,
            max_tokens=8000,
            
            # AI features
            enable_summarization=True,
            enable_memory_tools=True,
            
            # Performance
            enable_rate_limiting=True,
            requests_per_second=2.0,
            
            # Multi-agent
            enable_supervisor=True
        )
        
        print(f"✅ Comprehensive agent created: {config.name}")
        print(f"   Memory backend: {config.memory_backend}")
        print(f"   Memory types: {config.memory_types}")
        print(f"   Short-term: {config.enable_short_term_memory}")
        print(f"   Long-term: {config.enable_long_term_memory}")
        print(f"   Session: {config.enable_shared_memory}")
        print(f"   Semantic: {config.enable_semantic_search}")
        print(f"   TTL: {config.enable_ttl}")
        print(f"   Memory tools: {config.enable_memory_tools}")
        print(f"   Rate limiting: {config.enable_rate_limiting}")
        print(f"   Supervisor: {config.enable_supervisor}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print()


def test_backward_compatibility():
    """Test 7: Backward compatibility properties"""
    print("🧪 TEST 7: Backward Compatibility")
    print("=" * 60)
    
    try:
        config = AgentConfig(
            name="BackwardCompatAgent",
            enable_memory=True,
            memory_types=["short_term", "long_term"],
            memory_backend="postgres",
            postgres_url="postgresql://user:pass@localhost:5432/db"
        )
        
        # Test backward compatibility properties
        print(f"✅ Backward compatibility test:")
        print(f"   enable_short_term_memory: {config.enable_short_term_memory}")
        print(f"   enable_long_term_memory: {config.enable_long_term_memory}")
        print(f"   short_term_memory_type: {config.short_term_memory_type}")
        print(f"   long_term_memory_type: {config.long_term_memory_type}")
        print(f"   memory_type: {config.memory_type}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


if __name__ == "__main__":
    print("🎯 NEW AGENTCONFIG SYSTEM DEMO")
    print("Smart validation and user-friendly error messages")
    print("=" * 70)
    print()
    
    test_memory_disabled()
    test_memory_enabled_smart_validation()
    test_ttl_validation()
    test_memory_tools_validation()
    test_multi_agent_validation()
    test_comprehensive_valid_config()
    test_backward_compatibility()
    
    print("🎉 NEW CONFIG SYSTEM DEMO COMPLETED!")
    print()
    print("💡 Key Benefits:")
    print("  ✅ Prevents impossible configurations")
    print("  ✅ Clear, helpful error messages")
    print("  ✅ Backend-specific validation")
    print("  ✅ Memory type organization")
    print("  ✅ Backward compatibility preserved")
    print("  ✅ User-friendly configuration flow")