#!/usr/bin/env python3
"""
🎯 Pre-Built Configurations Demo
===============================

This demo shows how to use pre-built AgentConfig presets to quickly create
optimized agents for different scenarios.

Usage:
    python examples/pre_built_configs_demo.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pre_built_configs import *
from core_agent import CoreAgent


class MockChatModel:
    """Mock model for demo purposes"""
    def invoke(self, messages):
        return {"content": "This is a mock response for demo purposes."}


def demo_high_performance_config():
    """Demo: High Performance Production Configuration"""
    print("🚀 HIGH PERFORMANCE PRODUCTION CONFIG")
    print("=" * 50)
    
    # Get the config
    config = HIGH_PERFORMANCE_CONFIG
    config.model = MockChatModel()  # Add mock model for demo
    
    print(f"Agent Name: {config.name}")
    print(f"Memory Backend: {config.memory_backend}")
    print(f"Memory Types: {config.memory_types}")
    print(f"Rate Limiting: {config.requests_per_second}/s")
    print(f"TTL Enabled: {config.enable_ttl}")
    print(f"Semantic Search: {config.enable_semantic_search}")
    print(f"Memory Tools: {config.enable_memory_tools}")
    print(f"Summarization: {config.enable_summarization}")
    
    # Create agent
    try:
        agent = CoreAgent(config)
        print("✅ Agent created successfully!")
        print(f"   Memory Manager: {type(agent.memory_manager).__name__}")
        print(f"   Has Tools: {len(agent.get_tools()) > 0}")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_development_config():
    """Demo: Development Configuration"""
    print("🛠️ DEVELOPMENT CONFIG")
    print("=" * 50)
    
    # Get and customize the config
    config = get_config("development")
    config.model = MockChatModel()
    config.name = "MyDevAgent"  # Customize name
    
    print(f"Agent Name: {config.name}")
    print(f"Memory Backend: {config.memory_backend}")
    print(f"Memory Types: {config.memory_types}")
    print(f"Rate Limiting: {config.requests_per_second}/s")
    print(f"Session ID: {config.session_id}")
    
    # Create agent
    try:
        agent = CoreAgent(config)
        print("✅ Agent created successfully!")
        
        # Test conversation
        response = agent.invoke("Hello, test message")
        print(f"   Test response: {response['messages'][-1].content[:50]}...")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_customer_service_config():
    """Demo: Customer Service Configuration"""
    print("👥 CUSTOMER SERVICE CONFIG")
    print("=" * 50)
    
    config = CUSTOMER_SERVICE_CONFIG
    config.model = MockChatModel()
    
    print(f"Agent Name: {config.name}")
    print(f"Memory Backend: {config.memory_backend}")
    print(f"Memory Types: {config.memory_types}")
    print(f"Semantic Search: {config.enable_semantic_search}")
    print(f"Human Feedback: {config.enable_human_feedback}")
    print(f"Interrupts Before: {config.interrupt_before}")
    
    try:
        agent = CoreAgent(config)
        print("✅ Customer service agent ready!")
        print(f"   Memory namespace: {config.memory_namespace}")
        print(f"   Evaluation metrics: {config.evaluation_metrics}")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_custom_config():
    """Demo: Creating Custom Configuration"""
    print("🎨 CUSTOM CONFIG (Based on Development)")
    print("=" * 50)
    
    # Create custom config based on development preset
    custom_config = create_custom_config(
        name="MyCustomAgent",
        enable_rate_limiting=True,
        requests_per_second=5.0,
        max_tokens=4000,
        memory_namespace="custom_project",
        enable_evaluation=False  # Disable evaluation
    )
    custom_config.model = MockChatModel()
    
    print(f"Agent Name: {custom_config.name}")
    print(f"Memory Backend: {custom_config.memory_backend}")
    print(f"Rate Limiting: {custom_config.requests_per_second}/s")
    print(f"Max Tokens: {custom_config.max_tokens}")
    print(f"Memory Namespace: {custom_config.memory_namespace}")
    print(f"Evaluation: {custom_config.enable_evaluation}")
    
    try:
        agent = CoreAgent(custom_config)
        print("✅ Custom agent created successfully!")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_config_comparison():
    """Demo: Compare all configurations"""
    print("📊 CONFIGURATION COMPARISON")
    print("=" * 50)
    
    print_config_comparison()
    print()


def demo_config_listing():
    """Demo: List all available configs"""
    print("📋 AVAILABLE CONFIGURATIONS")
    print("=" * 50)
    
    configs = list_configs()
    for name, description in configs.items():
        print(f"🔹 {name:<20} {description}")
    
    print()


def demo_real_world_scenarios():
    """Demo: Real-world usage scenarios"""
    print("🌍 REAL-WORLD USAGE SCENARIOS")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "E-commerce Customer Support",
            "config": "customer_service",
            "description": "Handle customer inquiries with session continuity",
            "features": ["Session memory", "Knowledge base search", "Human handoff"]
        },
        {
            "name": "Research Paper Analysis",
            "config": "research_assistant", 
            "description": "Analyze academic papers and build knowledge",
            "features": ["PostgreSQL storage", "Semantic search", "Large context"]
        },
        {
            "name": "Content Creation Tool",
            "config": "creative_assistant",
            "description": "Generate creative content with inspiration tracking",
            "features": ["Fast iteration", "Creative evaluation", "No rate limits"]
        },
        {
            "name": "Enterprise AI Assistant",
            "config": "enterprise",
            "description": "Corporate AI with compliance and audit trails",
            "features": ["Human oversight", "Audit trails", "Data governance"]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Config: {scenario['config']}")
        print(f"   Use case: {scenario['description']}")
        print(f"   Key features: {', '.join(scenario['features'])}")
        print()


def demo_config_selection_guide():
    """Demo: Guide for selecting the right config"""
    print("🎯 CONFIGURATION SELECTION GUIDE")
    print("=" * 50)
    
    guide = [
        {
            "question": "What's your deployment environment?",
            "options": {
                "Local development": "development",
                "Production server": "high_performance", 
                "Microservices/Cloud": "distributed",
                "Enterprise environment": "enterprise"
            }
        },
        {
            "question": "What's your primary use case?",
            "options": {
                "Customer support": "customer_service",
                "Research & analysis": "research_assistant",
                "Creative content": "creative_assistant",
                "Automated testing": "testing"
            }
        },
        {
            "question": "What are your performance requirements?",
            "options": {
                "High throughput": "high_performance",
                "Fast development": "development",
                "Compliance & audit": "enterprise",
                "Distributed scaling": "distributed"
            }
        }
    ]
    
    for guide_item in guide:
        print(f"❓ {guide_item['question']}")
        for option, config in guide_item['options'].items():
            print(f"   • {option} → {config}")
        print()


if __name__ == "__main__":
    print("🎯 PRE-BUILT AGENT CONFIGURATIONS DEMO")
    print("=====================================")
    print()
    
    # Run all demos
    demo_config_comparison()
    demo_config_listing()
    demo_high_performance_config()
    demo_development_config()
    demo_customer_service_config()
    demo_custom_config()
    demo_real_world_scenarios()
    demo_config_selection_guide()
    
    print("🎉 DEMO COMPLETED!")
    print()
    print("💡 Next Steps:")
    print("   1. Choose a pre-built config that matches your use case")
    print("   2. Import it: from pre_built_configs import HIGH_PERFORMANCE_CONFIG")
    print("   3. Add your model: config.model = ChatOpenAI()")
    print("   4. Create agent: agent = CoreAgent(config)")
    print("   5. Customize as needed!")