#!/usr/bin/env python3
"""
🧑‍💻 Software Development Workflow Demo
========================================

This demo shows how to use pre-built AgentConfig presets for software development workflow.
Includes configurations for Coder, Unit Tester, Executer, Code Reviewer, Build Agent, and Orchestrator.

Usage:
    python examples/software_dev_workflow_demo.py
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


def demo_coder_agent():
    """Demo: Coder Agent Configuration"""
    print("👨‍💻 CODER AGENT CONFIG")
    print("=" * 50)
    
    config = CODER_AGENT_CONFIG
    config.model = MockChatModel()
    
    print(f"Agent Name: {config.name}")
    print(f"Memory Backend: {config.memory_backend}")
    print(f"Memory Types: {config.memory_types}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Rate Limiting: {config.enable_rate_limiting}")
    print(f"Semantic Search: {config.enable_semantic_search}")
    print(f"Memory Tools: {config.enable_memory_tools}")
    
    try:
        agent = CoreAgent(config)
        print("✅ Coder agent created successfully!")
        print(f"   Optimized for: Large codebase analysis")
        print(f"   Best for: Feature implementation, refactoring, bug fixes")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_unit_tester_agent():
    """Demo: Unit Tester Agent Configuration"""
    print("🧪 UNIT TESTER AGENT CONFIG")
    print("=" * 50)
    
    config = UNIT_TESTER_AGENT_CONFIG
    config.model = MockChatModel()
    
    print(f"Agent Name: {config.name}")
    print(f"Memory Backend: {config.memory_backend}")
    print(f"Memory Types: {config.memory_types}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Rate Limiting: {config.enable_rate_limiting}")
    print(f"Memory Tools: {config.enable_memory_tools}")
    
    try:
        agent = CoreAgent(config)
        print("✅ Unit Tester agent created successfully!")
        print(f"   Optimized for: Fast test generation")
        print(f"   Best for: Test coverage analysis, edge case detection")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_executer_agent():
    """Demo: Executer Agent Configuration"""
    print("⚡ EXECUTER AGENT CONFIG")
    print("=" * 50)
    
    config = EXECUTER_AGENT_CONFIG
    config.model = MockChatModel()
    
    print(f"Agent Name: {config.name}")
    print(f"Memory Backend: {config.memory_backend}")
    print(f"Memory Types: {config.memory_types}")
    print(f"Rate Limiting: {config.requests_per_second}/s")
    print(f"Streaming: {config.enable_streaming}")
    print(f"Session Memory: {'session' in config.memory_types}")
    
    try:
        agent = CoreAgent(config)
        print("✅ Executer agent created successfully!")
        print(f"   Optimized for: High-throughput execution")
        print(f"   Best for: Running tests, build scripts, benchmarks")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_code_reviewer_agent():
    """Demo: Code Reviewer Agent Configuration"""
    print("🔍 CODE REVIEWER AGENT CONFIG")
    print("=" * 50)
    
    config = CODE_REVIEWER_AGENT_CONFIG
    config.model = MockChatModel()
    
    print(f"Agent Name: {config.name}")
    print(f"Memory Backend: {config.memory_backend}")
    print(f"Memory Types: {config.memory_types}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Rate Limiting: {config.requests_per_second}/s")
    print(f"Semantic Search: {config.enable_semantic_search}")
    print(f"Evaluation Metrics: {config.evaluation_metrics}")
    
    try:
        agent = CoreAgent(config)
        print("✅ Code Reviewer agent created successfully!")
        print(f"   Optimized for: Comprehensive code analysis")
        print(f"   Best for: PR reviews, security audits, best practices")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_build_agent():
    """Demo: Build Agent Configuration"""
    print("🏗️ BUILD AGENT CONFIG")
    print("=" * 50)
    
    config = BUILD_AGENT_CONFIG
    config.model = MockChatModel()
    
    print(f"Agent Name: {config.name}")
    print(f"Memory Backend: {config.memory_backend}")
    print(f"Memory Types: {config.memory_types}")
    print(f"TTL Enabled: {config.enable_ttl}")
    print(f"TTL Duration: {config.default_ttl_minutes} minutes")
    print(f"Rate Limiting: {config.requests_per_second}/s")
    print(f"Streaming: {config.enable_streaming}")
    
    try:
        agent = CoreAgent(config)
        print("✅ Build agent created successfully!")
        print(f"   Optimized for: CI/CD pipeline automation")
        print(f"   Best for: Version control, build automation, deployment")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_orchestrator_agent():
    """Demo: Orchestrator Agent Configuration"""
    print("🎼 ORCHESTRATOR AGENT CONFIG")
    print("=" * 50)
    
    config = ORCHESTRATOR_AGENT_CONFIG
    config.model = MockChatModel()
    
    print(f"Agent Name: {config.name}")
    print(f"Memory Backend: {config.memory_backend}")
    print(f"Memory Types: {config.memory_types}")
    print(f"Supervisor Enabled: {config.enable_supervisor}")
    print(f"Rate Limiting: {config.requests_per_second}/s")
    print(f"Streaming: {config.enable_streaming}")
    print(f"Evaluation: {config.evaluation_metrics}")
    
    try:
        agent = CoreAgent(config)
        print("✅ Orchestrator agent created successfully!")
        print(f"   Optimized for: Multi-agent coordination")
        print(f"   Best for: Workflow management, task distribution")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    print()


def demo_workflow_scenarios():
    """Demo: Real-world software development scenarios"""
    print("🌍 SOFTWARE DEVELOPMENT WORKFLOWS")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "Feature Development Pipeline",
            "agents": ["coder", "unit_tester", "code_reviewer", "executer"],
            "description": "Complete feature development with testing and review",
            "workflow": [
                "1. Coder Agent: Implement feature code",
                "2. Unit Tester Agent: Generate comprehensive tests",
                "3. Code Reviewer Agent: Review code quality and security",
                "4. Executer Agent: Run tests and validate implementation"
            ]
        },
        {
            "name": "CI/CD Automation Pipeline",
            "agents": ["executer", "build", "orchestrator"],
            "description": "Automated build and deployment workflow",
            "workflow": [
                "1. Executer Agent: Run all tests and quality checks",
                "2. Build Agent: Create build artifacts and version tags",
                "3. Orchestrator Agent: Coordinate deployment process"
            ]
        },
        {
            "name": "Code Quality Assurance",
            "agents": ["code_reviewer", "unit_tester", "executer"],
            "description": "Comprehensive code quality validation",
            "workflow": [
                "1. Code Reviewer Agent: Static analysis and best practices",
                "2. Unit Tester Agent: Generate missing test cases",
                "3. Executer Agent: Validate test coverage and performance"
            ]
        },
        {
            "name": "Legacy Code Refactoring",
            "agents": ["coder", "unit_tester", "code_reviewer", "executer"],
            "description": "Safe refactoring with comprehensive validation",
            "workflow": [
                "1. Code Reviewer Agent: Analyze current code structure",
                "2. Unit Tester Agent: Create safety net of tests",
                "3. Coder Agent: Implement refactoring changes",
                "4. Executer Agent: Validate refactoring success"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Agents: {', '.join(scenario['agents'])}")
        print(f"   Purpose: {scenario['description']}")
        print(f"   Workflow:")
        for step in scenario['workflow']:
            print(f"     {step}")
        print()


def demo_orchestrator_setup():
    """Demo: Setting up Orchestrator with multiple agents"""
    print("🎼 ORCHESTRATOR MULTI-AGENT SETUP")
    print("=" * 50)
    
    # Create individual agent configs
    coder_config = CODER_AGENT_CONFIG
    coder_config.model = MockChatModel()
    
    tester_config = UNIT_TESTER_AGENT_CONFIG  
    tester_config.model = MockChatModel()
    
    reviewer_config = CODE_REVIEWER_AGENT_CONFIG
    reviewer_config.model = MockChatModel()
    
    executer_config = EXECUTER_AGENT_CONFIG
    executer_config.model = MockChatModel()
    
    build_config = BUILD_AGENT_CONFIG
    build_config.model = MockChatModel()
    
    # Create orchestrator with supervised agents
    orchestrator_config = ORCHESTRATOR_AGENT_CONFIG
    orchestrator_config.model = MockChatModel()
    
    # Configure supervised agents
    orchestrator_config.agents = {
        "coder": coder_config,
        "unit_tester": tester_config,
        "code_reviewer": reviewer_config,
        "executer": executer_config,
        "build": build_config
    }
    
    print("Agent Configuration:")
    print(f"  🎼 Orchestrator: Workflow coordination")
    print(f"  👨‍💻 Coder: {coder_config.max_tokens} tokens, PostgreSQL knowledge")
    print(f"  🧪 Unit Tester: {tester_config.max_tokens} tokens, fast iteration")
    print(f"  🔍 Code Reviewer: {reviewer_config.max_tokens} tokens, quality analysis")
    print(f"  ⚡ Executer: {executer_config.requests_per_second}/s rate, real-time feedback")
    print(f"  🏗️ Build: TTL {build_config.default_ttl_minutes}min, CI/CD automation")
    
    try:
        orchestrator = CoreAgent(orchestrator_config)
        print("✅ Multi-agent orchestrator setup successful!")
        print(f"   Total agents: {len(orchestrator_config.agents) + 1}")
        print(f"   Coordination: Supervisor pattern enabled")
        
    except Exception as e:
        print(f"❌ Orchestrator setup failed: {e}")
    
    print()


def demo_config_selection_guide():
    """Demo: Guide for selecting the right agent config"""
    print("🎯 AGENT SELECTION GUIDE")
    print("=" * 50)
    
    guide = [
        {
            "task": "Generate new feature code",
            "agent": "coder",
            "why": "Large context, PostgreSQL knowledge, semantic code search"
        },
        {
            "task": "Write comprehensive unit tests",
            "agent": "unit_tester", 
            "why": "Fast iteration, test pattern recognition, coverage analysis"
        },
        {
            "task": "Review pull requests",
            "agent": "code_reviewer",
            "why": "Quality analysis, security detection, best practices enforcement"
        },
        {
            "task": "Run tests and benchmarks",
            "agent": "executer",
            "why": "High throughput, real-time feedback, execution tracking"
        },
        {
            "task": "Build and deploy applications",
            "agent": "build",
            "why": "CI/CD automation, version control, artifact management"
        },
        {
            "task": "Coordinate development workflow",
            "agent": "orchestrator",
            "why": "Multi-agent supervision, workflow state management"
        }
    ]
    
    print("Task-Agent Mapping:")
    for item in guide:
        print(f"📋 {item['task']}")
        print(f"   → Use: {item['agent']} agent")
        print(f"   → Why: {item['why']}")
        print()


if __name__ == "__main__":
    print("🧑‍💻 SOFTWARE DEVELOPMENT WORKFLOW DEMO")
    print("=====================================")
    print()
    
    # Configuration comparison
    print_config_comparison()
    print()
    
    # Individual agent demos
    demo_coder_agent()
    demo_unit_tester_agent()
    demo_executer_agent()
    demo_code_reviewer_agent()
    demo_build_agent()
    demo_orchestrator_agent()
    
    # Workflow scenarios
    demo_workflow_scenarios()
    demo_orchestrator_setup()
    demo_config_selection_guide()
    
    print("🎉 DEMO COMPLETED!")
    print()
    print("💡 Next Steps:")
    print("   1. Choose agents for your development workflow")
    print("   2. Import configs: from pre_built_configs import CODER_AGENT_CONFIG")
    print("   3. Add your model: config.model = ChatOpenAI()")
    print("   4. Create agents: coder = CoreAgent(CODER_AGENT_CONFIG)")
    print("   5. Set up orchestrator for multi-agent coordination!")
    print()
    print("🚀 Happy coding with AI agents!")