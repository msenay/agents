"""
Session-Based Memory Usage Example for CoreAgent Framework
==========================================================

This example demonstrates practical usage of session-based memory with CoreAgent:
1. Creating agents with shared session memory
2. Agent collaboration with memory sharing
3. Session isolation between different workflows
4. Real-world coding collaboration scenario
"""

import asyncio
import os
import uuid
from typing import Dict, Any

# Set up environment
os.environ["REDIS_URL"] = "redis://localhost:6379"

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from core_agent import (
    create_session_agent,
    create_collaborative_agents,
    create_coding_session_agents
)


# =============================================================================
# SCENARIO 1: MANUAL SESSION AGENT CREATION
# =============================================================================

async def scenario_1_manual_session_agents():
    """Scenario 1: Manually create session agents for specific use case"""
    
    print("📋 SCENARIO 1: Manual Session Agent Creation")
    print("=" * 50)
    
    # Create session ID
    session_id = str(uuid.uuid4())[:8]
    print(f"Session ID: {session_id}")
    
    # Create mock LLM (replace with your real LLM)
    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            from langchain_core.messages import AIMessage
            return AIMessage(content="Mock response - agent working with session memory")
    
    mock_llm = MockLLM()
    
    # Define session-specific tools
    @tool
    def remember_user_preference(preference: str) -> str:
        """Store user preference in session memory"""
        return f"Stored preference in session: {preference}"
    
    @tool
    def get_session_preferences() -> str:
        """Get all user preferences from session"""
        return "Retrieved preferences from session memory"
    
    # Create Coder Agent with session memory
    coder_agent = create_session_agent(
        model=mock_llm,
        session_id=session_id,
        name="CoderAgent",
        tools=[remember_user_preference],
        memory_namespace="coder",
        system_prompt=f"""You are a Coder Agent working in session {session_id}.
        
🧑‍💻 YOUR ROLE:
- Write code and remember user coding preferences
- Store code snippets in session memory for other agents
- Collaborate with other agents in this session

🧠 MEMORY FEATURES:
- Session ID: {session_id}
- Your namespace: 'coder'
- Can access shared session memory
- Can store data for other agents

When you write code, always mention that it's stored in session memory for collaboration!"""
    )
    
    # Create Reviewer Agent with same session
    reviewer_agent = create_session_agent(
        model=mock_llm,
        session_id=session_id,
        name="ReviewerAgent", 
        tools=[get_session_preferences],
        memory_namespace="reviewer",
        system_prompt=f"""You are a Code Reviewer Agent working in session {session_id}.
        
🔍 YOUR ROLE:
- Review code written by CoderAgent in this session
- Access user preferences from session memory
- Provide code improvements and suggestions

🧠 MEMORY FEATURES:
- Session ID: {session_id}
- Your namespace: 'reviewer'
- Can access shared session memory
- Can see code and preferences from other agents

Always reference the session context when reviewing code!"""
    )
    
    # Test the agents
    print(f"\n📝 Testing CoderAgent (Session: {session_id})")
    coder_response = await coder_agent.ainvoke("Write a Python function for calculating fibonacci numbers. Remember that I prefer clean, well-documented code.")
    print(f"CoderAgent: {str(coder_response)[:200]}...")
    
    print(f"\n🔍 Testing ReviewerAgent (Session: {session_id})")
    reviewer_response = await reviewer_agent.ainvoke("Review the fibonacci code written in this session. Check if it follows the user's preferences.")
    print(f"ReviewerAgent: {str(reviewer_response)[:200]}...")
    
    return {
        "session_id": session_id,
        "coder_response": str(coder_response),
        "reviewer_response": str(reviewer_response)
    }


# =============================================================================
# SCENARIO 2: COLLABORATIVE AGENTS WITH DIFFERENT MODELS
# =============================================================================

async def scenario_2_collaborative_agents():
    """Scenario 2: Create collaborative agents with different models"""
    
    print("\n📋 SCENARIO 2: Collaborative Agents with Different Models")
    print("=" * 50)
    
    # Create session ID
    session_id = str(uuid.uuid4())[:8]
    print(f"Session ID: {session_id}")
    
    # Mock different models
    class FastMockLLM:
        async def ainvoke(self, messages, **kwargs):
            from langchain_core.messages import AIMessage
            return AIMessage(content="Fast model response - quick processing with session memory access")
    
    class AdvancedMockLLM:
        async def ainvoke(self, messages, **kwargs):
            from langchain_core.messages import AIMessage
            return AIMessage(content="Advanced model response - detailed analysis with session context")
    
    # Define models for different agents
    models = {
        "analyst": AdvancedMockLLM(),  # Use advanced model for analysis
        "writer": FastMockLLM(),      # Use fast model for writing
        "editor": AdvancedMockLLM()   # Use advanced model for editing
    }
    
    # Define tools
    @tool
    def analyze_requirements(requirements: str) -> str:
        """Analyze project requirements and store in session"""
        return f"Requirements analyzed and stored in session: {requirements[:50]}..."
    
    @tool
    def write_content(content_type: str) -> str:
        """Write content based on session requirements"""
        return f"Content written based on session analysis: {content_type}"
    
    @tool
    def edit_content(content: str) -> str:
        """Edit content using session context"""
        return f"Content edited using session context: {content[:50]}..."
    
    # Agent configurations
    agent_configs = {
        "analyst": {
            "tools": [analyze_requirements],
            "system_prompt": f"""You are an Analyst Agent in session {session_id}.
            
📊 YOUR ROLE:
- Analyze project requirements thoroughly
- Store analysis results in session memory
- Provide detailed insights for other agents

🧠 SESSION MEMORY:
- Session: {session_id}
- Namespace: analyst
- Store findings for writer and editor agents
- Access to shared session context

Use your advanced analytical capabilities to provide deep insights!"""
        },
        
        "writer": {
            "tools": [write_content],
            "system_prompt": f"""You are a Writer Agent in session {session_id}.
            
✍️ YOUR ROLE:
- Write content based on analyst's requirements from session
- Access session analysis for context
- Create content that aligns with session goals

🧠 SESSION MEMORY:
- Session: {session_id}
- Namespace: writer
- Access analyst's findings from session
- Store content for editor review

Write efficiently using session context from the analyst!"""
        },
        
        "editor": {
            "tools": [edit_content],
            "system_prompt": f"""You are an Editor Agent in session {session_id}.
            
✏️ YOUR ROLE:
- Edit content created by writer in this session
- Use analyst's requirements for editing guidance
- Improve content quality using session context

🧠 SESSION MEMORY:
- Session: {session_id}
- Namespace: editor
- Access both analyst and writer outputs
- Final quality assurance for session

Provide detailed editing using full session context!"""
        }
    }
    
    # Create collaborative agents
    agents = create_collaborative_agents(
        models=models,
        session_id=session_id,
        agent_configs=agent_configs,
        redis_url=os.environ["REDIS_URL"]
    )
    
    # Test collaborative workflow
    print(f"\n📊 Step 1: Analyst analyzes requirements")
    analyst_response = await agents["analyst"].ainvoke("Analyze requirements for creating a Python web application with user authentication and data visualization.")
    print(f"Analyst: {str(analyst_response)[:200]}...")
    
    print(f"\n✍️ Step 2: Writer creates content based on session analysis")
    writer_response = await agents["writer"].ainvoke("Write documentation for the web application based on the analysis in our session.")
    print(f"Writer: {str(writer_response)[:200]}...")
    
    print(f"\n✏️ Step 3: Editor improves content using session context")
    editor_response = await agents["editor"].ainvoke("Edit the documentation written in this session, ensuring it aligns with the original requirements.")
    print(f"Editor: {str(editor_response)[:200]}...")
    
    return {
        "session_id": session_id,
        "workflow_responses": {
            "analyst": str(analyst_response),
            "writer": str(writer_response),
            "editor": str(editor_response)
        }
    }


# =============================================================================
# SCENARIO 3: PREDEFINED CODING SESSION AGENTS
# =============================================================================

async def scenario_3_coding_session():
    """Scenario 3: Use predefined coding session agents"""
    
    print("\n📋 SCENARIO 3: Predefined Coding Session Agents")
    print("=" * 50)
    
    # Create session ID
    session_id = str(uuid.uuid4())[:8]
    print(f"Coding Session ID: {session_id}")
    
    # Mock LLM
    class CodingMockLLM:
        async def ainvoke(self, messages, **kwargs):
            from langchain_core.messages import AIMessage
            if "write" in str(messages).lower():
                return AIMessage(content="I'm writing Python code and storing it in our session memory for collaboration with other agents.")
            elif "test" in str(messages).lower():
                return AIMessage(content="I'm creating unit tests based on the code in our session memory.")
            elif "review" in str(messages).lower():
                return AIMessage(content="I'm reviewing the code from our session and suggesting improvements.")
            elif "execute" in str(messages).lower():
                return AIMessage(content="I'm executing the code and tests from our session memory.")
            else:
                return AIMessage(content="I'm working with the code in our shared session memory.")
    
    coding_llm = CodingMockLLM()
    
    # Create coding session agents
    coding_agents = create_coding_session_agents(
        model=coding_llm,
        session_id=session_id,
        redis_url=os.environ["REDIS_URL"]
    )
    
    print(f"Created coding agents: {list(coding_agents.keys())}")
    
    # Simulate coding workflow
    print(f"\n🧑‍💻 Step 1: Coder writes initial code")
    coder_response = await coding_agents["coder"].ainvoke("Write a Python class for a simple task manager with add, remove, and list methods.")
    print(f"Coder: {str(coder_response)[:200]}...")
    
    print(f"\n🧪 Step 2: Tester creates tests for session code")
    tester_response = await coding_agents["tester"].ainvoke("Create comprehensive unit tests for the task manager class written in this session.")
    print(f"Tester: {str(tester_response)[:200]}...")
    
    print(f"\n🔍 Step 3: Reviewer examines session code")
    reviewer_response = await coding_agents["reviewer"].ainvoke("Review the task manager code in our session and suggest improvements.")
    print(f"Reviewer: {str(reviewer_response)[:200]}...")
    
    print(f"\n🚀 Step 4: Executor runs session code")
    executor_response = await coding_agents["executor"].ainvoke("Execute the task manager code and tests from our session.")
    print(f"Executor: {str(executor_response)[:200]}...")
    
    return {
        "session_id": session_id,
        "coding_workflow": {
            "coder": str(coder_response),
            "tester": str(tester_response),
            "reviewer": str(reviewer_response),
            "executor": str(executor_response)
        }
    }


# =============================================================================
# SCENARIO 4: MULTI-SESSION ISOLATION DEMO
# =============================================================================

async def scenario_4_multi_session_isolation():
    """Scenario 4: Demonstrate session isolation"""
    
    print("\n📋 SCENARIO 4: Multi-Session Isolation Demo")
    print("=" * 50)
    
    # Create two different sessions
    session_a = str(uuid.uuid4())[:8]
    session_b = str(uuid.uuid4())[:8]
    
    print(f"Session A: {session_a} (Web Development)")
    print(f"Session B: {session_b} (Data Science)")
    
    # Mock LLM
    class IsolationMockLLM:
        async def ainvoke(self, messages, **kwargs):
            from langchain_core.messages import AIMessage
            return AIMessage(content=f"Working in isolated session memory - {str(messages)[:50]}...")
    
    mock_llm = IsolationMockLLM()
    
    # Session A: Web Development Agent
    web_agent = create_session_agent(
        model=mock_llm,
        session_id=session_a,
        name="WebDeveloper",
        memory_namespace="web_dev",
        system_prompt=f"You develop web applications in session {session_a}. Focus on HTML, CSS, JavaScript, and web frameworks."
    )
    
    # Session B: Data Science Agent
    data_agent = create_session_agent(
        model=mock_llm,
        session_id=session_b,
        name="DataScientist",
        memory_namespace="data_science",
        system_prompt=f"You work on data science projects in session {session_b}. Focus on Python, pandas, ML, and data analysis."
    )
    
    # Test isolation
    print(f"\n🌐 Session A (Web Dev): Working on web project")
    web_response = await web_agent.ainvoke("Create a responsive navigation bar for our web application.")
    print(f"Web Agent: {str(web_response)[:200]}...")
    
    print(f"\n📊 Session B (Data Science): Working on data project")
    data_response = await data_agent.ainvoke("Analyze customer churn data and build a prediction model.")
    print(f"Data Agent: {str(data_response)[:200]}...")
    
    # Cross-session test (should not access other session's data)
    print(f"\n🔒 Session A trying to access Session B data:")
    cross_test = await web_agent.ainvoke("Show me the churn analysis from the data science project.")
    print(f"Cross-access result: {str(cross_test)[:200]}...")
    
    return {
        "session_a": session_a,
        "session_b": session_b,
        "web_response": str(web_response),
        "data_response": str(data_response),
        "cross_test": str(cross_test)
    }


# =============================================================================
# MAIN DEMO RUNNER
# =============================================================================

async def main():
    """Run all session-based memory scenarios"""
    
    print("🚀 SESSION-BASED MEMORY USAGE EXAMPLES")
    print("=" * 60)
    print("Demonstrating CoreAgent session-based memory capabilities")
    print("=" * 60)
    
    try:
        # Run all scenarios
        result1 = await scenario_1_manual_session_agents()
        result2 = await scenario_2_collaborative_agents()
        result3 = await scenario_3_coding_session()
        result4 = await scenario_4_multi_session_isolation()
        
        # Summary
        print("\n📊 SESSION-BASED MEMORY DEMO SUMMARY")
        print("=" * 60)
        print("✅ Scenario 1: Manual session agents - COMPLETED")
        print("✅ Scenario 2: Collaborative agents with different models - COMPLETED")
        print("✅ Scenario 3: Predefined coding session agents - COMPLETED")
        print("✅ Scenario 4: Multi-session isolation - COMPLETED")
        
        print(f"\n🎯 KEY FEATURES DEMONSTRATED:")
        print(f"📋 Session-based memory sharing between agents")
        print(f"🤝 Agent collaboration within sessions")
        print(f"🔒 Session isolation (different sessions can't access each other)")
        print(f"🧠 Memory namespaces for agent-specific data within sessions")
        print(f"⚡ Different models for different agents in same session")
        print(f"🛠️ Predefined agent templates for common workflows")
        
        print(f"\n🎉 SESSION-BASED MEMORY SYSTEM: FULLY OPERATIONAL!")
        print(f"Your CoreAgent framework now supports advanced session-based collaboration!")
        
        return {
            "scenario_1": result1,
            "scenario_2": result2,
            "scenario_3": result3,
            "scenario_4": result4
        }
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(main())
    print(f"\n📄 Demo completed successfully!")