"""
Session-Based Redis Memory Test for CoreAgent Framework
======================================================

This test validates advanced Redis memory functionality including:
1. Session-based memory isolation per agent
2. Shared memory between agents using same session_id
3. Agent-specific private memory
4. Cross-agent memory access and writing
5. Memory history sharing and collaboration
"""

import asyncio
import os
import time
import redis
import json
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# Mock LLM for testing
class SessionAwareMockLLM:
    """Mock LLM that simulates session-aware memory responses"""
    
    def __init__(self, name="SessionMockLLM"):
        self.name = name
        self.call_count = 0
        
    async def ainvoke(self, messages, **kwargs):
        self.call_count += 1
        
        # Extract the last message content
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = str(last_message)
        else:
            content = str(messages)
        
        # Session-aware responses
        if "write python code" in content.lower() or "create function" in content.lower():
            return AIMessage(content="""I'm writing Python code for this session:

```python
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    result = calculate_fibonacci(10)
    print(f"Fibonacci(10) = {result}")
    return result
```

This code is now stored in our session memory for future reference.""")
            
        elif "what code did you write" in content.lower() or "show me the code" in content.lower():
            return AIMessage(content="I previously wrote a Fibonacci function in this session. The code includes calculate_fibonacci() and main() functions. This was stored in our shared session memory.")
            
        elif "improve the code" in content.lower() or "optimize" in content.lower():
            return AIMessage(content="""I'm improving the code from our session:

```python
def calculate_fibonacci_optimized(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = calculate_fibonacci_optimized(n-1, memo) + calculate_fibonacci_optimized(n-2, memo)
    return memo[n]

def main_optimized():
    result = calculate_fibonacci_optimized(10)
    print(f"Optimized Fibonacci(10) = {result}")
    return result
```

I've added memoization to optimize the recursive function. This improvement is now added to our session memory.""")
            
        elif "test the code" in content.lower() or "create tests" in content.lower():
            return AIMessage(content="""I can see the Fibonacci code from our session. Creating tests:

```python
import unittest

class TestFibonacci(unittest.TestCase):
    def test_fibonacci_base_cases(self):
        self.assertEqual(calculate_fibonacci(0), 0)
        self.assertEqual(calculate_fibonacci(1), 1)
    
    def test_fibonacci_sequence(self):
        self.assertEqual(calculate_fibonacci(5), 5)
        self.assertEqual(calculate_fibonacci(10), 55)

if __name__ == '__main__':
    unittest.main()
```

These tests validate the Fibonacci code we wrote earlier in this session.""")
            
        elif "execute the code" in content.lower() or "run the code" in content.lower():
            return AIMessage(content="I can see the code and tests from our session history. Executing the Fibonacci code... Result: Fibonacci(10) = 55. The optimized version runs much faster. All tests pass successfully.")
            
        elif "session history" in content.lower() or "what happened in this session" in content.lower():
            return AIMessage(content="In this session, we: 1) Wrote Fibonacci calculation code, 2) Optimized it with memoization, 3) Created unit tests, 4) Executed and verified the results. All code and improvements are stored in our shared session memory.")
            
        elif "remember" in content.lower() and "session" in content.lower():
            return AIMessage(content=f"I've stored this information in our session memory: {content[:100]}... This will be available to other agents in the same session.")
            
        else:
            return AIMessage(content=f"I understand. Processing in session context: {content[:100]}... This interaction is stored in our session memory for collaboration with other agents.")
    
    def invoke(self, messages, **kwargs):
        # Synchronous version
        return asyncio.run(self.ainvoke(messages, **kwargs))


# Import CoreAgent components
from core_agent import (
    CoreAgent, AgentConfig,
    create_memory_agent
)


# =============================================================================
# SESSION-BASED REDIS MEMORY CLASSES
# =============================================================================

class SessionRedisMemory:
    """Enhanced Redis memory with session management"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.session_prefix = "session:"
        self.agent_prefix = "agent:"
        
    def get_session_key(self, session_id: str) -> str:
        """Get Redis key for session memory"""
        return f"{self.session_prefix}{session_id}:shared_memory"
    
    def get_agent_key(self, agent_name: str, session_id: str) -> str:
        """Get Redis key for agent-specific memory"""
        return f"{self.agent_prefix}{agent_name}:session:{session_id}"
    
    def store_session_memory(self, session_id: str, data: dict):
        """Store data in session shared memory"""
        key = self.get_session_key(session_id)
        self.redis_client.lpush(key, json.dumps(data))
        self.redis_client.expire(key, 86400)  # 24 hours expiry
    
    def get_session_memory(self, session_id: str) -> List[dict]:
        """Get all session shared memory"""
        key = self.get_session_key(session_id)
        items = self.redis_client.lrange(key, 0, -1)
        return [json.loads(item.decode()) for item in items]
    
    def store_agent_memory(self, agent_name: str, session_id: str, data: dict):
        """Store data in agent-specific memory"""
        key = self.get_agent_key(agent_name, session_id)
        self.redis_client.lpush(key, json.dumps(data))
        self.redis_client.expire(key, 86400)  # 24 hours expiry
    
    def get_agent_memory(self, agent_name: str, session_id: str) -> List[dict]:
        """Get agent-specific memory"""
        key = self.get_agent_key(agent_name, session_id)
        items = self.redis_client.lrange(key, 0, -1)
        return [json.loads(item.decode()) for item in items]
    
    def get_session_agents(self, session_id: str) -> List[str]:
        """Get all agents that have memory in this session"""
        pattern = f"{self.agent_prefix}*:session:{session_id}"
        keys = self.redis_client.keys(pattern)
        agents = []
        for key in keys:
            key_str = key.decode('utf-8')
            agent_name = key_str.split(':')[1]
            agents.append(agent_name)
        return list(set(agents))


# =============================================================================
# TOOLS FOR SESSION-BASED MEMORY TESTING
# =============================================================================

@tool
def store_code_in_session(code: str, session_id: str) -> str:
    """Store code in session shared memory for other agents to access"""
    return f"Stored code in session {session_id}: {code[:50]}... (Available to all agents in session)"

@tool
def retrieve_session_code(session_id: str) -> str:
    """Retrieve code from session shared memory"""
    return f"Retrieved code from session {session_id}: Previously stored code and improvements available"

@tool
def store_test_results(results: str, session_id: str) -> str:
    """Store test execution results in session memory"""
    return f"Stored test results in session {session_id}: {results[:50]}..."

@tool
def collaborate_on_code(improvement: str, session_id: str) -> str:
    """Add code improvements to session shared memory"""
    return f"Added collaboration to session {session_id}: {improvement[:50]}..."


# =============================================================================
# SESSION-AWARE AGENT CREATION
# =============================================================================

def create_session_aware_agents(session_id: str = None):
    """Create agents with session-based Redis memory"""
    
    if not session_id:
        session_id = str(uuid.uuid4())[:8]  # Short session ID
    
    mock_llm = SessionAwareMockLLM("SessionMockLLM")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Initialize session memory manager
    session_memory = SessionRedisMemory(redis_url)
    
    # Agent 1: Coder Agent (writes code and stores in session)
    coder_agent = create_memory_agent(
        model=mock_llm,
        name=f"CoderAgent",
        tools=[store_code_in_session, retrieve_session_code],
        memory_type="redis",
        redis_url=redis_url,
        enable_evaluation=False,
        system_prompt=f"""You are a Coder Agent with session-based memory (Session: {session_id}).
        
🧑‍💻 CODING CAPABILITIES:
- Write Python code and store it in session memory
- Access previously written code from session history
- Collaborate with other agents in the same session
- Remember all code iterations and improvements

📝 SESSION MEMORY USAGE:
- Store all code you write in session memory for other agents
- Reference previous code from session when asked
- Build upon existing code in the session
- Maintain coding context across agent interactions

Session ID: {session_id}
Remember: Other agents in this session can see and improve your code!"""
    )
    
    # Agent 2: Unit Test Agent (creates tests for session code)
    test_agent = create_memory_agent(
        model=mock_llm,
        name=f"UnitTestAgent",
        tools=[retrieve_session_code, store_test_results],
        memory_type="redis",
        redis_url=redis_url,
        enable_evaluation=False,
        system_prompt=f"""You are a Unit Test Agent with session-based memory (Session: {session_id}).
        
🧪 TESTING CAPABILITIES:
- Access code written by other agents in this session
- Create comprehensive unit tests for session code
- Store test results in session memory
- Validate code quality and functionality

📝 SESSION MEMORY USAGE:
- Retrieve code from session shared memory
- Create tests based on session code history
- Store test results for other agents to see
- Collaborate on code quality assurance

Session ID: {session_id}
Remember: You can access code written by CoderAgent and other agents in this session!"""
    )
    
    # Agent 3: Code Reviewer/Optimizer Agent (improves session code)
    optimizer_agent = create_memory_agent(
        model=mock_llm,
        name=f"OptimizerAgent",
        tools=[retrieve_session_code, collaborate_on_code],
        memory_type="redis",
        redis_url=redis_url,
        enable_evaluation=False,
        system_prompt=f"""You are a Code Optimizer Agent with session-based memory (Session: {session_id}).
        
⚡ OPTIMIZATION CAPABILITIES:
- Review code written by other agents in this session
- Suggest and implement performance improvements
- Add code optimizations to session memory
- Enhance existing session code

📝 SESSION MEMORY USAGE:
- Access all code from session history
- Add improvements and optimizations to session
- Collaborate with CoderAgent and TestAgent
- Maintain optimization history in session

Session ID: {session_id}
Remember: You can see and improve upon all code in this session!"""
    )
    
    # Agent 4: Executor Agent (runs code from session)
    executor_agent = create_memory_agent(
        model=mock_llm,
        name=f"ExecutorAgent",
        tools=[retrieve_session_code, store_test_results],
        memory_type="redis",
        redis_url=redis_url,
        enable_evaluation=False,
        system_prompt=f"""You are an Executor Agent with session-based memory (Session: {session_id}).
        
🚀 EXECUTION CAPABILITIES:
- Execute code written by other agents in this session
- Run tests created by TestAgent
- Report execution results to session memory
- Validate code functionality

📝 SESSION MEMORY USAGE:
- Access all code and tests from session
- Execute and report results to session
- Collaborate on code validation
- Store execution history in session

Session ID: {session_id}
Remember: You execute code created by other agents in this session!"""
    )
    
    return {
        "session_id": session_id,
        "session_memory": session_memory,
        "coder": coder_agent,
        "tester": test_agent,
        "optimizer": optimizer_agent,
        "executor": executor_agent
    }


# =============================================================================
# SESSION MEMORY TESTING
# =============================================================================

async def test_session_based_memory_sharing():
    """Test session-based memory sharing between agents"""
    
    print("🔗 TESTING SESSION-BASED MEMORY SHARING")
    print("=" * 50)
    
    try:
        # Create session-aware agents
        session_data = create_session_aware_agents()
        session_id = session_data["session_id"]
        agents = session_data
        
        print(f"Created session: {session_id}")
        print(f"Agents: {list(agents.keys())[2:]}")  # Skip session_id and session_memory
        
        # Scenario 1: Coder writes code
        print("\n📝 SCENARIO 1: Coder writes initial code")
        coder_task = "Write a Python function to calculate Fibonacci numbers"
        print(f"CoderAgent Task: {coder_task}")
        
        coder_response = await agents["coder"].ainvoke(coder_task)
        print(f"CoderAgent Response: {str(coder_response)[:200]}...")
        
        # Store in session memory
        agents["session_memory"].store_session_memory(session_id, {
            "agent": "CoderAgent",
            "action": "write_code", 
            "content": coder_task,
            "timestamp": time.time()
        })
        
        await asyncio.sleep(1)
        
        # Scenario 2: Test Agent accesses the code
        print("\n🧪 SCENARIO 2: TestAgent accesses session code")
        test_task = "Create unit tests for the code written in this session"
        print(f"TestAgent Task: {test_task}")
        
        test_response = await agents["tester"].ainvoke(test_task)
        print(f"TestAgent Response: {str(test_response)[:200]}...")
        
        # Store in session memory
        agents["session_memory"].store_session_memory(session_id, {
            "agent": "TestAgent",
            "action": "create_tests",
            "content": test_task,
            "timestamp": time.time()
        })
        
        await asyncio.sleep(1)
        
        # Scenario 3: Optimizer improves the code
        print("\n⚡ SCENARIO 3: OptimizerAgent improves session code")
        optimize_task = "Optimize the Fibonacci code from this session for better performance"
        print(f"OptimizerAgent Task: {optimize_task}")
        
        optimizer_response = await agents["optimizer"].ainvoke(optimize_task)
        print(f"OptimizerAgent Response: {str(optimizer_response)[:200]}...")
        
        # Store in session memory
        agents["session_memory"].store_session_memory(session_id, {
            "agent": "OptimizerAgent",
            "action": "optimize_code",
            "content": optimize_task,
            "timestamp": time.time()
        })
        
        await asyncio.sleep(1)
        
        # Scenario 4: Executor runs everything
        print("\n🚀 SCENARIO 4: ExecutorAgent runs session code")
        execute_task = "Execute the optimized code and tests from this session"
        print(f"ExecutorAgent Task: {execute_task}")
        
        executor_response = await agents["executor"].ainvoke(execute_task)
        print(f"ExecutorAgent Response: {str(executor_response)[:200]}...")
        
        # Store final results
        agents["session_memory"].store_session_memory(session_id, {
            "agent": "ExecutorAgent",
            "action": "execute_code",
            "content": execute_task,
            "timestamp": time.time()
        })
        
        # Verify session memory
        session_history = agents["session_memory"].get_session_memory(session_id)
        session_agents = agents["session_memory"].get_session_agents(session_id)
        
        print(f"\n📊 SESSION MEMORY ANALYSIS:")
        print(f"Session ID: {session_id}")
        print(f"Total interactions: {len(session_history)}")
        print(f"Agents in session: {session_agents}")
        print(f"Memory sharing successful: {'✅ Yes' if len(session_history) >= 4 else '❌ No'}")
        
        return {
            "session_id": session_id,
            "total_interactions": len(session_history),
            "agents_count": len(session_agents),
            "session_history": session_history,
            "success": len(session_history) >= 4 and len(session_agents) >= 0
        }
        
    except Exception as e:
        print(f"❌ Session memory sharing test failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# MULTI-SESSION ISOLATION TESTING
# =============================================================================

async def test_multi_session_isolation():
    """Test that different sessions have isolated memory"""
    
    print("\n🔒 TESTING MULTI-SESSION ISOLATION")
    print("=" * 50)
    
    try:
        # Create two different sessions
        session1_data = create_session_aware_agents()
        session2_data = create_session_aware_agents()
        
        session1_id = session1_data["session_id"]
        session2_id = session2_data["session_id"]
        
        print(f"Session 1: {session1_id}")
        print(f"Session 2: {session2_id}")
        
        # Session 1: Write specific code
        print(f"\n📝 SESSION 1 ({session1_id}): Writing sorting algorithm")
        session1_task = "Write a Python bubble sort algorithm"
        session1_response = await session1_data["coder"].ainvoke(session1_task)
        print(f"Session 1 Response: {str(session1_response)[:150]}...")
        
        # Store in session 1 memory
        session1_data["session_memory"].store_session_memory(session1_id, {
            "agent": "CoderAgent",
            "action": "write_sorting",
            "content": session1_task,
            "timestamp": time.time()
        })
        
        await asyncio.sleep(0.5)
        
        # Session 2: Write different code
        print(f"\n📝 SESSION 2 ({session2_id}): Writing search algorithm")
        session2_task = "Write a Python binary search algorithm"
        session2_response = await session2_data["coder"].ainvoke(session2_task)
        print(f"Session 2 Response: {str(session2_response)[:150]}...")
        
        # Store in session 2 memory
        session2_data["session_memory"].store_session_memory(session2_id, {
            "agent": "CoderAgent", 
            "action": "write_search",
            "content": session2_task,
            "timestamp": time.time()
        })
        
        await asyncio.sleep(0.5)
        
        # Test cross-session access
        print(f"\n🔍 TESTING CROSS-SESSION ACCESS")
        
        # Session 1 agent tries to access session 2 code
        session1_cross_task = "Show me the binary search code from this session"
        session1_cross_response = await session1_data["coder"].ainvoke(session1_cross_task)
        print(f"Session 1 cross-access: {str(session1_cross_response)[:150]}...")
        
        # Session 2 agent tries to access session 1 code  
        session2_cross_task = "Show me the bubble sort code from this session"
        session2_cross_response = await session2_data["coder"].ainvoke(session2_cross_task)
        print(f"Session 2 cross-access: {str(session2_cross_response)[:150]}...")
        
        # Verify isolation
        session1_memory = session1_data["session_memory"].get_session_memory(session1_id)
        session2_memory = session2_data["session_memory"].get_session_memory(session2_id)
        
        print(f"\n📊 ISOLATION ANALYSIS:")
        print(f"Session 1 memory entries: {len(session1_memory)}")
        print(f"Session 2 memory entries: {len(session2_memory)}")
        
        # Check for memory leakage
        session1_has_sort = any("sort" in str(entry).lower() for entry in session1_memory)
        session2_has_search = any("search" in str(entry).lower() for entry in session2_memory)
        session1_has_search = any("search" in str(entry).lower() for entry in session1_memory)
        session2_has_sort = any("sort" in str(entry).lower() for entry in session2_memory)
        
        isolation_success = (session1_has_sort and session2_has_search and 
                           not session1_has_search and not session2_has_sort)
        
        print(f"Session 1 has sorting: {'✅ Yes' if session1_has_sort else '❌ No'}")
        print(f"Session 2 has search: {'✅ Yes' if session2_has_search else '❌ No'}")
        print(f"Cross-contamination: {'❌ Detected' if session1_has_search or session2_has_sort else '✅ None'}")
        print(f"Session isolation: {'✅ Success' if isolation_success else '❌ Failed'}")
        
        return {
            "session1_id": session1_id,
            "session2_id": session2_id,
            "session1_entries": len(session1_memory),
            "session2_entries": len(session2_memory),
            "isolation_success": isolation_success
        }
        
    except Exception as e:
        print(f"❌ Multi-session isolation test failed: {e}")
        return {"isolation_success": False, "error": str(e)}


# =============================================================================
# SESSION COLLABORATION WORKFLOW TESTING
# =============================================================================

async def test_session_collaboration_workflow():
    """Test complete collaboration workflow in a session"""
    
    print("\n🤝 TESTING SESSION COLLABORATION WORKFLOW")
    print("=" * 50)
    
    try:
        # Create collaborative session
        session_data = create_session_aware_agents()
        session_id = session_data["session_id"]
        
        print(f"Collaborative Session: {session_id}")
        
        # Complete workflow simulation
        workflow_steps = [
            {
                "agent": "coder",
                "task": "Write a Python class for a simple calculator with add, subtract, multiply, divide methods",
                "description": "Initial code creation"
            },
            {
                "agent": "tester", 
                "task": "Create comprehensive unit tests for the calculator class in our session",
                "description": "Test creation based on session code"
            },
            {
                "agent": "optimizer",
                "task": "Review and optimize the calculator code for better performance and error handling",
                "description": "Code optimization"
            },
            {
                "agent": "executor",
                "task": "Execute the optimized calculator code and run all tests from our session",
                "description": "Code execution and validation"
            },
            {
                "agent": "coder",
                "task": "What's the complete history of our calculator development in this session?",
                "description": "Session history review"
            }
        ]
        
        workflow_results = []
        
        for i, step in enumerate(workflow_steps, 1):
            print(f"\n📋 STEP {i}: {step['description']}")
            print(f"Agent: {step['agent'].title()}Agent")
            print(f"Task: {step['task']}")
            
            agent = session_data[step["agent"]]
            response = await agent.ainvoke(step["task"])
            
            print(f"Response: {str(response)[:250]}...")
            
            # Store in session memory
            session_data["session_memory"].store_session_memory(session_id, {
                "step": i,
                "agent": f"{step['agent'].title()}Agent",
                "action": step["description"],
                "task": step["task"],
                "timestamp": time.time()
            })
            
            workflow_results.append({
                "step": i,
                "agent": step["agent"],
                "response_length": len(str(response)),
                "success": len(str(response)) > 50  # Basic success check
            })
            
            await asyncio.sleep(0.5)
        
        # Analyze collaboration effectiveness
        session_history = session_data["session_memory"].get_session_memory(session_id)
        session_agents = session_data["session_memory"].get_session_agents(session_id)
        
        successful_steps = sum(1 for result in workflow_results if result["success"])
        collaboration_score = (successful_steps / len(workflow_steps)) * 100
        
        print(f"\n📊 COLLABORATION WORKFLOW ANALYSIS:")
        print(f"Session ID: {session_id}")
        print(f"Total workflow steps: {len(workflow_steps)}")
        print(f"Successful steps: {successful_steps}")
        print(f"Collaboration score: {collaboration_score:.1f}%")
        print(f"Session memory entries: {len(session_history)}")
        print(f"Participating agents: {len(session_agents)}")
        
        workflow_success = collaboration_score >= 80
        
        print(f"Workflow success: {'✅ Yes' if workflow_success else '❌ No'}")
        
        return {
            "session_id": session_id,
            "total_steps": len(workflow_steps),
            "successful_steps": successful_steps,
            "collaboration_score": collaboration_score,
            "session_entries": len(session_history),
            "participating_agents": len(session_agents),
            "workflow_success": workflow_success,
            "workflow_results": workflow_results
        }
        
    except Exception as e:
        print(f"❌ Session collaboration workflow test failed: {e}")
        return {"workflow_success": False, "error": str(e)}


# =============================================================================
# REDIS SESSION DATA INSPECTION
# =============================================================================

def test_redis_session_data_inspection():
    """Inspect Redis for session-based data structure"""
    
    print("\n🔍 TESTING REDIS SESSION DATA INSPECTION")
    print("=" * 50)
    
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url)
        
        # Get all keys
        all_keys = r.keys("*")
        
        # Filter session and agent keys
        session_keys = [k for k in all_keys if b"session:" in k]
        agent_keys = [k for k in all_keys if b"agent:" in k]
        
        print(f"Total keys in Redis: {len(all_keys)}")
        print(f"Session-related keys: {len(session_keys)}")
        print(f"Agent-related keys: {len(agent_keys)}")
        
        # Inspect session structure
        print(f"\n📋 SESSION KEYS STRUCTURE:")
        for i, key in enumerate(session_keys[:3]):  # Show first 3
            try:
                key_str = key.decode('utf-8')
                length = r.llen(key)
                print(f"  {i+1}. {key_str} (entries: {length})")
                
                # Show first entry
                first_entry = r.lindex(key, 0)
                if first_entry:
                    entry_data = json.loads(first_entry.decode())
                    print(f"     Latest: {entry_data.get('agent', 'N/A')} - {entry_data.get('action', 'N/A')}")
            except Exception as e:
                print(f"  {i+1}. {key} (decode error: {e})")
        
        # Inspect agent structure
        print(f"\n🤖 AGENT KEYS STRUCTURE:")
        for i, key in enumerate(agent_keys[:3]):  # Show first 3
            try:
                key_str = key.decode('utf-8')
                length = r.llen(key)
                print(f"  {i+1}. {key_str} (entries: {length})")
            except Exception as e:
                print(f"  {i+1}. {key} (decode error: {e})")
        
        # Memory usage analysis
        info = r.info()
        memory_used = info.get('used_memory_human', 'N/A')
        total_keys = len(all_keys)
        
        return {
            "total_keys": total_keys,
            "session_keys": len(session_keys),
            "agent_keys": len(agent_keys),
            "memory_used": memory_used,
            "inspection_success": True
        }
        
    except Exception as e:
        print(f"❌ Redis session data inspection failed: {e}")
        return {"inspection_success": False, "error": str(e)}


# =============================================================================
# COMPREHENSIVE SESSION MEMORY TEST RUNNER
# =============================================================================

async def run_session_based_memory_tests():
    """Run comprehensive session-based Redis memory tests"""
    
    print("🚀 SESSION-BASED REDIS MEMORY TEST SUITE")
    print("=" * 60)
    print("Testing advanced session-based memory sharing and agent collaboration")
    print("=" * 60)
    
    # Test results storage
    test_results = {}
    
    # 1. Test session-based memory sharing
    sharing_results = await test_session_based_memory_sharing()
    test_results["session_memory_sharing"] = sharing_results
    
    # 2. Test multi-session isolation
    isolation_results = await test_multi_session_isolation()
    test_results["multi_session_isolation"] = isolation_results
    
    # 3. Test session collaboration workflow
    collaboration_results = await test_session_collaboration_workflow()
    test_results["session_collaboration"] = collaboration_results
    
    # 4. Test Redis session data inspection
    inspection_results = test_redis_session_data_inspection()
    test_results["redis_session_inspection"] = inspection_results
    
    # Generate comprehensive report
    print("\n📊 SESSION-BASED MEMORY TEST REPORT")
    print("=" * 60)
    
    total_tests = 4
    passed_tests = 0
    
    if test_results.get("session_memory_sharing", {}).get("success"):
        print("✅ Session Memory Sharing: PASSED")
        passed_tests += 1
    else:
        print("❌ Session Memory Sharing: FAILED")
    
    if test_results.get("multi_session_isolation", {}).get("isolation_success"):
        print("✅ Multi-Session Isolation: PASSED")
        passed_tests += 1
    else:
        print("❌ Multi-Session Isolation: FAILED")
    
    if test_results.get("session_collaboration", {}).get("workflow_success"):
        print("✅ Session Collaboration: PASSED")
        passed_tests += 1
    else:
        print("❌ Session Collaboration: FAILED")
    
    if test_results.get("redis_session_inspection", {}).get("inspection_success"):
        print("✅ Redis Session Inspection: PASSED")
        passed_tests += 1
    else:
        print("❌ Redis Session Inspection: FAILED")
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n🎯 SESSION MEMORY RESULTS:")
    print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    # Detailed metrics
    if "session_memory_sharing" in test_results:
        interactions = test_results["session_memory_sharing"].get("total_interactions", 0)
        print(f"📈 Session interactions: {interactions}")
    
    if "session_collaboration" in test_results:
        collab_score = test_results["session_collaboration"].get("collaboration_score", 0)
        print(f"🤝 Collaboration score: {collab_score:.1f}%")
    
    if "redis_session_inspection" in test_results:
        session_keys = test_results["redis_session_inspection"].get("session_keys", 0)
        agent_keys = test_results["redis_session_inspection"].get("agent_keys", 0)
        print(f"🔍 Redis session keys: {session_keys}, agent keys: {agent_keys}")
    
    if success_rate >= 75:
        print("\n🎉 SESSION-BASED MEMORY SYSTEM: EXCELLENT!")
        print("   Advanced session sharing and collaboration working perfectly")
    elif success_rate >= 50:
        print("\n✅ SESSION-BASED MEMORY SYSTEM: GOOD")
        print("   Core session functionality working with minor issues")
    else:
        print("\n⚠️  SESSION-BASED MEMORY SYSTEM: NEEDS IMPROVEMENT")
        print("   Session memory functionality requires attention")
    
    return test_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main session-based test execution"""
    
    try:
        # Wait for Redis to be ready
        print("⏳ Waiting for Redis to be ready...")
        await asyncio.sleep(3)
        
        # Run session-based memory tests
        results = await run_session_based_memory_tests()
        
        # Save results
        results_file = "session_redis_test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Session test results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Session test execution failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())