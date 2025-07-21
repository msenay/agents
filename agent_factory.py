#!/usr/bin/env python3
"""
🏭 Agent Factory System
=======================

Otomatik agent geliştirme pipeline'ı. Task'a göre:
1. Coder Agent: Agent kodunu yazar 
2. Unit Tester: Unit testler oluşturur
3. Code Reviewer: Kodu review eder
4. Agent Tester: Yazılan agent'ı test eder
5. Orchestrator: Tüm süreci koordine eder

Usage:
    factory = AgentFactory()
    agent_code = factory.create_agent(
        task="Create a sentiment analysis agent",
        api_key="your-openai-key",
        base_url="https://api.openai.com/v1"
    )
"""

import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Import our pre-built configs
from pre_built_configs import (
    CODER_AGENT_CONFIG, 
    UNIT_TESTER_AGENT_CONFIG,
    CODE_REVIEWER_AGENT_CONFIG,
    ORCHESTRATOR_AGENT_CONFIG
)
from core.core_agent import CoreAgent
from langchain_openai import ChatOpenAI


@dataclass
class AgentRequest:
    """Agent creation request"""
    task: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    tools: Optional[List[str]] = None
    requirements: Optional[List[str]] = None
    
    
@dataclass
class AgentResult:
    """Agent creation result"""
    success: bool
    agent_code: str
    test_code: str
    review_feedback: str
    test_results: str
    file_path: Optional[str] = None
    errors: Optional[List[str]] = None


class AgentTesterAgent:
    """Specialized agent for testing generated agents"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        
        # Create agent tester configuration
        config = UNIT_TESTER_AGENT_CONFIG
        config.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=api_key,
            base_url=base_url
        )
        config.name = "AgentTesterAgent"
        config.memory_namespace = "agent_testing"
        
        self.agent = CoreAgent(config)
    
    def test_agent(self, agent_code: str, task_description: str) -> Tuple[bool, str]:
        """Test the generated agent code"""
        
        test_prompt = f"""
        Task: Test this generated agent code to see if it works correctly.
        
        Original Task: {task_description}
        
        Agent Code:
        ```python
        {agent_code}
        ```
        
        Please:
        1. Analyze if the code can be executed
        2. Check if it fulfills the original task requirements
        3. Identify any potential runtime errors
        4. Test with sample inputs if possible
        5. Provide a comprehensive test report
        
        Return your analysis in this format:
        SUCCESS: [True/False]
        ISSUES: [List any issues found]
        TEST_REPORT: [Detailed analysis]
        RECOMMENDATIONS: [Suggestions for improvement]
        """
        
        try:
            response = self.agent.invoke(test_prompt)
            test_result = response['messages'][-1].content
            
            # Parse the result to determine success
            success = "SUCCESS: True" in test_result or "SUCCESS:True" in test_result
            
            return success, test_result
            
        except Exception as e:
            return False, f"Agent testing failed: {str(e)}"


class AgentFactory:
    """Main Agent Factory for creating specialized agents"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="agent_factory_")
        self.created_agents = []
        
    def create_agent(self, request: AgentRequest) -> AgentResult:
        """Create a complete agent based on the request"""
        
        print(f"🏭 AGENT FACTORY: Creating agent for task: {request.task}")
        print("=" * 60)
        
        # Initialize all agents with the provided API credentials
        agents = self._initialize_agents(request.api_key, request.base_url, request.model)
        
        try:
            # Step 1: Generate agent code
            print("👨‍💻 Step 1: Generating agent code...")
            agent_code = self._generate_agent_code(agents['coder'], request)
            
            # Step 2: Generate unit tests
            print("🧪 Step 2: Generating unit tests...")
            test_code = self._generate_tests(agents['unit_tester'], agent_code, request.task)
            
            # Step 3: Review the code
            print("🔍 Step 3: Reviewing code quality...")
            review_feedback = self._review_code(agents['code_reviewer'], agent_code, request.task)
            
            # Step 4: Test the agent
            print("⚡ Step 4: Testing the generated agent...")
            test_success, test_results = self._test_agent(agents['agent_tester'], agent_code, request.task)
            
            # Step 5: Save the agent
            print("💾 Step 5: Saving agent files...")
            file_path = self._save_agent(agent_code, test_code, request.task)
            
            # Step 6: Orchestrate final validation
            print("🎼 Step 6: Final orchestration and validation...")
            final_validation = self._orchestrate_validation(
                agents['orchestrator'], 
                agent_code, 
                test_code, 
                review_feedback, 
                test_results,
                request.task
            )
            
            print("✅ Agent creation completed successfully!")
            
            return AgentResult(
                success=test_success,
                agent_code=agent_code,
                test_code=test_code,
                review_feedback=review_feedback,
                test_results=test_results,
                file_path=file_path
            )
            
        except Exception as e:
            print(f"❌ Agent creation failed: {str(e)}")
            return AgentResult(
                success=False,
                agent_code="",
                test_code="",
                review_feedback="",
                test_results=f"Creation failed: {str(e)}",
                errors=[str(e)]
            )
    
    def _initialize_agents(self, api_key: str, base_url: str, model: str) -> Dict[str, Any]:
        """Initialize all required agents with API credentials"""
        
        # Create the model for all agents
        llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            api_key=api_key,
            base_url=base_url
        )
        
        agents = {}
        
        # Coder Agent
        coder_config = CODER_AGENT_CONFIG
        coder_config.model = llm
        coder_config.memory_namespace = "agent_creation"
        agents['coder'] = CoreAgent(coder_config)
        
        # Unit Tester Agent
        tester_config = UNIT_TESTER_AGENT_CONFIG
        tester_config.model = llm
        tester_config.memory_namespace = "agent_testing"
        agents['unit_tester'] = CoreAgent(tester_config)
        
        # Code Reviewer Agent
        reviewer_config = CODE_REVIEWER_AGENT_CONFIG
        reviewer_config.model = llm
        reviewer_config.memory_namespace = "agent_review"
        agents['code_reviewer'] = CoreAgent(reviewer_config)
        
        # Agent Tester (Specialized)
        agents['agent_tester'] = AgentTesterAgent(api_key, base_url)
        
        # Orchestrator Agent
        orchestrator_config = ORCHESTRATOR_AGENT_CONFIG
        orchestrator_config.model = llm
        orchestrator_config.memory_namespace = "agent_orchestration"
        orchestrator_config.agents = {
            "coder": coder_config,
            "unit_tester": tester_config,
            "code_reviewer": reviewer_config
        }
        agents['orchestrator'] = CoreAgent(orchestrator_config)
        
        return agents
    
    def _generate_agent_code(self, coder_agent: CoreAgent, request: AgentRequest) -> str:
        """Generate agent code based on the request"""
        
        tools_section = ""
        if request.tools:
            tools_section = f"\nRequired Tools: {', '.join(request.tools)}"
        
        requirements_section = ""
        if request.requirements:
            requirements_section = f"\nSpecial Requirements: {', '.join(request.requirements)}"
        
        prompt = f"""
        Create a complete, production-ready Python agent for the following task:
        
        Task: {request.task}
        API Key: {request.api_key}
        Base URL: {request.base_url}
        Model: {request.model}{tools_section}{requirements_section}
        
        Please create a complete Python class that:
        1. Uses the provided API credentials
        2. Implements the required functionality
        3. Includes proper error handling
        4. Has clear documentation
        5. Can be imported and used immediately
        6. Uses best coding practices
        
        The agent should be a complete, standalone Python file that can be executed.
        Include all necessary imports and make it production-ready.
        
        Format your response as:
        ```python
        [Complete Python code here]
        ```
        """
        
        response = coder_agent.invoke(prompt)
        agent_code = response['messages'][-1].content
        
        # Extract Python code from markdown if present
        if "```python" in agent_code:
            agent_code = agent_code.split("```python")[1].split("```")[0].strip()
        elif "```" in agent_code:
            agent_code = agent_code.split("```")[1].split("```")[0].strip()
            
        return agent_code
    
    def _generate_tests(self, tester_agent: CoreAgent, agent_code: str, task: str) -> str:
        """Generate unit tests for the agent"""
        
        prompt = f"""
        Create comprehensive unit tests for this agent code:
        
        Original Task: {task}
        
        Agent Code:
        ```python
        {agent_code}
        ```
        
        Please create:
        1. Unit tests using pytest
        2. Test various scenarios and edge cases
        3. Mock external dependencies if needed
        4. Test error handling
        5. Include setup and teardown if necessary
        
        Make the tests thorough and production-ready.
        
        Format your response as:
        ```python
        [Complete test code here]
        ```
        """
        
        response = tester_agent.invoke(prompt)
        test_code = response['messages'][-1].content
        
        # Extract Python code from markdown if present
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0].strip()
        elif "```" in test_code:
            test_code = test_code.split("```")[1].split("```")[0].strip()
            
        return test_code
    
    def _review_code(self, reviewer_agent: CoreAgent, agent_code: str, task: str) -> str:
        """Review the generated agent code"""
        
        prompt = f"""
        Review this generated agent code for quality, security, and best practices:
        
        Original Task: {task}
        
        Code to Review:
        ```python
        {agent_code}
        ```
        
        Please provide:
        1. Code quality assessment
        2. Security vulnerability analysis
        3. Performance considerations
        4. Best practices compliance
        5. Suggestions for improvement
        6. Overall rating (1-10)
        
        Be thorough and constructive in your review.
        """
        
        response = reviewer_agent.invoke(prompt)
        return response['messages'][-1].content
    
    def _test_agent(self, agent_tester: AgentTesterAgent, agent_code: str, task: str) -> Tuple[bool, str]:
        """Test the generated agent"""
        return agent_tester.test_agent(agent_code, task)
    
    def _orchestrate_validation(self, orchestrator: CoreAgent, agent_code: str, test_code: str, 
                              review_feedback: str, test_results: str, task: str) -> str:
        """Final orchestration and validation"""
        
        prompt = f"""
        As the orchestrator, provide final validation for this agent creation:
        
        Original Task: {task}
        
        Agent Code Quality: {len(agent_code)} characters
        Test Coverage: {len(test_code)} characters  
        Review Feedback: {review_feedback[:500]}...
        Test Results: {test_results[:500]}...
        
        Please provide:
        1. Overall assessment of the agent creation process
        2. Quality score (1-10)
        3. Recommendations for deployment
        4. Any final concerns or suggestions
        """
        
        response = orchestrator.invoke(prompt)
        return response['messages'][-1].content
    
    def _save_agent(self, agent_code: str, test_code: str, task: str) -> str:
        """Save the generated agent and tests to files"""
        
        # Create a safe filename from task
        safe_task = "".join(c for c in task if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_task = safe_task.replace(' ', '_').lower()[:50]
        
        # Agent file
        agent_file = Path(self.temp_dir) / f"{safe_task}_agent.py"
        with open(agent_file, 'w', encoding='utf-8') as f:
            f.write(f'"""\nGenerated Agent for: {task}\n"""\n\n{agent_code}')
        
        # Test file
        test_file = Path(self.temp_dir) / f"test_{safe_task}_agent.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(f'"""\nUnit tests for: {task}\n"""\n\n{test_code}')
        
        return str(agent_file)
    
    def run_live_test(self, agent_file_path: str) -> Tuple[bool, str]:
        """Run the generated agent in a live environment"""
        
        try:
            # Try to import and run the agent
            result = subprocess.run(
                [sys.executable, "-c", f"exec(open('{agent_file_path}').read())"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, f"Agent executed successfully!\nOutput: {result.stdout}"
            else:
                return False, f"Agent execution failed!\nError: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Agent execution timed out (30s limit)"
        except Exception as e:
            return False, f"Live test failed: {str(e)}"
    
    def get_created_agents(self) -> List[str]:
        """Get list of created agent files"""
        return [str(f) for f in Path(self.temp_dir).glob("*_agent.py")]
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")


def main():
    """Demo of the Agent Factory system"""
    
    print("🏭 AGENT FACTORY DEMO")
    print("=" * 50)
    
    # Example usage
    factory = AgentFactory()
    
    # Create an agent request
    request = AgentRequest(
        task="Create a sentiment analysis agent that analyzes text and returns positive/negative/neutral sentiment",
        api_key="your-openai-api-key-here",  # Replace with real key
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        tools=["text_processing", "sentiment_analysis"],
        requirements=["Handle multiple languages", "Return confidence scores"]
    )
    
    # Create the agent
    result = factory.create_agent(request)
    
    # Display results
    print("\n📊 CREATION RESULTS:")
    print(f"Success: {result.success}")
    print(f"Agent Code Length: {len(result.agent_code)} characters")
    print(f"Test Code Length: {len(result.test_code)} characters")
    print(f"Saved to: {result.file_path}")
    
    if result.success:
        print("\n🎉 Agent created successfully!")
        print("\nAgent Code Preview:")
        print(result.agent_code[:500] + "..." if len(result.agent_code) > 500 else result.agent_code)
        
        # Run live test
        if result.file_path:
            live_success, live_result = factory.run_live_test(result.file_path)
            print(f"\n🧪 Live Test: {'✅ PASSED' if live_success else '❌ FAILED'}")
            print(f"Result: {live_result[:200]}...")
    else:
        print("\n❌ Agent creation failed")
        if result.errors:
            print("Errors:", result.errors)
    
    # Cleanup
    factory.cleanup()


if __name__ == "__main__":
    main()