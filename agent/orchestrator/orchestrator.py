"""
Orchestrator Agent Implementation

This agent coordinates multiple specialized agents (Coder, Tester, Executor) to work together
in harmony for complete development workflows.
"""

import os
import uuid
from typing import Dict, Any
from datetime import datetime

from agent.coder.coder import CoderAgent
from agent.executor.executor import ExecutorAgent
from agent.tester.tester import TesterAgent
from core import CoreAgent, AgentConfig

from agent.orchestrator.prompts import SYSTEM_PROMPT
from agent.orchestrator.tools import get_orchestrator_tools
from langchain_openai import AzureChatOpenAI


class OrchestratorAgent(CoreAgent):
    """
    OrchestratorAgent coordinates multiple specialized agents to complete complex workflows.
    
    It supports multiple coordination patterns:
    - Supervisor (default): Sequential task delegation with quality control
    - Swarm: Parallel execution for independent tasks
    - Pipeline: Strict sequential processing
    - Adaptive: Dynamic pattern selection based on task
    """
    
    def __init__(
        self,
        session_id: str = None,
        coordination_pattern: str = "supervisor",
        use_all_tools: bool = False,
        enable_monitoring: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize the OrchestratorAgent.
        
        Args:
            session_id: Unique session identifier
            coordination_pattern: Pattern to use (supervisor, swarm, pipeline, adaptive)
            use_all_tools: Whether to use all available orchestration tools
            enable_monitoring: Enable workflow monitoring
            max_retries: Maximum retries for failed tasks
        """
        self.session_id = session_id or f"orchestrator_{uuid.uuid4()}"
        self.coordination_pattern = coordination_pattern
        self.enable_monitoring = enable_monitoring
        self.max_retries = max_retries
        
        # Initialize specialized agents
        self.agents = {
            "coder": CoderAgent(session_id=f"{self.session_id}_coder"),
            "tester": TesterAgent(session_id=f"{self.session_id}_tester"),
            "executor": ExecutorAgent(session_id=f"{self.session_id}_executor")
        }
        
        # Workflow state
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "pending_steps": [],
            "results": {},
            "errors": [],
            "start_time": None,
            "end_time": None
        }
        
        # Configure model
        model = self._get_model()
        
        # Get orchestration tools
        if use_all_tools:
            tools = get_orchestrator_tools(model)
        else:
            # Default tools for orchestration
            default_tools = ["plan_workflow", "delegate_task", "check_quality", "aggregate_results"]
            if coordination_pattern == "swarm":
                default_tools.append("execute_parallel")
            if enable_monitoring:
                default_tools.append("monitor_workflow")
            tools = get_orchestrator_tools(model, default_tools)
        
        # Configure CoreAgent
        config = AgentConfig(
            name="OrchestratorAgent",
            description="Orchestrates multiple agents for complex development workflows",
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            
            # Enable supervisor pattern by default
            enable_supervisor=True if coordination_pattern == "supervisor" else False,
            agents=self.agents if coordination_pattern == "supervisor" else None,
            
            # Enable memory for workflow context
            enable_memory=True,
            memory_types=["short_term", "session"],
            memory_backend="inmemory",
            session_id=self.session_id,
            
            # Performance settings
            enable_rate_limiting=True,
            requests_per_second=10.0,
            
            # Features
            enable_message_trimming=True,
            max_tokens=16000,
            enable_streaming=True
        )
        
        # Initialize CoreAgent
        super().__init__(config)
        
        print(f"✅ OrchestratorAgent initialized with pattern: {coordination_pattern}")
        print(f"📋 Session ID: {self.session_id}")
        print(f"🛠️ Tools: {[tool.name for tool in tools]}")
    
    def _get_model(self):
        """Get the configured model for the orchestrator"""
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
            openai_api_version=os.getenv("OPENAI_API_VERSION", "2023-12-01-preview"),
            deployment_name=os.getenv("GPT4_DEPLOYMENT_NAME", "gpt4"),
            model_name=os.getenv("GPT4_MODEL_NAME", "gpt-4"),
            temperature=0.3,  # Balanced for coordination
            max_tokens=4000,
            timeout=60,
            max_retries=self.max_retries,
            streaming=True
        )
    
    def orchestrate(self, request: str, workflow_type: str = "auto") -> Dict[str, Any]:
        """
        Orchestrate a complete workflow based on the request.
        
        Args:
            request: The user's request
            workflow_type: Type of workflow (auto, full_development, code_review, bug_fix, optimization)
            
        Returns:
            Dictionary containing workflow results and metadata
        """
        print(f"\n🎭 Orchestrating workflow for: {request[:100]}...")
        
        # Initialize workflow state
        self.workflow_state["start_time"] = datetime.now()
        self.workflow_state["request"] = request
        self.workflow_state["workflow_type"] = workflow_type
        
        try:
            # Use the appropriate coordination pattern
            if self.coordination_pattern == "supervisor":
                result = self._orchestrate_supervisor(request, workflow_type)
            elif self.coordination_pattern == "swarm":
                result = self._orchestrate_swarm(request, workflow_type)
            elif self.coordination_pattern == "pipeline":
                result = self._orchestrate_pipeline(request, workflow_type)
            elif self.coordination_pattern == "adaptive":
                result = self._orchestrate_adaptive(request, workflow_type)
            else:
                # Default to supervisor pattern
                result = self._orchestrate_supervisor(request, workflow_type)
            
            # Finalize workflow
            self.workflow_state["end_time"] = datetime.now()
            self.workflow_state["status"] = "completed"
            
            # Create final report
            final_report = self._create_final_report(result)
            
            return {
                "success": True,
                "result": result,
                "report": final_report,
                "workflow_state": self.workflow_state,
                "duration": str(self.workflow_state["end_time"] - self.workflow_state["start_time"])
            }
            
        except Exception as e:
            self.workflow_state["errors"].append(str(e))
            self.workflow_state["status"] = "failed"
            return {
                "success": False,
                "error": str(e),
                "workflow_state": self.workflow_state
            }
    
    def _orchestrate_supervisor(self, request: str, workflow_type: str) -> Dict[str, Any]:
        """Orchestrate using supervisor pattern (sequential with quality control)"""
        # Use the chat interface which has access to tools
        supervisor_request = f"""
Orchestrate this request using the supervisor pattern:

Request: {request}
Workflow Type: {workflow_type}

Steps:
1. First, plan the workflow using the plan_workflow tool
2. Then delegate tasks to appropriate agents sequentially
3. Check quality after each step
4. Aggregate final results

Remember to use the tools available to you for coordination.
"""
        
        response = self.chat(supervisor_request)
        return {"supervisor_result": response}
    
    def _orchestrate_swarm(self, request: str, workflow_type: str) -> Dict[str, Any]:
        """Orchestrate using swarm pattern (parallel execution)"""
        swarm_request = f"""
Orchestrate this request using the swarm pattern for parallel execution:

Request: {request}
Workflow Type: {workflow_type}

Steps:
1. Identify tasks that can be executed in parallel
2. Use the execute_parallel tool to coordinate parallel tasks
3. Synchronize results when needed
4. Aggregate final results

Focus on maximizing parallelism while ensuring correctness.
"""
        
        response = self.chat(swarm_request)
        return {"swarm_result": response}
    
    def _orchestrate_pipeline(self, request: str, workflow_type: str) -> Dict[str, Any]:
        """Orchestrate using pipeline pattern (strict sequential)"""
        pipeline_request = f"""
Orchestrate this request using a strict pipeline pattern:

Request: {request}
Workflow Type: {workflow_type}

Execute in strict sequence:
1. CoderAgent generates/analyzes code
2. TesterAgent creates tests
3. ExecutorAgent validates
4. Loop if needed until all tests pass

Each step must complete successfully before proceeding.
"""
        
        response = self.chat(pipeline_request)
        return {"pipeline_result": response}
    
    def _orchestrate_adaptive(self, request: str, workflow_type: str) -> Dict[str, Any]:
        """Orchestrate using adaptive pattern (dynamic selection)"""
        adaptive_request = f"""
Analyze this request and choose the best coordination pattern:

Request: {request}
Workflow Type: {workflow_type}

Consider:
- Use supervisor for complex workflows needing quality control
- Use swarm for independent parallel tasks
- Use pipeline for strict sequential dependencies

Then orchestrate using the chosen pattern.
"""
        
        response = self.chat(adaptive_request)
        return {"adaptive_result": response}
    
    def _create_final_report(self, result: Dict[str, Any]) -> str:
        """Create a comprehensive final report of the workflow"""
        report = f"""
# Workflow Execution Report

## Overview
- **Session ID**: {self.session_id}
- **Coordination Pattern**: {self.coordination_pattern}
- **Start Time**: {self.workflow_state['start_time']}
- **End Time**: {self.workflow_state['end_time']}
- **Duration**: {self.workflow_state['end_time'] - self.workflow_state['start_time']}
- **Status**: {self.workflow_state.get('status', 'Unknown')}

## Request
{self.workflow_state.get('request', 'N/A')}

## Workflow Type
{self.workflow_state.get('workflow_type', 'N/A')}

## Results Summary
{self._summarize_results(result)}

## Completed Steps
{self._format_completed_steps()}

## Errors
{self._format_errors()}

## Recommendations
{self._generate_recommendations(result)}
"""
        return report
    
    def _summarize_results(self, result: Dict[str, Any]) -> str:
        """Summarize the workflow results"""
        # Extract key information from results
        summary_lines = []
        for key, value in result.items():
            if isinstance(value, str):
                # Truncate long strings
                preview = value[:200] + "..." if len(value) > 200 else value
                summary_lines.append(f"- **{key}**: {preview}")
            else:
                summary_lines.append(f"- **{key}**: {type(value).__name__}")
        
        return "\n".join(summary_lines) if summary_lines else "No results to summarize"
    
    def _format_completed_steps(self) -> str:
        """Format completed steps for the report"""
        steps = self.workflow_state.get("completed_steps", [])
        if not steps:
            return "No steps completed"
        
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    
    def _format_errors(self) -> str:
        """Format errors for the report"""
        errors = self.workflow_state.get("errors", [])
        if not errors:
            return "No errors encountered"
        
        return "\n".join([f"- {error}" for error in errors])
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> str:
        """Generate recommendations based on the workflow results"""
        recommendations = []
        
        # Check for common patterns
        if self.workflow_state.get("errors"):
            recommendations.append("- Review and address the errors encountered")
        
        if self.coordination_pattern == "supervisor" and "parallel" in str(result):
            recommendations.append("- Consider using swarm pattern for better parallelism")
        
        if not recommendations:
            recommendations.append("- Workflow completed successfully")
        
        return "\n".join(recommendations)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all managed agents"""
        status = {
            "orchestrator": {
                "session_id": self.session_id,
                "pattern": self.coordination_pattern,
                "workflow_state": self.workflow_state
            },
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            try:
                agent_status = agent.get_status()
                status["agents"][name] = agent_status
            except:
                status["agents"][name] = {"status": "unknown"}
        
        return status
    
    def reset_workflow(self):
        """Reset the workflow state"""
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "pending_steps": [],
            "results": {},
            "errors": [],
            "start_time": None,
            "end_time": None
        }
        print("🔄 Workflow state reset")
    
    # Direct agent access methods
    def coder(self, task: str) -> str:
        """Direct access to CoderAgent"""
        return self.agents["coder"].chat(task)
    
    def tester(self, task: str) -> str:
        """Direct access to TesterAgent"""
        return self.agents["tester"].chat(task)
    
    def executor(self, task: str) -> str:
        """Direct access to ExecutorAgent"""
        return self.agents["executor"].chat(task)