"""
Orchestrator Agent Implementation

This agent coordinates multiple specialized agents (Coder, Tester, Executor) to work together
in harmony for complete development workflows.
"""

import uuid
from typing import Dict, Any
from datetime import datetime

from agent.coder.coder import CoderAgent
from agent.executor.executor import ExecutorAgent
from agent.tester.tester import TesterAgent
from ai_factory.agents.core import CoreAgent, AgentConfig

from agent.orchestrator.prompts import SYSTEM_PROMPT
from agent.orchestrator.tools import get_orchestrator_tools
from ai_factory.agents.core import get_orchestrator_llm


# OrchestratorConfig removed - now using LLM Factory


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
        
        # Add name attributes to agents
        for name, agent in self.agents.items():
            agent.name = name
        
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
        
        # Create tools dictionary for easy access
        self.tools_dict = {}
        
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
            
            # Enable supervisor pattern only if we have agents
            enable_supervisor=False,  # We'll handle coordination manually
            agents=None,  # Don't pass agents to avoid supervisor initialization issues
            
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
        
        # Populate tools dictionary after initialization
        self.tools_dict = {tool.name: tool for tool in tools}
        
        print(f"✅ OrchestratorAgent initialized with pattern: {coordination_pattern}")
        print(f"📋 Session ID: {self.session_id}")
        print(f"🛠️ Tools: {[tool.name for tool in tools]}")
    
    def _get_model(self):
        """Get the configured model for the orchestrator"""
        return get_orchestrator_llm()
    
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
        self.workflow_state["completed_tasks"] = 0
        self.workflow_state["errors"] = []
        self.workflow_state["status"] = "running"
        
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
        results = {}
        
        # Step 1: Plan the workflow
        workflow_plan = self.tools_dict["plan_workflow"](
            request=request,
            workflow_type=workflow_type,
            available_agents=list(self.agents.keys())
        )
        self.workflow_state["current_step"] = "Workflow planned"
        results["workflow_plan"] = workflow_plan
        
        # Step 2: Execute tasks sequentially
        task_results = []
        for i, task in enumerate(workflow_plan.get("tasks", [])):
            # Delegate task
            task_result = self.tools_dict["delegate_task"](
                task=task["description"],
                agent=task["agent"],
                context=str(task_results),
                requirements=task.get("requirements", [])
            )
            
            # Check quality
            quality_result = self.tools_dict["check_quality"](
                task=task["description"],
                result=task_result,
                criteria=task.get("success_criteria", [])
            )
            
            task_results.append({
                "task": task,
                "result": task_result,
                "quality": quality_result
            })
            
            self.workflow_state["completed_tasks"] += 1
            
            # Stop if quality check fails
            if not quality_result.get("passed", True):
                self.workflow_state["errors"].append(f"Quality check failed for task {i+1}")
                break
        
        # Step 3: Aggregate results
        final_result = self.tools_dict["aggregate_results"](
            results=task_results,
            original_request=request
        )
        
        return final_result
    
    def _orchestrate_swarm(self, request: str, workflow_type: str) -> Dict[str, Any]:
        """Orchestrate using swarm pattern (parallel execution)"""
        # Step 1: Plan the workflow
        workflow_plan = self.tools_dict["plan_workflow"](
            request=request,
            workflow_type=workflow_type,
            available_agents=list(self.agents.keys())
        )
        
        # Step 2: Execute tasks in parallel
        parallel_tasks = []
        for task in workflow_plan.get("tasks", []):
            parallel_tasks.append({
                "agent": task["agent"],
                "task": task["description"],
                "requirements": task.get("requirements", [])
            })
        
        # Execute all tasks in parallel
        parallel_results = self.tools_dict["execute_parallel"](
            tasks=parallel_tasks,
            timeout=300  # 5 minutes timeout
        )
        
        # Step 3: Aggregate results
        final_result = self.tools_dict["aggregate_results"](
            results=parallel_results,
            original_request=request
        )
        
        self.workflow_state["completed_tasks"] = len(parallel_tasks)
        
        return final_result
    
    def _orchestrate_pipeline(self, request: str, workflow_type: str) -> Dict[str, Any]:
        """Orchestrate using pipeline pattern (strict sequential)"""
        results = []
        
        # Step 1: Code generation with CoderAgent
        code_task = self.tools_dict["delegate_task"](
            task=f"Generate code for: {request}",
            agent="coder",
            context="",
            requirements=["Generate complete, working code", "Include error handling", "Follow best practices"]
        )
        results.append({"step": "code_generation", "result": code_task})
        
        # Step 2: Test generation with TesterAgent
        test_task = self.tools_dict["delegate_task"](
            task=f"Generate comprehensive tests for the code",
            agent="tester",
            context=str(code_task),
            requirements=["Create unit tests", "Test edge cases", "Ensure high coverage"]
        )
        results.append({"step": "test_generation", "result": test_task})
        
        # Step 3: Execution and validation with ExecutorAgent
        exec_task = self.tools_dict["delegate_task"](
            task=f"Execute and validate the code with tests",
            agent="executor",
            context=str(results),
            requirements=["Run all tests", "Check for errors", "Validate functionality"]
        )
        results.append({"step": "execution", "result": exec_task})
        
        # Aggregate results
        final_result = self.tools_dict["aggregate_results"](
            results=results,
            original_request=request
        )
        
        self.workflow_state["completed_tasks"] = 3
        
        return final_result
    
    def _orchestrate_adaptive(self, request: str, workflow_type: str) -> Dict[str, Any]:
        """Orchestrate using adaptive pattern (dynamic selection)"""
        # Analyze the request to determine the best pattern
        workflow_plan = self.tools_dict["plan_workflow"](
            request=request,
            workflow_type=workflow_type,
            available_agents=list(self.agents.keys())
        )
        
        # Determine the best pattern based on task dependencies
        tasks = workflow_plan.get("tasks", [])
        has_dependencies = any(task.get("depends_on") for task in tasks)
        needs_quality_check = workflow_type in ["full_development", "production"]
        
        # Choose pattern
        if needs_quality_check:
            # Use supervisor for quality-critical workflows
            self.coordination_pattern = "supervisor"
            return self._orchestrate_supervisor(request, workflow_type)
        elif not has_dependencies and len(tasks) > 2:
            # Use swarm for independent parallel tasks
            self.coordination_pattern = "swarm"
            return self._orchestrate_swarm(request, workflow_type)
        else:
            # Use pipeline for simple sequential workflows
            self.coordination_pattern = "pipeline"
            return self._orchestrate_pipeline(request, workflow_type)
    
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
                # Add name attribute if missing
                if not hasattr(agent, 'name'):
                    agent.name = name
                
                agent_status = agent.get_status() if hasattr(agent, 'get_status') else {"name": name, "status": "active"}
                status["agents"][name] = agent_status
            except Exception as e:
                status["agents"][name] = {"name": name, "status": "unknown", "error": str(e)}
        
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
    
    def chat(self, message: str) -> str:
        """Chat interface for the orchestrator - delegates to invoke"""
        return self.invoke(message)
    
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