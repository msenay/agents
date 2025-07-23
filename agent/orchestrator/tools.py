"""
Orchestration Tools for OrchestratorAgent

These tools enable the orchestrator to coordinate multiple agents effectively.
"""

from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from agent.orchestrator.prompts import SUPERVISOR_PROMPT, QUALITY_CONTROL_PROMPT, WORKFLOW_TEMPLATES


# Tool Input Schemas
class DelegateTaskInput(BaseModel):
    """Input for delegating a task to a specific agent"""
    agent_name: str = Field(description="Name of the agent (coder, tester, executor)")
    task: str = Field(description="Task to delegate to the agent")
    context: str = Field(default="", description="Additional context from previous steps")


class WorkflowPlanInput(BaseModel):
    """Input for creating a workflow plan"""
    request: str = Field(description="User request to plan for")
    workflow_type: str = Field(default="auto", description="Type of workflow (auto, full_development, code_review, bug_fix, optimization)")


class QualityCheckInput(BaseModel):
    """Input for quality checking agent output"""
    agent_name: str = Field(description="Name of the agent that produced the output")
    output: str = Field(description="Output to check")
    requirements: str = Field(description="Original requirements to check against")


class ParallelTasksInput(BaseModel):
    """Input for executing tasks in parallel"""
    tasks: List[Dict[str, str]] = Field(description="List of tasks with agent assignments")


def create_workflow_planner_tool(model):
    """Create a tool for planning workflows"""
    
    class WorkflowPlannerTool(BaseTool):
        name: str = "plan_workflow"
        description: str = "Create an execution plan for a user request"
        args_schema: type[BaseModel] = WorkflowPlanInput
        
        def _run(self, request: str, workflow_type: str = "auto") -> str:
            """Plan the workflow for the request"""
            
            # Auto-detect workflow type if needed
            if workflow_type == "auto":
                if "fix" in request.lower() or "bug" in request.lower():
                    workflow_type = "bug_fix"
                elif "optimize" in request.lower() or "performance" in request.lower():
                    workflow_type = "optimization"
                elif "review" in request.lower() or "analyze" in request.lower():
                    workflow_type = "code_review"
                else:
                    workflow_type = "full_development"
            
            # Get workflow template
            template = WORKFLOW_TEMPLATES.get(workflow_type, WORKFLOW_TEMPLATES["full_development"])
            
            # Create detailed plan
            prompt = f"""Create a detailed execution plan for this request:

Request: {request}

Base Template:
{template}

Provide a specific step-by-step plan with:
1. Which agent to use for each step
2. What each agent should do
3. Dependencies between steps
4. Success criteria for each step"""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return WorkflowPlannerTool()


def create_task_delegator_tool(model):
    """Create a tool for delegating tasks to specific agents"""
    
    class TaskDelegatorTool(BaseTool):
        name: str = "delegate_task"
        description: str = "Delegate a task to a specific agent (coder, tester, executor)"
        args_schema: type[BaseModel] = DelegateTaskInput
        
        def _run(self, agent_name: str, task: str, context: str = "") -> str:
            """Delegate task to the specified agent"""
            
            # Create delegation prompt
            prompt = f"""Delegate this task to the {agent_name} agent:

Task: {task}

Previous Context:
{context if context else "No previous context"}

Format the request appropriately for the {agent_name} agent, ensuring:
1. Clear instructions
2. Relevant context is included
3. Expected output format is specified"""
            
            response = model.invoke([HumanMessage(content=prompt)])
            
            # Return formatted delegation
            return f"[DELEGATED TO {agent_name.upper()}]\n{response.content}"
    
    return TaskDelegatorTool()


def create_quality_checker_tool(model):
    """Create a tool for checking output quality"""
    
    class QualityCheckerTool(BaseTool):
        name: str = "check_quality"
        description: str = "Check the quality of an agent's output"
        args_schema: type[BaseModel] = QualityCheckInput
        
        def _run(self, agent_name: str, output: str, requirements: str) -> str:
            """Check output quality against requirements"""
            
            prompt = QUALITY_CONTROL_PROMPT.format(
                agent_name=agent_name,
                output=output
            )
            
            prompt += f"\n\nOriginal Requirements:\n{requirements}"
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return QualityCheckerTool()


def create_parallel_executor_tool(model):
    """Create a tool for executing tasks in parallel"""
    
    class ParallelExecutorTool(BaseTool):
        name: str = "execute_parallel"
        description: str = "Execute multiple tasks in parallel across different agents"
        args_schema: type[BaseModel] = ParallelTasksInput
        
        def _run(self, tasks: List[Dict[str, str]]) -> str:
            """Execute tasks in parallel"""
            
            # Format tasks for parallel execution
            task_list = []
            for i, task in enumerate(tasks):
                agent = task.get("agent", "unknown")
                description = task.get("task", "")
                task_list.append(f"{i+1}. [{agent}] {description}")
            
            prompt = f"""Coordinate these tasks for parallel execution:

{chr(10).join(task_list)}

Provide:
1. Execution strategy
2. Synchronization points
3. How to combine results
4. Error handling approach"""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return ParallelExecutorTool()


def create_result_aggregator_tool(model):
    """Create a tool for aggregating results from multiple agents"""
    
    class ResultAggregatorTool(BaseTool):
        name: str = "aggregate_results"
        description: str = "Aggregate and summarize results from multiple agents"
        
        def _run(self, results: Dict[str, str]) -> str:
            """Aggregate results from multiple agents"""
            
            # Format results
            formatted_results = []
            for agent, result in results.items():
                formatted_results.append(f"**{agent} Results:**\n{result}\n")
            
            prompt = f"""Aggregate and summarize these results from multiple agents:

{chr(10).join(formatted_results)}

Provide:
1. Overall summary
2. Key achievements
3. Any issues or conflicts
4. Recommendations for next steps"""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return ResultAggregatorTool()


def create_workflow_monitor_tool(model):
    """Create a tool for monitoring workflow progress"""
    
    class WorkflowMonitorTool(BaseTool):
        name: str = "monitor_workflow"
        description: str = "Monitor and report on workflow progress"
        
        def _run(self, workflow_state: Dict[str, Any]) -> str:
            """Monitor workflow progress"""
            
            prompt = f"""Monitor this workflow state:

Current Step: {workflow_state.get('current_step', 'Unknown')}
Completed Steps: {workflow_state.get('completed_steps', [])}
Pending Steps: {workflow_state.get('pending_steps', [])}
Errors: {workflow_state.get('errors', [])}

Provide:
1. Progress summary
2. Current status
3. Any bottlenecks or issues
4. Estimated completion"""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return WorkflowMonitorTool()


# Dispatcher Function
def get_orchestrator_tools(model, tool_names: List[str] = None):
    """
    Get specific tools for OrchestratorAgent or all tools if none specified
    
    Args:
        model: The LLM model to pass to tools
        tool_names: List of tool names to create. If None, returns all tools.
    
    Returns:
        List of tool instances
    """
    available_tools = {
        "plan_workflow": create_workflow_planner_tool,
        "delegate_task": create_task_delegator_tool,
        "check_quality": create_quality_checker_tool,
        "execute_parallel": create_parallel_executor_tool,
        "aggregate_results": create_result_aggregator_tool,
        "monitor_workflow": create_workflow_monitor_tool,
    }
    
    if tool_names is None:
        return [factory(model) for factory in available_tools.values()]
    
    tools = []
    for name in tool_names:
        if name in available_tools:
            tools.append(available_tools[name](model))
        else:
            print(f"Warning: Tool '{name}' not found in orchestrator tools")
    
    return tools