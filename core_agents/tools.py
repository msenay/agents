# ============================================================================
# AGENT GENERATION PROMPTS
# ============================================================================

SIMPLE_AGENT_PROMPT = """Generate a complete standalone LangGraph agent:

Name: {agent_name}  
Purpose: {purpose}

Create a SIMPLE agent with:
1. Clean StateGraph implementation
2. TypedDict for state definition
3. Clear node functions for each step
4. Proper edge connections
5. Compiled graph with entry/exit points
6. All necessary imports
7. Example usage at the end

Requirements:
- Use minimal dependencies
- Clear, readable code
- Proper error handling
- Follow LangGraph best practices

Generate ONLY the Python code, no explanations."""

WITH_TOOLS_AGENT_PROMPT = """Generate a complete LangGraph agent with tools:

Name: {agent_name}
Purpose: {purpose}
Tools Required: {tools_needed}

Create an agent WITH TOOLS that includes:
1. StateGraph with proper state management
2. Tool node for handling tool calls
3. Agent node that decides when to use tools
4. Conditional edges for tool routing
5. Proper tool integration and result handling
6. Error handling for tool failures
7. All necessary imports
8. Example usage demonstrating tool usage

Requirements:
- Implement proper tool calling logic
- Handle tool errors gracefully
- Include tool result processing
- Follow LangGraph tool patterns

Generate ONLY the Python code, no explanations."""

MULTI_AGENT_PROMPT = """Generate a complete multi-agent system:

Name: {agent_name}
Purpose: {purpose}
Sub-agents needed: {tools_needed}

Create a MULTI-AGENT system with:
1. Supervisor agent that coordinates tasks
2. Multiple worker agents (based on tools_needed)
3. Proper delegation logic
4. State management across agents
5. Result aggregation
6. Error handling and fallbacks
7. All necessary imports
8. Example usage showing coordination

Requirements:
- Use supervisor pattern
- Clear agent responsibilities
- Proper communication between agents
- Scalable architecture

Generate ONLY the Python code, no explanations."""

CORE_AGENT_SIMPLE_PROMPT = """Generate a LangGraph agent using Core Agent infrastructure:

Name: {agent_name}
Purpose: {purpose}

Create a SIMPLE Core Agent based implementation:
1. Import CoreAgent from /workspace/core/core_agent.py
2. Import AgentConfig from /workspace/core/config.py
3. Create a class that inherits from CoreAgent
4. Use AgentConfig for configuration
5. Implement any custom methods needed
6. Include proper initialization
7. Add example usage

Requirements:
- Leverage Core Agent's built-in features
- Use appropriate configuration options
- Follow Core Agent patterns

Generate ONLY the Python code, no explanations."""

CORE_AGENT_WITH_TOOLS_PROMPT = """Generate a Core Agent with tools:

Name: {agent_name}
Purpose: {purpose}
Tools: {tools_needed}

Create a Core Agent WITH TOOLS:
1. Import necessary Core Agent components
2. Import or create required tools
3. Configure AgentConfig with tools
4. Leverage Core Agent's tool handling
5. Use memory if beneficial
6. Include error handling
7. Add example usage

Requirements:
- Use Core Agent tool integration
- Configure tools properly
- Enable appropriate features

Generate ONLY the Python code, no explanations."""

def create_agent_generator_tool(model):
    """Factory function to create AgentGeneratorTool with model"""
    
    class AgentGeneratorTool(BaseTool):
        """Tool for generating agent code"""
        name: str = "agent_generator"
        description: str = "Generate LangGraph agent code based on specifications"
        args_schema: type[BaseModel] = AgentGeneratorInput
        
        def _run(self, template_type: str, agent_name: str, purpose: str, 
                 tools_needed: List[str] = None, use_our_core: bool = False) -> str:
            """Generate agent code based on template type"""
            
            if tools_needed is None:
                tools_needed = []
            
            # Select appropriate prompt based on type and core usage
            if use_our_core:
                if template_type == "simple":
                    prompt = CORE_AGENT_SIMPLE_PROMPT
                elif template_type == "with_tools":
                    prompt = CORE_AGENT_WITH_TOOLS_PROMPT
                else:  # multi_agent
                    prompt = MULTI_AGENT_PROMPT  # Multi-agent always uses supervisor pattern
            else:
                if template_type == "simple":
                    prompt = SIMPLE_AGENT_PROMPT
                elif template_type == "with_tools":
                    prompt = WITH_TOOLS_AGENT_PROMPT
                else:  # multi_agent
                    prompt = MULTI_AGENT_PROMPT
            
            # Format the prompt with actual values
            formatted_prompt = prompt.format(
                agent_name=agent_name,
                purpose=purpose,
                tools_needed=tools_needed if tools_needed else "None"
            )
            
            response = model.invoke([HumanMessage(content=formatted_prompt)])
            return response.content
    
    return AgentGeneratorTool()