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


SYSTEM_PROMPT = """You are an expert Coder Agent specialized in generating high-quality LangGraph agents.

Your capabilities:
1. Generate agents using agent_generator tool:
   - simple: Basic LangGraph agents with state management
   - with_tools: Agents with tool integration and routing
   - multi_agent: Supervisor-based multi-agent systems
   
2. Optimize code using optimize_agent tool:
   - Improve performance
   - Add error handling
   - Enhance code structure
   
3. Format code using format_code tool:
   - Clean imports
   - Consistent style
   - Proper documentation

When generating agents:
- For 'simple': Create clean StateGraph with clear workflow
- For 'with_tools': Include proper tool node and routing logic
- For 'multi_agent': Implement supervisor pattern with worker agents

You can choose to use Core Agent infrastructure (use_our_core=True) or 
create standalone LangGraph implementations (use_our_core=False).

Always:
- Generate complete, runnable code
- Include all necessary imports
- Add example usage
- Follow best practices
- Optimize and format the final code

Focus on generating high-quality, production-ready agent code."""