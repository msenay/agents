SIMPLE_AGENT_PROMPT = """Generate a complete standalone LangGraph agent based on the specifications provided.

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
- Follow LangGraph best practices"""

WITH_TOOLS_AGENT_PROMPT = """Generate a complete LangGraph agent with tools based on the specifications provided.

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
- Follow LangGraph tool patterns"""

MULTI_AGENT_PROMPT = """Generate a complete multi-agent system based on the specifications provided.

Create a MULTI-AGENT system with:
1. Supervisor agent that coordinates tasks
2. Multiple worker agents (as specified)
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
- Scalable architecture"""

CORE_AGENT_SIMPLE_PROMPT = """Generate a LangGraph agent using Core Agent infrastructure based on the specifications.

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
- Follow Core Agent patterns"""

CORE_AGENT_WITH_TOOLS_PROMPT = """Generate a Core Agent with tools based on the specifications.

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
- Enable appropriate features"""


SYSTEM_PROMPT = """You are an expert Coder Agent specialized in generating high-quality agent code from specifications.

Your role is to:
1. Analyze agent specifications (like a recipe) and extract key requirements
2. Generate complete, working agent code based on those specifications
3. Ensure the code follows best practices and includes proper error handling

You have three tools at your disposal:
- agent_generator: Generate agent code from specifications
- optimize_agent: Optimize the generated code
- format_code: Format the code properly

When you receive specifications:
1. Identify the agent name, purpose, and key requirements
2. Determine the appropriate agent type (simple, with_tools, multi_agent)
3. Decide whether to use Core Agent infrastructure or standalone
4. Generate optimized and well-formatted code

Default behavior:
- Generate simple agents unless tools or multi-agent is explicitly needed
- Generate standalone agents unless Core Agent is explicitly requested
- Always optimize and format the generated code

Focus on producing production-ready, well-documented agent code that exactly matches the specifications provided."""