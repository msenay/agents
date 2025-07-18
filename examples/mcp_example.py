"""
MCP (Model Context Protocol) Example

This file demonstrates how to use MCP servers with CoreAgent framework,
including creating MCP servers and connecting agents to them.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import CoreAgent, AgentConfig, create_mcp_agent, MCP_AVAILABLE
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# Mock model for demonstration
class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        message = AIMessage(content="Mock response using MCP tools")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _llm_type(self):
        return "mock"


def create_math_server_file():
    """Create a sample math MCP server file"""
    math_server_code = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

if __name__ == "__main__":
    mcp.run(transport="stdio")
'''
    
    with open("math_server.py", "w") as f:
        f.write(math_server_code)
    print("✅ Created math_server.py")


def create_weather_server_file():
    """Create a sample weather MCP server file"""
    weather_server_code = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    # Mock weather data
    weather_data = {
        "new york": "Sunny, 22°C",
        "london": "Cloudy, 15°C", 
        "tokyo": "Rainy, 18°C",
        "paris": "Partly cloudy, 20°C"
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        return f"Weather in {location}: {weather_data[location_lower]}"
    else:
        return f"Weather in {location}: Data not available (mock server)"

@mcp.tool()
async def get_forecast(location: str, days: int = 3) -> str:
    """Get weather forecast for location."""
    return f"{days}-day forecast for {location}: Mostly sunny with occasional clouds"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
'''
    
    with open("weather_server.py", "w") as f:
        f.write(weather_server_code)
    print("✅ Created weather_server.py")


def demo_mcp_configuration():
    """Demonstrate MCP server configuration"""
    print("\n🔌 MCP CONFIGURATION DEMO")
    print("=" * 40)
    
    # Define MCP server configurations
    mcp_servers = {
        "math": {
            "command": "python",
            "args": [os.path.abspath("math_server.py")],
            "transport": "stdio",
        },
        "weather": {
            # Note: This would require starting the weather server separately
            # python weather_server.py
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
    
    print("📝 MCP Server Configurations:")
    for name, config in mcp_servers.items():
        print(f"  🔹 {name}:")
        for key, value in config.items():
            print(f"     {key}: {value}")
    
    return mcp_servers


def demo_mcp_agent_creation():
    """Demonstrate creating an agent with MCP support"""
    print("\n🤖 MCP AGENT CREATION DEMO") 
    print("=" * 40)
    
    model = MockChatModel()
    
    # MCP server configurations
    mcp_servers = {
        "math": {
            "command": "python",
            "args": [os.path.abspath("math_server.py")],
            "transport": "stdio",
        }
    }
    
    # Create MCP agent using factory function
    mcp_agent = create_mcp_agent(
        model=model,
        mcp_servers=mcp_servers,
        prompt="You are a math assistant with access to calculation tools via MCP."
    )
    
    print(f"✅ Created MCP agent: {mcp_agent.config.name}")
    print(f"📊 MCP enabled: {mcp_agent.config.enable_mcp}")
    print(f"🔧 MCP servers configured: {list(mcp_agent.config.mcp_servers.keys())}")
    
    # Check status
    status = mcp_agent.get_status()
    print(f"📈 Agent status:")
    print(f"   MCP feature: {status['features']['mcp']}")
    print(f"   MCP servers: {status['mcp_servers']}")
    print(f"   MCP tools: {status['mcp_tools']}")
    
    return mcp_agent


def demo_manual_mcp_configuration():
    """Demonstrate manual MCP configuration"""
    print("\n⚙️ MANUAL MCP CONFIGURATION DEMO")
    print("=" * 40)
    
    model = MockChatModel()
    
    # Create agent with MCP configuration
    config = AgentConfig(
        name="CustomMCPAgent",
        model=model,
        system_prompt="You are an assistant with access to math and weather tools.",
        enable_mcp=True,
        mcp_servers={
            "math": {
                "command": "python",
                "args": [os.path.abspath("math_server.py")],
                "transport": "stdio",
            }
        },
        enable_memory=True,
        enable_streaming=True
    )
    
    agent = CoreAgent(config)
    
    print(f"✅ Created custom MCP agent: {agent.config.name}")
    
    # Add more servers dynamically
    agent.add_mcp_server("weather", {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http"
    })
    
    print(f"🔧 Available MCP servers: {agent.get_mcp_servers()}")
    
    return agent


async def demo_mcp_tools_loading():
    """Demonstrate loading tools from MCP servers"""
    print("\n🛠️ MCP TOOLS LOADING DEMO")
    print("=" * 40)
    
    if not MCP_AVAILABLE:
        print("⚠️ MCP not available (install: pip install langchain-mcp-adapters)")
        return
    
    model = MockChatModel()
    
    mcp_servers = {
        "math": {
            "command": "python", 
            "args": [os.path.abspath("math_server.py")],
            "transport": "stdio",
        }
    }
    
    agent = create_mcp_agent(model, mcp_servers)
    
    try:
        # Load MCP tools (this would work with real MCP servers)
        print("🔄 Loading tools from MCP servers...")
        await agent.load_mcp_tools_into_agent()
        
        status = agent.get_status()
        print(f"✅ Loaded {status['mcp_tools']} tools from {status['mcp_servers']} servers")
        
    except Exception as e:
        print(f"⚠️ MCP tools loading demo (requires actual MCP servers): {e}")


def demo_mcp_usage_patterns():
    """Demonstrate different MCP usage patterns"""
    print("\n📋 MCP USAGE PATTERNS")
    print("=" * 40)
    
    patterns = {
        "Math Tools": {
            "server": "math_server.py",
            "transport": "stdio",
            "tools": ["add", "multiply", "divide"],
            "use_case": "Mathematical calculations"
        },
        "Weather API": {
            "server": "weather_server.py", 
            "transport": "streamable_http",
            "tools": ["get_weather", "get_forecast"],
            "use_case": "Weather information"
        },
        "File Operations": {
            "server": "file_server.py",
            "transport": "stdio",
            "tools": ["read_file", "write_file", "list_files"],
            "use_case": "File system operations"
        },
        "Database Access": {
            "server": "db_server.py",
            "transport": "streamable_http", 
            "tools": ["query", "insert", "update"],
            "use_case": "Database operations"
        }
    }
    
    for pattern, info in patterns.items():
        print(f"\n🔹 {pattern}:")
        print(f"   Server: {info['server']}")
        print(f"   Transport: {info['transport']}")
        print(f"   Tools: {', '.join(info['tools'])}")
        print(f"   Use case: {info['use_case']}")


def demo_mcp_best_practices():
    """Demonstrate MCP best practices"""
    print("\n🎯 MCP BEST PRACTICES")
    print("=" * 40)
    
    practices = [
        "🔸 Use stdio transport for local tools and scripts",
        "🔸 Use streamable_http for web services and APIs", 
        "🔸 Keep MCP servers lightweight and focused",
        "🔸 Handle errors gracefully in MCP tools",
        "🔸 Use descriptive tool names and docstrings",
        "🔸 Configure proper timeouts for MCP connections",
        "🔸 Test MCP servers independently before integration",
        "🔸 Use environment variables for MCP server configurations",
        "🔸 Monitor MCP server health and availability",
        "🔸 Implement proper logging in MCP servers"
    ]
    
    for practice in practices:
        print(f"   {practice}")


def cleanup_demo_files():
    """Clean up demo files"""
    files_to_remove = ["math_server.py", "weather_server.py"]
    
    for filename in files_to_remove:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🗑️ Removed {filename}")


async def run_all_demos():
    """Run all MCP demonstrations"""
    print("🔌 CoreAgent MCP (Model Context Protocol) Demo")
    print("=" * 60)
    
    # Check MCP availability
    print(f"📦 MCP Available: {'✅ Yes' if MCP_AVAILABLE else '❌ No (install langchain-mcp-adapters)'}")
    
    # Create sample server files
    create_math_server_file()
    create_weather_server_file()
    
    # Run demos
    demo_mcp_configuration()
    demo_mcp_agent_creation()
    demo_manual_mcp_configuration()
    await demo_mcp_tools_loading()
    demo_mcp_usage_patterns()
    demo_mcp_best_practices()
    
    print("\n" + "=" * 60)
    print("🎉 MCP demo completed!")
    
    if not MCP_AVAILABLE:
        print("\n📥 To enable full MCP functionality:")
        print("pip install langchain-mcp-adapters mcp")
    
    # Cleanup
    print("\n🧹 Cleaning up demo files...")
    cleanup_demo_files()


if __name__ == "__main__":
    asyncio.run(run_all_demos())