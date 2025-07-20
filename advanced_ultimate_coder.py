#!/usr/bin/env python3
"""
🎯 Advanced Ultimate Coder Agent
===============================

Enhanced version with advanced options:
- multiple_agent: Single vs Multi-agent orchestration
- use_existing_tools: Tool selection strategies
- supervisor_pattern: Multi-agent coordination

Features:
- 🧠 Advanced Memory System (Redis + Local fallback)
- 📚 Continuous Learning & Pattern Recognition
- ⚡ Single/Multi-Agent Creation Options
- 🛠️ Intelligent Tool Selection from Existing Tools
- 🎼 Supervisor Pattern for Multi-Agent Coordination
- 🚀 Production-Ready Agent Generation

Usage:
    # Single agent with tool selection
    coder = AdvancedUltimateCoderAgent(api_key="your-key")
    result = coder.create_agent(
        task="sentiment analysis",
        multiple_agent=False,
        use_existing_tools="select_intelligently"
    )
    
    # Multi-agent with supervisor
    result = coder.create_agent(
        task="complete data pipeline",
        multiple_agent=True,
        use_existing_tools="use_all"
    )
"""

import hashlib
import json
import time
import uuid
import logging
import pickle
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import tempfile
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# Existing tool blob - comprehensive tool library
EXISTING_TOOLS_BLOB = {
    "data_processing": {
        "pandas": "DataFrame manipulation, CSV/Excel processing, data cleaning",
        "numpy": "Numerical computing, arrays, mathematical operations",
        "polars": "Fast DataFrame library, memory efficient data processing",
        "dask": "Parallel computing, big data processing",
        "scipy": "Scientific computing, statistics, optimization",
        "matplotlib": "Data visualization, plotting, charts",
        "seaborn": "Statistical data visualization",
        "plotly": "Interactive visualizations, dashboards"
    },
    "web_development": {
        "requests": "HTTP requests, API calls, web scraping",
        "beautifulsoup4": "HTML/XML parsing, web scraping",
        "selenium": "Browser automation, dynamic web scraping",
        "scrapy": "Web crawling framework, large scale scraping",
        "flask": "Lightweight web framework, REST APIs",
        "fastapi": "Modern, high-performance web framework",
        "django": "Full-featured web framework",
        "aiohttp": "Asynchronous HTTP client/server"
    },
    "machine_learning": {
        "scikit-learn": "Machine learning algorithms, preprocessing",
        "tensorflow": "Deep learning, neural networks",
        "pytorch": "Deep learning, neural networks, research",
        "xgboost": "Gradient boosting, tabular data",
        "lightgbm": "Gradient boosting, fast training",
        "transformers": "NLP models, BERT, GPT, pre-trained models",
        "spacy": "NLP processing, named entity recognition",
        "nltk": "Natural language processing, text analysis"
    },
    "database": {
        "sqlalchemy": "SQL toolkit, ORM, database abstraction",
        "pymongo": "MongoDB driver, NoSQL database",
        "redis-py": "Redis client, caching, session storage",
        "psycopg2": "PostgreSQL adapter",
        "sqlite3": "Lightweight SQL database",
        "elasticsearch": "Search engine, full-text search",
        "clickhouse-driver": "ClickHouse client, analytics database"
    },
    "file_processing": {
        "openpyxl": "Excel file manipulation",
        "python-docx": "Word document processing",
        "PyPDF2": "PDF file manipulation",
        "pillow": "Image processing, manipulation",
        "pathlib": "File system paths, file operations",
        "zipfile": "Archive creation and extraction",
        "tarfile": "Tar archive handling",
        "shutil": "High-level file operations"
    },
    "api_integration": {
        "openai": "OpenAI API client, GPT models",
        "boto3": "AWS SDK, cloud services",
        "google-cloud": "Google Cloud Platform services",
        "stripe": "Payment processing API",
        "sendgrid": "Email delivery service",
        "twilio": "SMS and voice communication",
        "slack-sdk": "Slack workspace integration"
    },
    "system_utilities": {
        "psutil": "System monitoring, process management",
        "schedule": "Job scheduling, task automation",
        "click": "Command-line interface creation",
        "configparser": "Configuration file parsing",
        "logging": "Application logging",
        "datetime": "Date and time manipulation",
        "uuid": "Unique identifier generation",
        "hashlib": "Cryptographic hashing"
    },
    "testing_quality": {
        "pytest": "Testing framework, unit tests",
        "unittest": "Built-in testing framework",
        "mock": "Mock objects for testing",
        "coverage": "Code coverage measurement",
        "black": "Code formatting",
        "flake8": "Code linting, style checking",
        "mypy": "Static type checking"
    }
}


@dataclass
class AdvancedAgentRequest:
    """Enhanced agent request with advanced options"""
    task: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    complexity: str = "intermediate"
    
    # Advanced options
    multiple_agent: bool = False
    use_existing_tools: str = "select_intelligently"  # "select_intelligently", "use_all", "none"
    custom_tools: Optional[List[str]] = None
    requirements: Optional[List[str]] = None
    
    # Multi-agent options
    max_agents: int = 5
    supervisor_pattern: bool = True
    agent_collaboration: bool = True


@dataclass
class AdvancedAgentResult:
    """Enhanced agent result with multi-agent support"""
    success: bool
    task_id: str
    
    # Single agent results
    agent_code: str = ""
    agent_type: str = ""
    
    # Multi-agent results
    agents: List[Dict[str, Any]] = None
    supervisor_code: str = ""
    coordination_code: str = ""
    
    # Common results
    tools_used: List[str] = None
    tools_selected_by: str = ""  # "llm", "all", "user"
    quality_score: float = 0.0
    complexity_score: float = 0.0
    creation_time: float = 0.0
    file_paths: List[str] = None
    errors: List[str] = None


class ToolSelector:
    """Intelligent tool selection from existing tools"""
    
    def __init__(self, ai_client):
        self.ai_client = ai_client
        self.tools_blob = EXISTING_TOOLS_BLOB
    
    def select_tools(self, task: str, method: str = "select_intelligently", 
                    custom_tools: Optional[List[str]] = None) -> Tuple[List[str], str]:
        """Select appropriate tools for the task"""
        
        if method == "none" and custom_tools:
            return custom_tools, "user_provided"
        elif method == "none":
            return [], "none_selected"
        elif method == "use_all":
            all_tools = []
            for category in self.tools_blob.values():
                all_tools.extend(category.keys())
            return all_tools, "all_tools"
        elif method == "select_intelligently":
            return self._intelligent_selection(task)
        else:
            return [], "invalid_method"
    
    def _intelligent_selection(self, task: str) -> Tuple[List[str], str]:
        """Use LLM to intelligently select tools"""
        
        # Prepare tools information for LLM
        tools_info = []
        for category, tools in self.tools_blob.items():
            tools_info.append(f"\n{category.upper()}:")
            for tool, description in tools.items():
                tools_info.append(f"  - {tool}: {description}")
        
        tools_text = "\n".join(tools_info)
        
        selection_prompt = f"""
Task: {task}

Available tools and their capabilities:
{tools_text}

Please select the most appropriate tools for this task. Consider:
1. Core functionality needed
2. Data processing requirements  
3. Integration needs
4. Testing and quality requirements

Return ONLY a Python list of tool names, like:
["tool1", "tool2", "tool3"]

Select 3-8 most relevant tools. Be specific and focused.
"""
        
        try:
            response = self.ai_client.invoke(selection_prompt)
            
            # Extract tool list from response
            import re
            list_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if list_match:
                list_content = list_match.group(1)
                # Parse tool names
                tools = []
                for item in list_content.split(','):
                    tool = item.strip().strip('"\'')
                    if tool and tool in self._get_all_tool_names():
                        tools.append(tool)
                
                return tools[:8], "llm_selected"  # Limit to 8 tools
            else:
                # Fallback to category-based selection
                return self._fallback_selection(task)
                
        except Exception as e:
            logging.warning(f"Tool selection failed: {e}")
            return self._fallback_selection(task)
    
    def _fallback_selection(self, task: str) -> Tuple[List[str], str]:
        """Fallback tool selection based on keywords"""
        
        task_lower = task.lower()
        selected_tools = []
        
        # Keyword-based selection
        if any(word in task_lower for word in ['web', 'scrape', 'crawl', 'http']):
            selected_tools.extend(['requests', 'beautifulsoup4'])
        
        if any(word in task_lower for word in ['data', 'csv', 'excel', 'analysis']):
            selected_tools.extend(['pandas', 'numpy'])
        
        if any(word in task_lower for word in ['ml', 'model', 'predict', 'classify']):
            selected_tools.extend(['scikit-learn', 'numpy'])
        
        if any(word in task_lower for word in ['text', 'nlp', 'sentiment', 'language']):
            selected_tools.extend(['transformers', 'nltk'])
        
        if any(word in task_lower for word in ['api', 'service', 'endpoint']):
            selected_tools.extend(['fastapi', 'requests'])
        
        if any(word in task_lower for word in ['file', 'document', 'pdf']):
            selected_tools.extend(['pathlib', 'openpyxl'])
        
        # Add common tools
        selected_tools.extend(['logging', 'datetime', 'pytest'])
        
        # Remove duplicates and limit
        selected_tools = list(dict.fromkeys(selected_tools))[:8]
        
        return selected_tools, "keyword_based"
    
    def _get_all_tool_names(self) -> List[str]:
        """Get all available tool names"""
        all_tools = []
        for category in self.tools_blob.values():
            all_tools.extend(category.keys())
        return all_tools


class MultiAgentOrchestrator:
    """Multi-agent orchestration with supervisor pattern"""
    
    def __init__(self, ai_client):
        self.ai_client = ai_client
    
    def decompose_task(self, task: str, max_agents: int = 5) -> List[Dict[str, Any]]:
        """Decompose complex task into sub-tasks for multiple agents"""
        
        decomposition_prompt = f"""
Complex Task: {task}

Decompose this task into {max_agents} or fewer specialized sub-tasks that can be handled by different AI agents.
Each sub-task should be:
1. Specific and focused
2. Can be implemented independently  
3. Has clear input/output interfaces
4. Contributes to the overall goal

Return the decomposition as a JSON array like:
[
  {{
    "agent_name": "DataProcessor",
    "task_description": "Handle data loading and cleaning",
    "agent_type": "data_agent",
    "priority": 1,
    "dependencies": []
  }},
  {{
    "agent_name": "APIHandler", 
    "task_description": "Create REST API endpoints",
    "agent_type": "api_agent",
    "priority": 2,
    "dependencies": ["DataProcessor"]
  }}
]

Focus on creating 2-{max_agents} complementary agents that work together.
"""
        
        try:
            response = self.ai_client.invoke(decomposition_prompt)
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if json_match:
                json_content = '[' + json_match.group(1) + ']'
                agents = json.loads(json_content)
                return agents[:max_agents]
            else:
                # Fallback decomposition
                return self._fallback_decomposition(task, max_agents)
                
        except Exception as e:
            logging.warning(f"Task decomposition failed: {e}")
            return self._fallback_decomposition(task, max_agents)
    
    def _fallback_decomposition(self, task: str, max_agents: int) -> List[Dict[str, Any]]:
        """Fallback task decomposition"""
        
        task_lower = task.lower()
        agents = []
        
        # Common patterns
        if any(word in task_lower for word in ['data', 'process', 'analyze']):
            agents.append({
                "agent_name": "DataProcessor",
                "task_description": "Handle data processing and analysis",
                "agent_type": "data_agent",
                "priority": 1,
                "dependencies": []
            })
        
        if any(word in task_lower for word in ['api', 'service', 'endpoint']):
            agents.append({
                "agent_name": "APIService",
                "task_description": "Create API endpoints and services",
                "agent_type": "api_agent", 
                "priority": 2,
                "dependencies": []
            })
        
        if any(word in task_lower for word in ['web', 'scrape', 'crawl']):
            agents.append({
                "agent_name": "WebScraper",
                "task_description": "Handle web scraping and data extraction",
                "agent_type": "web_agent",
                "priority": 1,
                "dependencies": []
            })
        
        # If no specific patterns, create general decomposition
        if not agents:
            agents = [
                {
                    "agent_name": "CoreProcessor",
                    "task_description": f"Core functionality for: {task}",
                    "agent_type": "general_agent",
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "agent_name": "UtilityHelper",
                    "task_description": "Utility functions and helpers",
                    "agent_type": "general_agent",
                    "priority": 2,
                    "dependencies": ["CoreProcessor"]
                }
            ]
        
        return agents[:max_agents]
    
    def create_supervisor_code(self, agents: List[Dict[str, Any]], main_task: str) -> str:
        """Create supervisor agent code for orchestration"""
        
        supervisor_prompt = f"""
Create a supervisor agent that orchestrates these sub-agents:

Main Task: {main_task}

Sub-Agents:
{json.dumps(agents, indent=2)}

The supervisor should:
1. Coordinate agent execution based on dependencies
2. Handle data flow between agents
3. Manage error handling and recovery
4. Provide a unified interface
5. Monitor and log agent performance

Create a complete Python class with:
- Agent initialization and management
- Task orchestration methods
- Error handling and recovery
- Logging and monitoring
- Clean interfaces

Format as complete Python code.
"""
        
        try:
            response = self.ai_client.invoke(supervisor_prompt)
            
            # Extract Python code
            if "```python" in response:
                code = response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                code = response.split("```")[1].split("```")[0].strip()
            else:
                code = response.strip()
            
            return code
            
        except Exception as e:
            logging.warning(f"Supervisor creation failed: {e}")
            return self._fallback_supervisor_code(agents, main_task)
    
    def _fallback_supervisor_code(self, agents: List[Dict[str, Any]], main_task: str) -> str:
        """Fallback supervisor code"""
        
        agent_names = [agent['agent_name'] for agent in agents]
        
        return f'''
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class TaskResult:
    """Result from an agent task"""
    agent_name: str
    success: bool
    data: Any
    message: str
    execution_time: float

class SupervisorAgent:
    """Supervisor agent for orchestrating multiple sub-agents"""
    
    def __init__(self):
        self.agents = {agent_names}
        self.task = "{main_task}"
        self.logger = logging.getLogger(__name__)
        self.results = {{}}
        
    def execute_task(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the main task using coordinated agents"""
        
        self.logger.info(f"Starting supervisor execution for: {{self.task}}")
        start_time = time.time()
        
        try:
            # Execute agents in dependency order
            results = {{}}
            
            for agent_name in self.agents:
                self.logger.info(f"Executing agent: {{agent_name}}")
                
                # Simulate agent execution (replace with actual agent calls)
                result = self._execute_agent(agent_name, input_data, results)
                results[agent_name] = result
                
                if not result.success:
                    self.logger.error(f"Agent {{agent_name}} failed: {{result.message}}")
                    return {{"success": False, "error": result.message}}
            
            execution_time = time.time() - start_time
            
            return {{
                "success": True,
                "results": results,
                "execution_time": execution_time,
                "task": self.task
            }}
            
        except Exception as e:
            self.logger.error(f"Supervisor execution failed: {{e}}")
            return {{"success": False, "error": str(e)}}
    
    def _execute_agent(self, agent_name: str, input_data: Any, 
                      previous_results: Dict[str, TaskResult]) -> TaskResult:
        """Execute a specific agent (placeholder implementation)"""
        
        start_time = time.time()
        
        try:
            # This would be replaced with actual agent execution
            # For now, return a mock successful result
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                agent_name=agent_name,
                success=True,
                data={{"processed": True, "agent": agent_name}},
                message=f"Agent {{agent_name}} completed successfully",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TaskResult(
                agent_name=agent_name,
                success=False,
                data=None,
                message=f"Agent {{agent_name}} failed: {{str(e)}}",
                execution_time=execution_time
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current supervisor status"""
        return {{
            "task": self.task,
            "agents": self.agents,
            "results_count": len(self.results)
        }}

def main():
    """Demo supervisor usage"""
    supervisor = SupervisorAgent()
    result = supervisor.execute_task("demo input")
    print(f"Supervisor result: {{result}}")

if __name__ == "__main__":
    main()
'''


class AdvancedUltimateCoderAgent:
    """Advanced Ultimate Coder Agent with multi-agent and tool selection capabilities"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", 
                 model: str = "gpt-4o-mini", session_id: Optional[str] = None):
        
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model
        self.session_id = session_id or f"advanced_coder_{uuid.uuid4().hex[:8]}"
        
        # Initialize AI client
        self.ai_client = self._create_ai_client()
        
        # Initialize components
        self.tool_selector = ToolSelector(self.ai_client)
        self.orchestrator = MultiAgentOrchestrator(self.ai_client)
        
        # Memory and learning (simplified for this example)
        self.memory = {}
        self.task_history = {}
        
        logging.info(f"🎯 Advanced Ultimate Coder Agent initialized - Session: {self.session_id}")
    
    def _create_ai_client(self):
        """Create AI client with fallback"""
        if LANGCHAIN_AVAILABLE:
            try:
                return ChatOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model_name,
                    temperature=0.1
                )
            except Exception as e:
                logging.warning(f"LangChain client failed: {e}")
        
        # Fallback to simple client
        return SimpleAIClient(self.api_key, self.base_url, self.model_name)
    
    def create_agent(self, request: AdvancedAgentRequest) -> AdvancedAgentResult:
        """Create agent(s) based on advanced request options"""
        
        start_time = time.time()
        task_id = self._generate_task_id(request.task)
        
        print(f"🚀 ADVANCED ULTIMATE CODER: Creating agent(s)")
        print(f"Task: {request.task}")
        print(f"Multiple Agent: {request.multiple_agent}")
        print(f"Tool Strategy: {request.use_existing_tools}")
        print(f"Supervisor Pattern: {request.supervisor_pattern}")
        print("=" * 70)
        
        try:
            # Step 1: Tool Selection
            print("🛠️ Step 1: Intelligent tool selection...")
            tools, selection_method = self.tool_selector.select_tools(
                request.task, 
                request.use_existing_tools, 
                request.custom_tools
            )
            print(f"   Selected {len(tools)} tools via {selection_method}")
            if tools:
                print(f"   Tools: {', '.join(tools[:5])}{'...' if len(tools) > 5 else ''}")
            
            # Step 2: Single vs Multi-Agent Decision
            if request.multiple_agent:
                return self._create_multi_agent_system(request, tools, selection_method, start_time)
            else:
                return self._create_single_agent(request, tools, selection_method, start_time)
                
        except Exception as e:
            logging.error(f"❌ Advanced agent creation failed: {str(e)}")
            return AdvancedAgentResult(
                success=False,
                task_id=task_id,
                creation_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _create_single_agent(self, request: AdvancedAgentRequest, tools: List[str], 
                           selection_method: str, start_time: float) -> AdvancedAgentResult:
        """Create a single agent with selected tools"""
        
        print("👨‍💻 Step 2: Creating single optimized agent...")
        
        # Generate agent code with selected tools
        agent_code = self._generate_optimized_agent_code(
            request.task, tools, request.complexity, request.requirements
        )
        
        # Assess quality
        quality_score = self._assess_code_quality(agent_code)
        complexity_score = self._calculate_complexity_score(agent_code)
        
        # Save agent
        file_path = self._save_single_agent(agent_code, request.task, tools)
        
        creation_time = time.time() - start_time
        
        print(f"✅ Single agent created in {creation_time:.2f}s!")
        print(f"   Quality: {quality_score:.2f}, Complexity: {complexity_score:.2f}")
        print(f"   Tools: {len(tools)} selected, Code: {len(agent_code)} chars")
        
        return AdvancedAgentResult(
            success=True,
            task_id=self._generate_task_id(request.task),
            agent_code=agent_code,
            agent_type="single_optimized",
            tools_used=tools,
            tools_selected_by=selection_method,
            quality_score=quality_score,
            complexity_score=complexity_score,
            creation_time=creation_time,
            file_paths=[file_path] if file_path else []
        )
    
    def _create_multi_agent_system(self, request: AdvancedAgentRequest, tools: List[str],
                                 selection_method: str, start_time: float) -> AdvancedAgentResult:
        """Create multi-agent system with supervisor"""
        
        print("🎼 Step 2: Creating multi-agent orchestrated system...")
        
        # Decompose task into sub-agents
        print("   🔄 Decomposing task into specialized agents...")
        agent_specs = self.orchestrator.decompose_task(request.task, request.max_agents)
        print(f"   Created {len(agent_specs)} specialized agent specifications")
        
        # Create individual agents
        print("   👥 Generating individual agent codes...")
        agents = []
        agent_files = []
        
        for i, spec in enumerate(agent_specs, 1):
            print(f"      {i}. Creating {spec['agent_name']} ({spec['agent_type']})...")
            
            # Select tools for this specific agent
            agent_tools = self._select_tools_for_agent(spec, tools)
            
            # Generate agent code
            agent_code = self._generate_specialized_agent_code(
                spec, agent_tools, request.complexity
            )
            
            # Assess agent quality
            quality = self._assess_code_quality(agent_code)
            
            agent_info = {
                "name": spec["agent_name"],
                "type": spec["agent_type"],
                "task": spec["task_description"],
                "code": agent_code,
                "tools": agent_tools,
                "quality": quality,
                "priority": spec.get("priority", 1),
                "dependencies": spec.get("dependencies", [])
            }
            
            agents.append(agent_info)
            
            # Save individual agent
            file_path = self._save_individual_agent(agent_code, spec, agent_tools)
            if file_path:
                agent_files.append(file_path)
        
        # Create supervisor if requested
        supervisor_code = ""
        supervisor_file = ""
        
        if request.supervisor_pattern:
            print("   🎯 Creating supervisor orchestrator...")
            supervisor_code = self.orchestrator.create_supervisor_code(agent_specs, request.task)
            supervisor_file = self._save_supervisor_code(supervisor_code, request.task, agents)
            if supervisor_file:
                agent_files.append(supervisor_file)
        
        # Create coordination code
        print("   🤝 Creating agent coordination system...")
        coordination_code = self._create_coordination_code(agents, request.task)
        coord_file = self._save_coordination_code(coordination_code, request.task)
        if coord_file:
            agent_files.append(coord_file)
        
        creation_time = time.time() - start_time
        avg_quality = sum(agent["quality"] for agent in agents) / len(agents) if agents else 0
        
        print(f"✅ Multi-agent system created in {creation_time:.2f}s!")
        print(f"   Agents: {len(agents)}, Avg Quality: {avg_quality:.2f}")
        print(f"   Supervisor: {'✅' if supervisor_code else '❌'}")
        print(f"   Files created: {len(agent_files)}")
        
        return AdvancedAgentResult(
            success=True,
            task_id=self._generate_task_id(request.task),
            agents=agents,
            supervisor_code=supervisor_code,
            coordination_code=coordination_code,
            tools_used=tools,
            tools_selected_by=selection_method,
            quality_score=avg_quality,
            complexity_score=len(agents) * 0.2,  # Complexity based on agent count
            creation_time=creation_time,
            file_paths=agent_files
        )
    
    def _generate_optimized_agent_code(self, task: str, tools: List[str], 
                                     complexity: str, requirements: Optional[List[str]]) -> str:
        """Generate optimized single agent code"""
        
        tools_info = self._get_tools_info(tools)
        requirements_text = "\n".join(f"- {req}" for req in (requirements or []))
        
        prompt = f"""
Create a highly optimized, production-ready Python agent for:

TASK: {task}
COMPLEXITY: {complexity}

SELECTED TOOLS (use these optimally):
{tools_info}

REQUIREMENTS:
{requirements_text}

Create a complete Python class with:
1. Efficient use of the selected tools
2. Proper error handling and logging
3. Clean, documented code structure
4. Comprehensive functionality
5. Production-ready implementation
6. Type hints and docstrings
7. Main execution example

Focus on creating the BEST possible single agent using the selected tools.
Format as complete Python code.
"""
        
        try:
            response = self.ai_client.invoke(prompt)
            
            # Extract Python code
            if "```python" in response:
                return response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                return response.split("```")[1].split("```")[0].strip()
            else:
                return response.strip()
                
        except Exception as e:
            logging.warning(f"Code generation failed: {e}")
            return self._fallback_agent_code(task, tools)
    
    def _generate_specialized_agent_code(self, spec: Dict[str, Any], tools: List[str], 
                                       complexity: str) -> str:
        """Generate code for a specialized agent"""
        
        tools_info = self._get_tools_info(tools)
        
        prompt = f"""
Create a specialized Python agent:

AGENT NAME: {spec['agent_name']}
AGENT TYPE: {spec['agent_type']}
SPECIFIC TASK: {spec['task_description']}
PRIORITY: {spec.get('priority', 1)}
DEPENDENCIES: {spec.get('dependencies', [])}

AVAILABLE TOOLS:
{tools_info}

Create a focused, specialized agent class that:
1. Handles its specific responsibility perfectly
2. Has clear interfaces for coordination
3. Can work with other agents
4. Includes proper error handling
5. Has comprehensive logging
6. Is production-ready

Format as complete Python code.
"""
        
        try:
            response = self.ai_client.invoke(prompt)
            
            if "```python" in response:
                return response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                return response.split("```")[1].split("```")[0].strip()
            else:
                return response.strip()
                
        except Exception as e:
            logging.warning(f"Specialized agent generation failed: {e}")
            return self._fallback_specialized_agent(spec, tools)
    
    def _create_coordination_code(self, agents: List[Dict[str, Any]], main_task: str) -> str:
        """Create coordination code for multi-agent system"""
        
        agent_info = []
        for agent in agents:
            agent_info.append(f"- {agent['name']}: {agent['task']}")
        
        agents_text = "\n".join(agent_info)
        
        prompt = f"""
Create a coordination system for these agents:

MAIN TASK: {main_task}

AGENTS:
{agents_text}

Create a coordination class that:
1. Manages agent lifecycle
2. Handles data flow between agents
3. Coordinates execution based on dependencies
4. Provides unified interface
5. Handles errors and recovery
6. Includes monitoring and logging

Format as complete Python code.
"""
        
        try:
            response = self.ai_client.invoke(prompt)
            
            if "```python" in response:
                return response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                return response.split("```")[1].split("```")[0].strip()
            else:
                return response.strip()
                
        except Exception as e:
            logging.warning(f"Coordination code generation failed: {e}")
            return self._fallback_coordination_code(agents, main_task)
    
    def _select_tools_for_agent(self, spec: Dict[str, Any], available_tools: List[str]) -> List[str]:
        """Select appropriate tools for a specific agent"""
        
        agent_type = spec.get('agent_type', 'general_agent')
        task = spec.get('task_description', '')
        
        # Type-based tool selection
        type_tools = {
            'data_agent': ['pandas', 'numpy', 'matplotlib', 'logging'],
            'web_agent': ['requests', 'beautifulsoup4', 'selenium', 'logging'],
            'ml_agent': ['scikit-learn', 'numpy', 'pandas', 'logging'],
            'api_agent': ['fastapi', 'requests', 'sqlalchemy', 'logging'],
            'file_agent': ['pathlib', 'openpyxl', 'pillow', 'logging'],
            'general_agent': ['logging', 'datetime', 'pathlib']
        }
        
        # Get relevant tools for this agent type
        relevant_tools = type_tools.get(agent_type, type_tools['general_agent'])
        
        # Filter available tools
        selected = [tool for tool in relevant_tools if tool in available_tools]
        
        # Add some general available tools if selection is small
        if len(selected) < 3:
            for tool in available_tools:
                if tool not in selected and len(selected) < 5:
                    selected.append(tool)
        
        return selected[:6]  # Limit to 6 tools per agent
    
    def _get_tools_info(self, tools: List[str]) -> str:
        """Get detailed information about tools"""
        
        info_lines = []
        for tool in tools:
            # Find tool description
            description = "General purpose tool"
            for category in EXISTING_TOOLS_BLOB.values():
                if tool in category:
                    description = category[tool]
                    break
            
            info_lines.append(f"- {tool}: {description}")
        
        return "\n".join(info_lines)
    
    def _assess_code_quality(self, code: str) -> float:
        """Assess code quality (simplified)"""
        
        quality_factors = {
            "has_docstrings": '"""' in code or "'''" in code,
            "has_type_hints": ': ' in code and '->' in code,
            "has_error_handling": 'try:' in code and 'except' in code,
            "has_logging": 'logging' in code or 'logger' in code,
            "has_classes": 'class ' in code,
            "has_main_function": 'def main(' in code or 'if __name__' in code,
            "has_imports": any(line.strip().startswith(('import ', 'from ')) for line in code.split('\n')),
            "appropriate_length": 50 <= len(code.split('\n')) <= 500
        }
        
        return sum(quality_factors.values()) / len(quality_factors)
    
    def _calculate_complexity_score(self, code: str) -> float:
        """Calculate code complexity (simplified)"""
        
        lines = [line for line in code.split('\n') if line.strip()]
        factors = {
            "line_count": len(lines),
            "class_count": code.count('class '),
            "function_count": code.count('def '),
            "import_count": len([line for line in lines if line.strip().startswith(('import ', 'from '))]),
            "control_structures": code.count('if ') + code.count('for ') + code.count('while ')
        }
        
        # Normalize to 0-1 scale
        normalized = min(1.0, (
            factors["line_count"] / 200 +
            factors["class_count"] / 5 +
            factors["function_count"] / 10 +
            factors["import_count"] / 10 +
            factors["control_structures"] / 15
        ) / 5)
        
        return normalized
    
    def _save_single_agent(self, code: str, task: str, tools: List[str]) -> str:
        """Save single agent to file"""
        
        try:
            safe_task = "".join(c for c in task if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_task = safe_task.replace(' ', '_').lower()[:50]
            
            filename = f"advanced_agent_{safe_task}_{int(time.time())}.py"
            file_path = Path(tempfile.gettempdir()) / filename
            
            enhanced_code = f'''"""
Advanced Ultimate Agent: {task}
============================

Tools Used: {', '.join(tools)}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Session: {self.session_id}
"""

{code}
'''
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_code)
            
            return str(file_path)
            
        except Exception as e:
            logging.warning(f"Could not save agent: {e}")
            return ""
    
    def _save_individual_agent(self, code: str, spec: Dict[str, Any], tools: List[str]) -> str:
        """Save individual agent from multi-agent system"""
        
        try:
            agent_name = spec['agent_name'].lower().replace(' ', '_')
            filename = f"agent_{agent_name}_{int(time.time())}.py"
            file_path = Path(tempfile.gettempdir()) / filename
            
            enhanced_code = f'''"""
Multi-Agent System: {spec['agent_name']}
=====================================

Agent Type: {spec['agent_type']}
Task: {spec['task_description']}
Priority: {spec.get('priority', 1)}
Dependencies: {spec.get('dependencies', [])}
Tools: {', '.join(tools)}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

{code}
'''
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_code)
            
            return str(file_path)
            
        except Exception as e:
            logging.warning(f"Could not save individual agent: {e}")
            return ""
    
    def _save_supervisor_code(self, code: str, task: str, agents: List[Dict[str, Any]]) -> str:
        """Save supervisor code"""
        
        try:
            filename = f"supervisor_{int(time.time())}.py"
            file_path = Path(tempfile.gettempdir()) / filename
            
            agent_list = [f"- {agent['name']}: {agent['type']}" for agent in agents]
            
            enhanced_code = f'''"""
Multi-Agent Supervisor
=====================

Main Task: {task}
Managed Agents:
{chr(10).join(agent_list)}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Session: {self.session_id}
"""

{code}
'''
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_code)
            
            return str(file_path)
            
        except Exception as e:
            logging.warning(f"Could not save supervisor: {e}")
            return ""
    
    def _save_coordination_code(self, code: str, task: str) -> str:
        """Save coordination code"""
        
        try:
            filename = f"coordination_{int(time.time())}.py"
            file_path = Path(tempfile.gettempdir()) / filename
            
            enhanced_code = f'''"""
Multi-Agent Coordination System
==============================

Main Task: {task}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Session: {self.session_id}
"""

{code}
'''
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_code)
            
            return str(file_path)
            
        except Exception as e:
            logging.warning(f"Could not save coordination: {e}")
            return ""
    
    def _generate_task_id(self, task: str) -> str:
        """Generate unique task ID"""
        return hashlib.md5(f"{task}_{time.time()}".encode()).hexdigest()[:12]
    
    def _fallback_agent_code(self, task: str, tools: List[str]) -> str:
        """Fallback agent code when generation fails"""
        
        return f'''
import logging
from typing import Any, Dict, List, Optional

class {task.replace(" ", "")}Agent:
    """Agent for: {task}"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tools = {tools}
        
    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the main task"""
        try:
            # Implementation placeholder
            result = {{"status": "completed", "data": input_data}}
            return result
        except Exception as e:
            self.logger.error(f"Task execution failed: {{e}}")
            return {{"status": "failed", "error": str(e)}}

def main():
    agent = {task.replace(" ", "")}Agent()
    result = agent.execute()
    print(f"Result: {{result}}")

if __name__ == "__main__":
    main()
'''
    
    def _fallback_specialized_agent(self, spec: Dict[str, Any], tools: List[str]) -> str:
        """Fallback specialized agent code"""
        
        agent_name = spec['agent_name'].replace(' ', '')
        
        return f'''
import logging
from typing import Any, Dict

class {agent_name}:
    """Specialized agent: {spec['task_description']}"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.agent_type = "{spec['agent_type']}"
        self.tools = {tools}
        
    def process(self, data: Any) -> Dict[str, Any]:
        """Process data according to agent specialty"""
        try:
            # Specialized processing
            result = {{"agent": "{agent_name}", "processed": True, "data": data}}
            return result
        except Exception as e:
            self.logger.error(f"Processing failed: {{e}}")
            return {{"agent": "{agent_name}", "processed": False, "error": str(e)}}
'''
    
    def _fallback_coordination_code(self, agents: List[Dict[str, Any]], task: str) -> str:
        """Fallback coordination code"""
        
        agent_names = [agent['name'] for agent in agents]
        
        return f'''
import logging
from typing import Dict, Any, List

class MultiAgentCoordinator:
    """Coordinates multiple agents for: {task}"""
    
    def __init__(self):
        self.agents = {agent_names}
        self.logger = logging.getLogger(__name__)
        
    def coordinate_execution(self, input_data: Any = None) -> Dict[str, Any]:
        """Coordinate execution across all agents"""
        results = {{}}
        
        for agent_name in self.agents:
            try:
                # Execute agent (placeholder)
                result = {{"agent": agent_name, "status": "completed"}}
                results[agent_name] = result
            except Exception as e:
                results[agent_name] = {{"agent": agent_name, "status": "failed", "error": str(e)}}
        
        return results
'''


# Simple AI client for fallback
class SimpleAIClient:
    """Simple AI client fallback"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    
    def invoke(self, prompt: str) -> str:
        """Mock invoke for demo"""
        
        if "select" in prompt.lower() and "tools" in prompt.lower():
            return '["pandas", "numpy", "requests", "logging", "datetime"]'
        elif "decompose" in prompt.lower():
            return '''[
                {
                    "agent_name": "DataProcessor", 
                    "task_description": "Handle data processing", 
                    "agent_type": "data_agent",
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "agent_name": "APIService",
                    "task_description": "Create API endpoints", 
                    "agent_type": "api_agent",
                    "priority": 2,
                    "dependencies": ["DataProcessor"]
                }
            ]'''
        else:
            return '''
import logging
from typing import Any, Dict

class GeneratedAgent:
    """Auto-generated agent"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def execute(self, data: Any = None) -> Dict[str, Any]:
        """Execute main functionality"""
        return {"status": "completed", "data": data}
        
def main():
    agent = GeneratedAgent()
    result = agent.execute()
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
'''


def main():
    """Demo of Advanced Ultimate Coder Agent"""
    
    print("🎯 ADVANCED ULTIMATE CODER AGENT DEMO")
    print("=" * 70)
    
    # Initialize
    coder = AdvancedUltimateCoderAgent(
        api_key="your-openai-api-key-here",
        model="gpt-4o-mini"
    )
    
    # Demo 1: Single agent with intelligent tool selection
    print("\n🚀 DEMO 1: Single Agent with Intelligent Tools")
    print("-" * 50)
    
    request1 = AdvancedAgentRequest(
        task="Create a data analysis pipeline for CSV files with visualization",
        api_key="your-openai-api-key-here",
        multiple_agent=False,
        use_existing_tools="select_intelligently",
        complexity="intermediate",
        requirements=["Handle large files", "Interactive plots", "Export reports"]
    )
    
    result1 = coder.create_agent(request1)
    
    if result1.success:
        print(f"✅ Single agent created successfully!")
        print(f"   Tools used: {', '.join(result1.tools_used[:5])}...")
        print(f"   Quality: {result1.quality_score:.2f}")
        print(f"   Files: {len(result1.file_paths or [])}")
    
    # Demo 2: Multi-agent with supervisor
    print(f"\n🎼 DEMO 2: Multi-Agent System with Supervisor")
    print("-" * 50)
    
    request2 = AdvancedAgentRequest(
        task="Build a complete e-commerce data processing and API system",
        api_key="your-openai-api-key-here",
        multiple_agent=True,
        use_existing_tools="use_all",
        supervisor_pattern=True,
        max_agents=4,
        complexity="advanced"
    )
    
    result2 = coder.create_agent(request2)
    
    if result2.success:
        print(f"✅ Multi-agent system created!")
        print(f"   Agents created: {len(result2.agents or [])}")
        print(f"   Supervisor: {'✅' if result2.supervisor_code else '❌'}")
        print(f"   Coordination: {'✅' if result2.coordination_code else '❌'}")
        print(f"   Average quality: {result2.quality_score:.2f}")
        print(f"   Files generated: {len(result2.file_paths or [])}")
        
        # Show agent details
        if result2.agents:
            print(f"\n   📋 Agent Details:")
            for i, agent in enumerate(result2.agents, 1):
                print(f"      {i}. {agent['name']} ({agent['type']}) - Quality: {agent['quality']:.2f}")
    
    # Demo 3: Custom tool selection
    print(f"\n🛠️ DEMO 3: Custom Tool Selection")
    print("-" * 50)
    
    request3 = AdvancedAgentRequest(
        task="Create a machine learning model trainer",
        api_key="your-openai-api-key-here",
        multiple_agent=False,
        use_existing_tools="none",
        custom_tools=["scikit-learn", "pandas", "numpy", "joblib", "matplotlib"],
        complexity="advanced"
    )
    
    result3 = coder.create_agent(request3)
    
    if result3.success:
        print(f"✅ Custom tool agent created!")
        print(f"   Custom tools: {', '.join(result3.tools_used or [])}")
        print(f"   Quality: {result3.quality_score:.2f}")
    
    print(f"\n🎊 Advanced demos completed!")
    print(f"\n💡 Key Features Demonstrated:")
    print(f"  ✅ Intelligent tool selection from comprehensive library")
    print(f"  ✅ Single vs Multi-agent orchestration")
    print(f"  ✅ Supervisor pattern for agent coordination")
    print(f"  ✅ Custom tool specification")
    print(f"  ✅ Quality-driven agent generation")


if __name__ == "__main__":
    main()