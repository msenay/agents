# 🧑‍💻 Software Development AI Agents

Pre-built, optimized AgentConfig presets for complete software development workflows. Each agent is specialized for specific development tasks and optimized for production use.

## 🚀 Quick Start

```python
from pre_built_configs import CODER_AGENT_CONFIG, ORCHESTRATOR_AGENT_CONFIG
from core_agent import CoreAgent
from langchain_openai import ChatOpenAI

# Create a coder agent
coder_config = CODER_AGENT_CONFIG
coder_config.model = ChatOpenAI(model="gpt-4o", temperature=0.1)
coder = CoreAgent(coder_config)

# Generate code
response = coder.invoke("Implement a binary search function in Python")
```

## 🎯 Available Development Agents

### 👨‍💻 **CODER_AGENT_CONFIG**
**Perfect for:** Code generation, refactoring, feature implementation, bug fixes

**Optimizations:**
- ✅ **32K context window** for large file analysis
- ✅ **PostgreSQL storage** for persistent code knowledge
- ✅ **Semantic search** for code pattern retrieval  
- ✅ **Memory tools** for learning coding patterns
- ✅ **No rate limiting** for intensive coding sessions

**Use Cases:**
- Feature implementation and enhancement
- Code refactoring and optimization
- Bug fixing and debugging
- API integration development
- Legacy code modernization

```python
from pre_built_configs import CODER_AGENT_CONFIG
```

---

### 🧪 **UNIT_TESTER_AGENT_CONFIG**
**Perfect for:** Test generation, coverage analysis, test optimization

**Optimizations:**
- ✅ **24K context window** for analyzing code + existing tests
- ✅ **InMemory backend** for fast test iteration
- ✅ **Memory tools** for test pattern recognition
- ✅ **No rate limiting** for rapid test generation
- ✅ **Test-focused evaluation** metrics

**Use Cases:**
- Unit test generation for new code
- Test coverage gap analysis
- Test case optimization and refactoring
- Edge case identification
- Test data generation

```python
from pre_built_configs import UNIT_TESTER_AGENT_CONFIG
```

---

### ⚡ **EXECUTER_AGENT_CONFIG**
**Perfect for:** Running tests, executing builds, environment management

**Optimizations:**
- ✅ **High throughput** (20 requests/sec) for rapid execution
- ✅ **Real-time streaming** for execution feedback
- ✅ **Session memory** for execution state tracking
- ✅ **InMemory backend** for fast cycles
- ✅ **Execution-focused evaluation** metrics

**Use Cases:**
- Running unit tests and test suites
- Executing build scripts and pipelines
- Environment setup and teardown
- Performance benchmarking
- Integration test execution

```python
from pre_built_configs import EXECUTER_AGENT_CONFIG
```

---

### 🔍 **CODE_REVIEWER_AGENT_CONFIG**
**Perfect for:** Code review, quality analysis, security auditing

**Optimizations:**
- ✅ **28K context window** for comprehensive review
- ✅ **PostgreSQL storage** for review knowledge persistence
- ✅ **Semantic search** for similar review cases
- ✅ **Controlled rate limiting** (2 req/sec) for thorough analysis
- ✅ **Quality-focused evaluation** metrics

**Use Cases:**
- Pull request review automation
- Code quality assessment
- Security vulnerability detection
- Best practice enforcement
- Architecture review and suggestions

```python
from pre_built_configs import CODE_REVIEWER_AGENT_CONFIG
```

---

### 🏗️ **BUILD_AGENT_CONFIG**
**Perfect for:** Build automation, CI/CD pipelines, versioning, GitHub operations

**Optimizations:**
- ✅ **Redis backend** for fast build state management
- ✅ **TTL support** (3 days) for build artifact cleanup
- ✅ **Session memory** for build pipeline tracking
- ✅ **Streaming output** for real-time build feedback
- ✅ **Build-focused evaluation** metrics

**Use Cases:**
- Automated build pipeline execution
- Version tagging and release management
- GitHub repository operations
- Docker image building and publishing
- Deployment automation

```python
from pre_built_configs import BUILD_AGENT_CONFIG
```

---

### 🎼 **ORCHESTRATOR_AGENT_CONFIG**
**Perfect for:** Workflow coordination, multi-agent supervision, task distribution

**Optimizations:**
- ✅ **Supervisor pattern** for agent coordination
- ✅ **Redis backend** for fast inter-agent communication
- ✅ **High coordination rate** (10 req/sec)
- ✅ **Session memory** for workflow state management
- ✅ **Real-time streaming** for workflow updates

**Use Cases:**
- Software development workflow coordination
- Multi-agent task distribution
- Build pipeline orchestration
- Quality assurance coordination
- Release management supervision

```python
from pre_built_configs import ORCHESTRATOR_AGENT_CONFIG
```

## 🔄 Complete Development Workflows

### 1. **Feature Development Pipeline**
```python
# Agents: Coder → Unit Tester → Code Reviewer → Executer
workflow = [
    "1. Coder Agent: Implement feature code",
    "2. Unit Tester Agent: Generate comprehensive tests", 
    "3. Code Reviewer Agent: Review code quality and security",
    "4. Executer Agent: Run tests and validate implementation"
]
```

### 2. **CI/CD Automation Pipeline**
```python
# Agents: Executer → Build → Orchestrator
workflow = [
    "1. Executer Agent: Run all tests and quality checks",
    "2. Build Agent: Create build artifacts and version tags",
    "3. Orchestrator Agent: Coordinate deployment process"
]
```

### 3. **Code Quality Assurance**
```python
# Agents: Code Reviewer → Unit Tester → Executer
workflow = [
    "1. Code Reviewer Agent: Static analysis and best practices",
    "2. Unit Tester Agent: Generate missing test cases", 
    "3. Executer Agent: Validate test coverage and performance"
]
```

### 4. **Legacy Code Refactoring**
```python
# Agents: Code Reviewer → Unit Tester → Coder → Executer
workflow = [
    "1. Code Reviewer Agent: Analyze current code structure",
    "2. Unit Tester Agent: Create safety net of tests",
    "3. Coder Agent: Implement refactoring changes",
    "4. Executer Agent: Validate refactoring success"
]
```

## 🎼 Multi-Agent Orchestration

### Setting up an Orchestrator with Multiple Agents

```python
from pre_built_configs import *
from core_agent import CoreAgent
from langchain_openai import ChatOpenAI

# Configure individual agents
model = ChatOpenAI(model="gpt-4o", temperature=0.1)

coder_config = CODER_AGENT_CONFIG
coder_config.model = model

tester_config = UNIT_TESTER_AGENT_CONFIG
tester_config.model = model

reviewer_config = CODE_REVIEWER_AGENT_CONFIG
reviewer_config.model = model

executer_config = EXECUTER_AGENT_CONFIG
executer_config.model = model

build_config = BUILD_AGENT_CONFIG
build_config.model = model

# Create orchestrator with supervised agents
orchestrator_config = ORCHESTRATOR_AGENT_CONFIG
orchestrator_config.model = model
orchestrator_config.agents = {
    "coder": coder_config,
    "unit_tester": tester_config,
    "code_reviewer": reviewer_config,
    "executer": executer_config,
    "build": build_config
}

# Create the orchestrator
orchestrator = CoreAgent(orchestrator_config)

# Coordinate development workflow
result = orchestrator.invoke("Implement user authentication feature with tests")
```

## 📊 Agent Configuration Comparison

| Agent | Backend | Context | Rate Limit | Key Features |
|-------|---------|---------|------------|--------------|
| **Coder** | PostgreSQL | 32K tokens | None | Large context, semantic search, knowledge persistence |
| **Unit Tester** | InMemory | 24K tokens | None | Fast iteration, pattern recognition |
| **Executer** | InMemory | 8K tokens | 20/s | High throughput, real-time feedback |
| **Code Reviewer** | PostgreSQL | 28K tokens | 2/s | Quality analysis, security detection |
| **Build** | Redis | 12K tokens | 5/s | CI/CD automation, TTL cleanup |
| **Orchestrator** | Redis | 8K tokens | 10/s | Multi-agent coordination |

## 🎯 Agent Selection Guide

### By Development Task
- **Code Generation** → `coder` (Large context, knowledge storage)
- **Test Writing** → `unit_tester` (Fast iteration, pattern recognition)
- **Code Review** → `code_reviewer` (Quality analysis, security)
- **Test Execution** → `executer` (High throughput, real-time feedback)
- **Build & Deploy** → `build` (CI/CD automation, version control)
- **Workflow Control** → `orchestrator` (Multi-agent supervision)

### By Context Requirements
- **Large Files (>20K tokens)** → `coder`, `unit_tester`, `code_reviewer`
- **Fast Execution** → `executer`, `unit_tester`
- **Persistent Knowledge** → `coder`, `code_reviewer` (PostgreSQL)
- **Real-time Feedback** → `executer`, `build`, `orchestrator`

### By Infrastructure Needs
- **No Dependencies** → `unit_tester`, `executer` (InMemory)
- **Redis Available** → `build`, `orchestrator`
- **PostgreSQL Available** → `coder`, `code_reviewer`
- **Full Infrastructure** → All agents with real backends

## 🛠️ Configuration Patterns

### Pattern 1: Single Agent Usage
```python
from pre_built_configs import CODER_AGENT_CONFIG
from core_agent import CoreAgent

config = CODER_AGENT_CONFIG
config.model = your_model
agent = CoreAgent(config)
```

### Pattern 2: Registry Access
```python
from pre_built_configs import get_config

config = get_config("coder")
config.model = your_model
agent = CoreAgent(config)
```

### Pattern 3: Custom Configuration
```python
from pre_built_configs import create_custom_config

config = create_custom_config(
    name="MyCoderAgent",
    max_tokens=16000,  # Smaller context
    memory_namespace="my_project"
)
config.model = your_model
agent = CoreAgent(config)
```

### Pattern 4: Environment-Specific Setup
```python
import os
from pre_built_configs import *

# Development environment
if os.getenv("ENV") == "development":
    config = get_config("development")
    
# Production with specific agents
elif os.getenv("ENV") == "production":
    if task == "coding":
        config = CODER_AGENT_CONFIG
    elif task == "review":
        config = CODE_REVIEWER_AGENT_CONFIG
```

## 🔧 Advanced Configuration

### Memory Configuration
```python
# PostgreSQL for persistent knowledge (Coder, Code Reviewer)
postgres_url = "postgresql://user:pass@localhost:5432/codeagent"

# Redis for fast communication (Build, Orchestrator)  
redis_url = "redis://localhost:6379"

# InMemory for development (Unit Tester, Executer)
# No external dependencies required
```

### Performance Tuning
```python
# High-performance coding
coder_config = CODER_AGENT_CONFIG
coder_config.max_tokens = 64000  # Even larger context
coder_config.memory_namespace = "high_perf_coding"

# Fast test execution
executer_config = EXECUTER_AGENT_CONFIG
executer_config.requests_per_second = 50.0  # Higher throughput
executer_config.max_bucket_size = 200.0  # Larger burst capacity
```

### Security & Compliance
```python
# Secure code review
reviewer_config = CODE_REVIEWER_AGENT_CONFIG
reviewer_config.evaluation_metrics.extend([
    "security_compliance", 
    "data_privacy", 
    "audit_trail"
])

# Controlled build environment
build_config = BUILD_AGENT_CONFIG
build_config.enable_human_feedback = True
build_config.interrupt_before = ["deploy_production"]
```

## 🚨 Common Issues & Solutions

### Issue 1: Database Connection Errors
```python
# Problem: PostgreSQL/Redis not available
# Solution: Configs automatically fallback to mock implementations

config = CODER_AGENT_CONFIG  # Uses PostgreSQL
# If PostgreSQL unavailable, automatically uses mock PostgreSQL store
```

### Issue 2: Large Context Memory Usage
```python
# Problem: 32K context uses too much memory
# Solution: Reduce context size for your use case

config = CODER_AGENT_CONFIG
config.max_tokens = 16000  # Reduce to 16K
config.trim_strategy = "first"  # Keep recent context
```

### Issue 3: Rate Limiting in Development
```python
# Problem: Rate limits slowing development
# Solution: Use development config or disable rate limiting

config = get_config("development")  # High rate limits
# Or disable completely:
config.enable_rate_limiting = False
```

## 📚 Real-World Examples

### Example 1: Automated Feature Development
```python
from pre_built_configs import *
from core_agent import CoreAgent

# Step 1: Generate feature code
coder = CoreAgent(CODER_AGENT_CONFIG)
coder.config.model = your_model

code_result = coder.invoke("""
Implement a REST API endpoint for user registration with:
- Email validation
- Password hashing
- Database integration
- Error handling
""")

# Step 2: Generate tests
tester = CoreAgent(UNIT_TESTER_AGENT_CONFIG)  
tester.config.model = your_model

test_result = tester.invoke(f"""
Generate comprehensive unit tests for this code:
{code_result['messages'][-1].content}
""")

# Step 3: Review code quality
reviewer = CoreAgent(CODE_REVIEWER_AGENT_CONFIG)
reviewer.config.model = your_model

review_result = reviewer.invoke(f"""
Review this code for security, best practices, and quality:
{code_result['messages'][-1].content}
""")
```

### Example 2: CI/CD Pipeline Automation
```python
# Step 1: Run all tests
executer = CoreAgent(EXECUTER_AGENT_CONFIG)
executer.config.model = your_model

test_results = executer.invoke("Run full test suite and generate coverage report")

# Step 2: Build and tag release  
builder = CoreAgent(BUILD_AGENT_CONFIG)
builder.config.model = your_model

build_result = builder.invoke("Build Docker image and tag as v1.2.3")

# Step 3: Coordinate deployment
orchestrator = CoreAgent(ORCHESTRATOR_AGENT_CONFIG)
orchestrator.config.model = your_model

deploy_result = orchestrator.invoke("Deploy v1.2.3 to staging environment")
```

### Example 3: Legacy Code Modernization
```python
# Create specialized modernization workflow
modernization_workflow = {
    "analyze": CODE_REVIEWER_AGENT_CONFIG,
    "test": UNIT_TESTER_AGENT_CONFIG, 
    "refactor": CODER_AGENT_CONFIG,
    "validate": EXECUTER_AGENT_CONFIG
}

for step, config in modernization_workflow.items():
    config.model = your_model
    config.memory_namespace = f"modernization_{step}"
    
# Execute modernization pipeline...
```

## 🎉 Summary

Software Development AI Agents provide:

- ✅ **Specialized Agents** - Each optimized for specific development tasks
- ✅ **Complete Workflows** - End-to-end development pipeline coverage
- ✅ **Production Ready** - Battle-tested configurations with proper fallbacks
- ✅ **Multi-Agent Coordination** - Orchestrator pattern for complex workflows
- ✅ **Flexible Configuration** - Easy customization for specific needs
- ✅ **Zero Dependencies** - InMemory fallbacks for development environments

Choose the agents that match your development workflow, configure your models, and start building with AI-powered development pipelines!

---

*For more examples and advanced usage, see the demo files and comprehensive test suites in the repository.*