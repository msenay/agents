# 🎯 Pre-Built Agent Configurations

Ready-to-use, optimized AgentConfig presets for common scenarios. Each configuration is carefully tuned for specific use cases and production environments.

## 🚀 Quick Start

```python
from pre_built_configs import HIGH_PERFORMANCE_CONFIG
from core_agent import CoreAgent
from langchain_openai import ChatOpenAI

# Use a pre-built config
config = HIGH_PERFORMANCE_CONFIG
config.model = ChatOpenAI(temperature=0.1)  # Add your model

# Create agent
agent = CoreAgent(config)
response = agent.invoke("Hello!")
```

## 📋 Available Configurations

### 🚀 Production Configurations

#### **HIGH_PERFORMANCE_CONFIG**
Perfect for: Production environments, high-throughput applications, enterprise use

**Features:**
- ✅ Redis for ultra-fast short-term memory and session sharing
- ✅ PostgreSQL for reliable long-term memory persistence  
- ✅ Aggressive rate limiting (5 req/sec) to prevent API abuse
- ✅ Message trimming for optimal token usage (8K context)
- ✅ AI summarization to preserve context efficiently
- ✅ Semantic search with embeddings for intelligent retrieval
- ✅ Memory tools for self-optimizing agents
- ✅ TTL support (48 hours) for automatic memory cleanup

**Use Cases:**
- Customer service chatbots with high volume
- Multi-tenant applications requiring isolation
- Enterprise AI assistants with complex workflows
- Production APIs serving thousands of users

```python
from pre_built_configs import HIGH_PERFORMANCE_CONFIG
```

#### **DISTRIBUTED_CONFIG**
Perfect for: Microservices, cloud-native apps, multi-region deployments

**Features:**
- ✅ Redis clustering for distributed short-term memory
- ✅ MongoDB for flexible, schema-less document storage
- ✅ Session-based memory for stateless microservices
- ✅ TTL support (24 hours) for automatic data expiration
- ✅ Supervisor pattern for service orchestration
- ✅ Rate limiting (3 req/sec) with shared state

**Use Cases:**
- Microservices architectures
- Multi-region deployments
- Cloud-native applications
- Container orchestration (Kubernetes)

```python
from pre_built_configs import DISTRIBUTED_CONFIG
```

#### **ENTERPRISE_CONFIG**
Perfect for: Large organizations, compliance requirements, audit trails, governance

**Features:**
- ✅ Multi-backend architecture (Redis + PostgreSQL) for redundancy
- ✅ Comprehensive audit trails and logging
- ✅ Strict rate limiting (1 req/sec) for resource control
- ✅ Session isolation for security
- ✅ Human oversight and approval workflows
- ✅ Evaluation for compliance monitoring
- ✅ TTL (7 days) for data governance with strict expiration

**Use Cases:**
- Enterprise AI assistants
- Compliance-sensitive applications
- Multi-departmental AI systems
- Audit-required environments
- Regulated industry applications

```python
from pre_built_configs import ENTERPRISE_CONFIG
```

### 🛠️ Development Configurations

#### **DEVELOPMENT_CONFIG**
Perfect for: Local development, testing, prototyping, debugging

**Features:**
- ✅ InMemory backend for fast iteration
- ✅ No external dependencies (Redis/PostgreSQL not required)
- ✅ Mock implementations for all features
- ✅ High rate limiting (10 req/sec) for development speed
- ✅ All memory features enabled for testing
- ✅ Comprehensive logging and debugging

**Development Benefits:**
- Zero infrastructure requirements
- Fast startup and teardown
- Deterministic behavior for testing
- Easy debugging and inspection

```python
from pre_built_configs import DEVELOPMENT_CONFIG
```

#### **TESTING_CONFIG**
Perfect for: Automated testing, CI/CD, quality assurance, benchmarking

**Features:**
- ✅ Deterministic behavior for consistent testing
- ✅ InMemory backend for isolated test environments
- ✅ All features enabled for comprehensive coverage
- ✅ No rate limiting for fast test execution
- ✅ Fixed session IDs for reproducible tests
- ✅ Evaluation hooks for automated quality assessment

**Testing Benefits:**
- Deterministic and repeatable results
- Fast execution without external dependencies
- Comprehensive feature coverage
- Easy assertion and validation

```python
from pre_built_configs import TESTING_CONFIG
```

### 🎯 Specialized Use Case Configurations

#### **CUSTOMER_SERVICE_CONFIG**
Perfect for: Customer support, help desks, service chatbots, FAQ systems

**Features:**
- ✅ Session memory for conversation continuity
- ✅ Semantic search for knowledge base integration
- ✅ Message trimming (6K context) to handle long conversations
- ✅ AI summarization for conversation history
- ✅ Memory tools for learning from interactions
- ✅ Human handoff support with interrupts
- ✅ Moderate rate limiting (2 req/sec) for customer comfort

**Customer Service Benefits:**
- Maintains conversation context across sessions
- Learns from previous interactions
- Handles escalations gracefully
- Provides consistent service quality

```python
from pre_built_configs import CUSTOMER_SERVICE_CONFIG
```

#### **RESEARCH_ASSISTANT_CONFIG**
Perfect for: Academic research, data analysis, literature reviews, knowledge work

**Features:**
- ✅ PostgreSQL for reliable long-term knowledge storage
- ✅ Advanced semantic search for literature and document retrieval
- ✅ Memory tools for building knowledge graphs
- ✅ No rate limiting for intensive research sessions
- ✅ Large context windows (16K) for document analysis
- ✅ Detailed AI summarization (512 tokens) for research synthesis

**Research Benefits:**
- Builds comprehensive knowledge over time
- Connects disparate information sources
- Maintains research methodology consistency
- Supports collaborative research projects

```python
from pre_built_configs import RESEARCH_ASSISTANT_CONFIG
```

#### **CREATIVE_ASSISTANT_CONFIG**
Perfect for: Content creation, creative writing, brainstorming, artistic projects

**Features:**
- ✅ InMemory for fast iteration and experimentation
- ✅ Session memory for creative project continuity
- ✅ Minimal constraints for creative freedom
- ✅ No rate limiting for rapid ideation
- ✅ Memory tools for inspiration and reference
- ✅ Large context (8K) for creative flow

**Creative Benefits:**
- Encourages free-flowing creative processes
- Maintains creative project context
- Supports iterative refinement
- Enables experimental approaches

```python
from pre_built_configs import CREATIVE_ASSISTANT_CONFIG
```

## 🎛️ Configuration Usage Patterns

### Pattern 1: Direct Usage
```python
from pre_built_configs import HIGH_PERFORMANCE_CONFIG
from core_agent import CoreAgent
from langchain_openai import ChatOpenAI

config = HIGH_PERFORMANCE_CONFIG
config.model = ChatOpenAI(temperature=0.1)
agent = CoreAgent(config)
```

### Pattern 2: Registry Access
```python
from pre_built_configs import get_config
from core_agent import CoreAgent

config = get_config("development")
config.model = your_model
agent = CoreAgent(config)
```

### Pattern 3: Custom Configuration
```python
from pre_built_configs import create_custom_config
from core_agent import CoreAgent

config = create_custom_config(
    name="MySpecialAgent",
    enable_rate_limiting=True,
    requests_per_second=5.0,
    max_tokens=4000,
    memory_namespace="my_project"
)
config.model = your_model
agent = CoreAgent(config)
```

### Pattern 4: Configuration Customization
```python
from pre_built_configs import DEVELOPMENT_CONFIG
from core_agent import CoreAgent

# Customize existing config
config = DEVELOPMENT_CONFIG
config.name = "MyDevAgent"
config.memory_namespace = "my_dev_project"
config.max_tokens = 4000
config.model = your_model

agent = CoreAgent(config)
```

## 🗃️ Configuration Registry

### List All Available Configs
```python
from pre_built_configs import list_configs

configs = list_configs()
for name, description in configs.items():
    print(f"{name}: {description}")
```

### Get Config by Name
```python
from pre_built_configs import get_config

# Get specific config
config = get_config("high_performance")

# Available config names:
# - "high_performance"
# - "distributed" 
# - "development"
# - "testing"
# - "customer_service"
# - "research_assistant"
# - "creative_assistant"
# - "enterprise"
```

## 📊 Configuration Comparison

| Configuration | Backend | Memory Types | Rate Limit | Key Features |
|--------------|---------|--------------|------------|--------------|
| **High Performance** | Redis | Short+Long+Session+Semantic | 5.0/s | Production, TTL, All features |
| **Distributed** | Redis | Short+Long+Session | 3.0/s | Microservices, MongoDB, TTL |
| **Development** | InMemory | Short+Long+Session+Semantic | 10.0/s | Local dev, All features |
| **Testing** | InMemory | Short+Long+Session+Semantic | None | Automated testing, Deterministic |
| **Customer Service** | Redis | Short+Long+Session+Semantic | 2.0/s | Support, Human handoff |
| **Research Assistant** | PostgreSQL | Long+Semantic | None | Research, Large context |
| **Creative Assistant** | InMemory | Short+Long+Session | None | Creative, No limits |
| **Enterprise** | Redis | Short+Long+Session+Semantic | 1.0/s | Compliance, Audit trails |

## 🎯 Selection Guide

### By Deployment Environment
- **Local Development** → `development`
- **Production Server** → `high_performance`
- **Microservices/Cloud** → `distributed`
- **Enterprise Environment** → `enterprise`

### By Use Case
- **Customer Support** → `customer_service`
- **Research & Analysis** → `research_assistant`
- **Creative Content** → `creative_assistant`
- **Automated Testing** → `testing`

### By Performance Requirements
- **High Throughput** → `high_performance`
- **Fast Development** → `development`
- **Compliance & Audit** → `enterprise`
- **Distributed Scaling** → `distributed`

## 🔧 Configuration Parameters

### Memory Configuration
```python
# Memory enable/disable master switch
enable_memory = True

# Memory types selection
memory_types = ["short_term", "long_term", "session", "semantic"]

# Backend selection
memory_backend = "redis"  # "inmemory", "redis", "postgres", "mongodb"

# Database connections
redis_url = "redis://localhost:6379"
postgres_url = "postgresql://user:pass@localhost:5432/db"
mongodb_url = "mongodb://localhost:27017/db"
```

### Rate Limiting Configuration
```python
# Rate limiting enable/disable
enable_rate_limiting = True

# Requests per second
requests_per_second = 5.0

# Burst capacity
max_bucket_size = 15.0

# Check interval
check_every_n_seconds = 0.1
```

### Context Management Configuration
```python
# Message trimming
enable_message_trimming = True
max_tokens = 8000
trim_strategy = "last"  # "first", "last"

# AI summarization
enable_summarization = True
max_summary_tokens = 256
summarization_trigger_tokens = 6000
```

### TTL Configuration
```python
# TTL enable (only for Redis/MongoDB)
enable_ttl = True
default_ttl_minutes = 2880  # 48 hours
refresh_on_read = True
```

## 🌟 Best Practices

### 1. Start with Pre-built Configs
```python
# ✅ Good: Start with appropriate preset
config = get_config("development")  # for local dev
config = get_config("high_performance")  # for production

# ❌ Avoid: Creating from scratch unnecessarily
config = AgentConfig()  # Too much manual configuration
```

### 2. Customize Minimally
```python
# ✅ Good: Minimal targeted customization
config = DEVELOPMENT_CONFIG
config.name = "MyAgent"
config.memory_namespace = "my_project"

# ❌ Avoid: Over-customization
config.memory_backend = "redis"  # Defeats purpose of dev config
```

### 3. Environment-Specific Configs
```python
# ✅ Good: Different configs for different environments
if environment == "development":
    config = DEVELOPMENT_CONFIG
elif environment == "production":
    config = HIGH_PERFORMANCE_CONFIG
elif environment == "testing":
    config = TESTING_CONFIG
```

### 4. Model Integration
```python
# ✅ Good: Always add your model
config = HIGH_PERFORMANCE_CONFIG
config.model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ❌ Avoid: Forgetting to set model
agent = CoreAgent(config)  # Will use default/mock model
```

## 🚨 Common Issues and Solutions

### Issue 1: Database Connection Errors
```python
# Problem: Redis/PostgreSQL not available
# Solution: Configs gracefully fallback to mock implementations

config = HIGH_PERFORMANCE_CONFIG  # Uses Redis
# If Redis unavailable, automatically uses mock Redis store
```

### Issue 2: Missing API Keys
```python
# Problem: OpenAI API key not set
# Solution: Configure embeddings properly

config = HIGH_PERFORMANCE_CONFIG
config.model = ChatOpenAI(api_key="your-key-here")
# Or use environment variable: OPENAI_API_KEY
```

### Issue 3: Rate Limiting in Development
```python
# Problem: Rate limiting slowing development
# Solution: Use development config or disable rate limiting

config = DEVELOPMENT_CONFIG  # High rate limits
# Or disable completely:
config.enable_rate_limiting = False
```

## 📚 Examples

### Example 1: E-commerce Customer Support
```python
from pre_built_configs import CUSTOMER_SERVICE_CONFIG
from core_agent import CoreAgent
from langchain_openai import ChatOpenAI

config = CUSTOMER_SERVICE_CONFIG
config.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
config.memory_namespace = "ecommerce_support"

agent = CoreAgent(config)

# Handle customer inquiry
response = agent.invoke("I need help with my order #12345")
print(response['messages'][-1].content)
```

### Example 2: Research Assistant
```python
from pre_built_configs import RESEARCH_ASSISTANT_CONFIG
from core_agent import CoreAgent

config = RESEARCH_ASSISTANT_CONFIG
config.model = your_model
config.memory_namespace = "climate_research"

agent = CoreAgent(config)

# Analyze research paper
response = agent.invoke("Analyze this climate change paper and extract key findings...")
```

### Example 3: Creative Writing Assistant
```python
from pre_built_configs import CREATIVE_ASSISTANT_CONFIG
from core_agent import CoreAgent

config = CREATIVE_ASSISTANT_CONFIG
config.model = your_model
config.memory_namespace = "novel_writing"

agent = CoreAgent(config)

# Creative writing session
response = agent.invoke("Help me develop a character for my sci-fi novel...")
```

## 🎉 Summary

Pre-built configurations provide:

- ✅ **Instant Setup** - Ready-to-use optimized configs
- ✅ **Best Practices** - Carefully tuned parameters
- ✅ **Flexibility** - Easy customization and extension
- ✅ **Production Ready** - Battle-tested configurations
- ✅ **Zero Config** - Sensible defaults for common scenarios

Choose the configuration that matches your use case, add your model, and start building immediately!

---

*For more information, see the complete API documentation and examples in the repository.*