# Multiple LLMs with CoreAgent Framework 🤖

## 🎯 Overview

**EVET, kesinlikle!** CoreAgent framework'ünde her agent için farklı LLM'ler kullanabilirsiniz. Bu size şu imkanları sunar:

- ✅ **Default LLM** tanımlama
- ✅ **Her agent için özel LLM** seçimi
- ✅ **Farklı model türleri** (GPT-4, GPT-3.5, Claude, vs.)
- ✅ **Farklı temperature settings** (creative vs analytical)
- ✅ **Farklı Azure deployments**
- ✅ **Farklı token limits**
- ✅ **Multi-agent orchestration** with mixed LLMs

## 🔧 How It Works

### CoreAgent Model Architecture

Her `CoreAgent` kendi `AgentConfig`'ine sahiptir ve her config kendi `model` parametresini içerir:

```python
@dataclass
class AgentConfig:
    name: str = "CoreAgent"
    model: Optional[BaseChatModel] = None  # ← Her agent'ın kendi LLM'i
    system_prompt: str = "You are a helpful AI assistant."
    tools: List[BaseTool] = field(default_factory=list)
    # ... other parameters
```

### Factory Functions Support

Tüm factory functions farklı LLM'leri destekler:

```python
def create_simple_agent(model: BaseChatModel, ...)  # ← Her çağrıda farklı model
def create_advanced_agent(model: BaseChatModel, ...)
def create_supervisor_agent(model: BaseChatModel, ...)
```

## 🏗️ Implementation Examples

### 1. **Different Models Setup**

```python
from langchain_openai import AzureChatOpenAI

# DEFAULT LLM - General purpose
default_llm = AzureChatOpenAI(
    azure_deployment="gpt4",
    temperature=0.1,
    max_tokens=2000
)

# CREATIVE LLM - High creativity
creative_llm = AzureChatOpenAI(
    azure_deployment="gpt4",
    temperature=0.8,  # ← High creativity
    max_tokens=2500
)

# ANALYTICAL LLM - Precise analysis
analytical_llm = AzureChatOpenAI(
    azure_deployment="gpt4",
    temperature=0.0,  # ← Maximum precision
    max_tokens=3000
)

# FAST LLM - Quick responses
fast_llm = AzureChatOpenAI(
    azure_deployment="gpt35-turbo",  # ← Different model
    temperature=0.2,
    max_tokens=1000
)
```

### 2. **Creating Agents with Different LLMs**

```python
# Each agent gets its specialized LLM
analytical_agent = create_advanced_agent(
    model=analytical_llm,  # ← Analytical LLM (temp=0.0)
    name="Data Analyst",
    tools=[analyze_data],
    system_prompt="You are a precise data analyst..."
)

creative_agent = create_advanced_agent(
    model=creative_llm,  # ← Creative LLM (temp=0.8)
    name="Creative Writer",
    tools=[generate_content],
    system_prompt="You are a creative content generator..."
)

fast_agent = create_simple_agent(
    model=fast_llm,  # ← Fast LLM (GPT-3.5)
    name="Quick Assistant",
    tools=[execute_simple_tasks],
    system_prompt="You are a rapid-response assistant..."
)

default_agent = create_simple_agent(
    model=default_llm,  # ← Default LLM (balanced)
    name="General Assistant",
    system_prompt="You are a general-purpose assistant..."
)
```

### 3. **Multi-Agent Orchestration with Mixed LLMs**

```python
# Create supervisor with its own specialized LLM
supervisor_llm = AzureChatOpenAI(
    azure_deployment="gpt4",
    temperature=0.1,
    max_tokens=1500
)

# Supervisor orchestrates agents with different LLMs
supervisor = create_supervisor_agent(
    model=supervisor_llm,  # ← Supervisor's own LLM
    agents={
        "analytical": analytical_agent,  # ← Uses analytical_llm
        "creative": creative_agent,      # ← Uses creative_llm
        "fast": fast_agent,             # ← Uses fast_llm
        "default": default_agent        # ← Uses default_llm
    },
    system_prompt="""You orchestrate agents with different LLM capabilities:
    - analytical: GPT-4 with temp=0.0 for precise analysis
    - creative: GPT-4 with temp=0.8 for creative tasks
    - fast: GPT-3.5 for quick responses
    - default: GPT-4 balanced for general tasks
    
    Choose the right agent based on task requirements!"""
)
```

## 🎨 Use Cases for Different LLMs

### **Analytical Tasks** (Temperature: 0.0)
```python
analytical_llm = AzureChatOpenAI(temperature=0.0)
# Perfect for: Data analysis, fact-checking, mathematical calculations
```

### **Creative Tasks** (Temperature: 0.7-0.9)
```python
creative_llm = AzureChatOpenAI(temperature=0.8)
# Perfect for: Content creation, storytelling, brainstorming
```

### **Quick Tasks** (Fast Model)
```python
fast_llm = AzureChatOpenAI(azure_deployment="gpt35-turbo")
# Perfect for: Simple Q&A, basic calculations, routine tasks
```

### **Balanced Tasks** (Temperature: 0.1-0.3)
```python
default_llm = AzureChatOpenAI(temperature=0.1)
# Perfect for: General assistance, mixed task types
```

### **Code Generation** (Temperature: 0.0-0.2)
```python
coding_llm = AzureChatOpenAI(temperature=0.1, max_tokens=4000)
# Perfect for: Code writing, debugging, technical documentation
```

## 🚀 Advanced Patterns

### **1. Dynamic Model Selection**

```python
def get_llm_for_task(task_type: str) -> BaseChatModel:
    """Dynamically select LLM based on task type"""
    llm_map = {
        "analysis": analytical_llm,
        "creative": creative_llm,
        "quick": fast_llm,
        "coding": coding_llm
    }
    return llm_map.get(task_type, default_llm)

# Create agent with task-appropriate LLM
task_type = "analysis"
agent = create_simple_agent(
    model=get_llm_for_task(task_type),
    name=f"{task_type.title()} Agent"
)
```

### **2. Model Pool Management**

```python
class LLMPool:
    """Manage multiple LLMs for different purposes"""
    
    def __init__(self):
        self.models = {
            "gpt4_precise": AzureChatOpenAI(temperature=0.0),
            "gpt4_creative": AzureChatOpenAI(temperature=0.8),
            "gpt4_balanced": AzureChatOpenAI(temperature=0.1),
            "gpt35_fast": AzureChatOpenAI(azure_deployment="gpt35-turbo"),
        }
    
    def get(self, model_type: str) -> BaseChatModel:
        return self.models.get(model_type, self.models["gpt4_balanced"])
    
    def create_agent(self, model_type: str, **kwargs) -> CoreAgent:
        return create_simple_agent(
            model=self.get(model_type),
            **kwargs
        )

# Usage
llm_pool = LLMPool()

data_agent = llm_pool.create_agent("gpt4_precise", name="Data Analyst")
creative_agent = llm_pool.create_agent("gpt4_creative", name="Content Creator")
quick_agent = llm_pool.create_agent("gpt35_fast", name="Quick Assistant")
```

### **3. Environment-Based Model Selection**

```python
def create_environment_optimized_agent(env: str) -> CoreAgent:
    """Create agent optimized for different environments"""
    
    if env == "development":
        # Use fast, cheaper model for dev
        model = AzureChatOpenAI(azure_deployment="gpt35-turbo")
    elif env == "production":
        # Use best model for production
        model = AzureChatOpenAI(azure_deployment="gpt4", temperature=0.1)
    elif env == "creative":
        # Use creative settings
        model = AzureChatOpenAI(azure_deployment="gpt4", temperature=0.8)
    else:
        # Default fallback
        model = AzureChatOpenAI(azure_deployment="gpt4", temperature=0.1)
    
    return create_simple_agent(model=model, name=f"{env.title()} Agent")
```

## 📊 Performance Considerations

### **Cost Optimization**
```python
# Use cheaper models for simple tasks
simple_tasks_llm = AzureChatOpenAI(azure_deployment="gpt35-turbo")

# Use premium models only for complex tasks
complex_tasks_llm = AzureChatOpenAI(azure_deployment="gpt4")
```

### **Speed Optimization**
```python
# Fast model with lower token limits for quick responses
fast_llm = AzureChatOpenAI(
    azure_deployment="gpt35-turbo",
    max_tokens=500,  # ← Faster responses
    temperature=0.1
)
```

### **Quality Optimization**
```python
# Premium model with higher token limits for detailed work
quality_llm = AzureChatOpenAI(
    azure_deployment="gpt4",
    max_tokens=4000,  # ← Detailed responses
    temperature=0.1
)
```

## 🎯 Test Results Summary

✅ **CONFIRMED WORKING FEATURES:**

1. **Multiple LLM Support** - Each agent can have different LLM ✅
2. **Temperature Variations** - Different creativity levels per agent ✅
3. **Model Type Variations** - GPT-4, GPT-3.5, etc. per agent ✅
4. **Token Limit Variations** - Different max_tokens per agent ✅
5. **Multi-Agent Orchestration** - Supervisor with mixed-LLM agents ✅
6. **Factory Function Support** - All creation patterns support custom LLMs ✅

### **Example from Live Test:**

```
✅ Analytical Agent (temp=0.0) - Created successfully
✅ Creative Agent (temp=0.8) - Created successfully  
✅ Default Agent (balanced) - Created successfully
✅ Supervisor Agent - Successfully orchestrating mixed-LLM team
```

## 🏆 Best Practices

### **1. Match LLM to Task Type**
- **Analytical tasks** → `temperature=0.0`
- **Creative tasks** → `temperature=0.7-0.9`
- **Code generation** → `temperature=0.1-0.2`
- **General tasks** → `temperature=0.1-0.3`

### **2. Use Model Hierarchy**
- **Simple tasks** → GPT-3.5 Turbo (cost-effective)
- **Complex tasks** → GPT-4 (high quality)
- **Specialized tasks** → Fine-tuned models

### **3. Resource Management**
- **Development** → Cheaper models
- **Production** → Premium models
- **Batch processing** → Optimized for throughput
- **Real-time** → Optimized for latency

### **4. Memory and Tools**
```python
# Different agents can have different capabilities
agent1 = create_advanced_agent(
    model=analytical_llm,
    enable_memory=True,    # ← Memory enabled
    enable_evaluation=True  # ← Evaluation enabled
)

agent2 = create_simple_agent(
    model=fast_llm,
    enable_memory=False    # ← No memory for speed
)
```

## 🎉 Conclusion

**CoreAgent framework perfect olarak multiple LLM desteği sağlıyor!**

- ✅ **Her agent farklı LLM kullanabilir**
- ✅ **Default LLM tanımlayabilirsiniz**
- ✅ **Temperature, model, token limits tamamen customizable**
- ✅ **Multi-agent orchestration mixed LLMs ile çalışıyor**
- ✅ **Cost ve performance optimization mümkün**

Bu size maximum flexibility sağlıyor - her agent'ı tam olarak ihtiyacına göre optimize edebilirsiniz! 🚀