# 🎯 Advanced Ultimate Coder Agent

**Süper güçlü AI agent sistem** - İstediğin seçeneklerle perfect agent'lar yaratır!

## 🚀 **YENİ GELİŞMİŞ ÖZELLİKLER**

### ⚡ **Single vs Multi-Agent Creation**
```python
# Single optimized agent
multiple_agent=False  → Tek güçlü agent

# Multi-agent orchestrated system  
multiple_agent=True   → Birden fazla agent + supervisor
```

### 🛠️ **Intelligent Tool Selection**
```python
# LLM selects best tools for task
use_existing_tools="select_intelligently"

# Use all 60+ available tools
use_existing_tools="use_all"

# Use only your specified tools
use_existing_tools="none" + custom_tools=["pandas", "fastapi"]
```

### 🎼 **Supervisor Pattern**
```python
# Multi-agent coordination
supervisor_pattern=True  → Creates supervisor for agent coordination
```

## 🎯 **Quick Start**

### Basic Usage

```python
from advanced_ultimate_coder import AdvancedUltimateCoderAgent, AdvancedAgentRequest

# Initialize agent
coder = AdvancedUltimateCoderAgent(api_key="your-openai-key")

# Single agent with smart tool selection
request = AdvancedAgentRequest(
    task="Create a PDF processor with text extraction",
    api_key="your-openai-key",
    multiple_agent=False,
    use_existing_tools="select_intelligently",
    complexity="intermediate"
)

result = coder.create_agent(request)
```

### Multi-Agent System

```python
# Multi-agent system with supervisor
request = AdvancedAgentRequest(
    task="Build a complete data science workflow",
    api_key="your-openai-key",
    multiple_agent=True,
    use_existing_tools="use_all",
    supervisor_pattern=True,
    max_agents=3,
    complexity="advanced"
)

result = coder.create_agent(request)
```

## 🛠️ **60+ Available Tools**

### 📊 **Data Processing**
- **pandas**: DataFrame manipulation, CSV/Excel processing
- **numpy**: Numerical computing, mathematical operations
- **polars**: Fast DataFrame library, memory efficient
- **dask**: Parallel computing, big data processing
- **scipy**: Scientific computing, statistics
- **matplotlib**: Data visualization, plotting
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations

### 🌐 **Web Development**
- **requests**: HTTP requests, API calls
- **beautifulsoup4**: HTML/XML parsing, web scraping
- **selenium**: Browser automation, dynamic scraping
- **scrapy**: Web crawling framework
- **flask**: Lightweight web framework
- **fastapi**: Modern, high-performance web framework
- **django**: Full-featured web framework
- **aiohttp**: Asynchronous HTTP client/server

### 🤖 **Machine Learning**
- **scikit-learn**: Machine learning algorithms
- **tensorflow**: Deep learning, neural networks
- **pytorch**: Deep learning, research
- **xgboost**: Gradient boosting, tabular data
- **transformers**: NLP models, BERT, GPT
- **spacy**: NLP processing, NER
- **nltk**: Natural language processing

### 🗄️ **Database & Storage**
- **sqlalchemy**: SQL toolkit, ORM
- **pymongo**: MongoDB driver, NoSQL
- **redis-py**: Redis client, caching
- **psycopg2**: PostgreSQL adapter
- **sqlite3**: Lightweight SQL database
- **elasticsearch**: Search engine, full-text search

### 📁 **File Processing**
- **openpyxl**: Excel file manipulation
- **python-docx**: Word document processing
- **PyPDF2**: PDF file manipulation
- **pillow**: Image processing
- **pathlib**: File system operations
- **zipfile**: Archive creation/extraction

### ☁️ **API Integration**
- **openai**: OpenAI API client
- **boto3**: AWS SDK, cloud services
- **google-cloud**: Google Cloud Platform
- **stripe**: Payment processing
- **sendgrid**: Email delivery
- **twilio**: SMS and voice communication

### ⚙️ **System & Utilities**
- **psutil**: System monitoring, processes
- **schedule**: Job scheduling, automation
- **click**: Command-line interfaces
- **configparser**: Configuration files
- **logging**: Application logging
- **pytest**: Testing framework
- **black**: Code formatting

## 📊 **Tool Selection Strategies**

### 1. **Intelligent Selection** (Recommended)
```python
use_existing_tools="select_intelligently"
```
- 🧠 **LLM analyzes your task**
- 🎯 **Selects 3-8 most relevant tools**
- ⚡ **Optimized for task requirements**
- 📈 **Best quality/performance ratio**

### 2. **Use All Tools**
```python
use_existing_tools="use_all"
```
- 🌍 **All 60+ tools available**
- 💪 **Maximum capabilities**
- 🔧 **Good for complex multi-feature tasks**
- ⏱️ **Longer generation time**

### 3. **Custom Tool Selection**
```python
use_existing_tools="none"
custom_tools=["fastapi", "sqlalchemy", "redis-py"]
```
- 🎯 **Exact control over tools**
- 🚀 **Fastest generation**
- 📦 **Minimal dependencies**
- 🎨 **Perfect for specific requirements**

## 🎼 **Multi-Agent Architecture**

### Automatic Task Decomposition
```python
multiple_agent=True
```

**System automatically:**
1. 🔄 **Analyzes complex task**
2. 🎯 **Decomposes into specialized sub-tasks**
3. 👥 **Creates specialized agents**
4. 🎼 **Generates supervisor for coordination**
5. 🤝 **Sets up inter-agent communication**

### Example Multi-Agent System
```python
# Task: "Build e-commerce data processing system"
# Results in:

DataProcessor Agent:
  - Handle data ingestion and cleaning
  - Tools: pandas, numpy, dask

APIService Agent:  
  - Create REST API endpoints
  - Tools: fastapi, sqlalchemy, pydantic

AnalyticsEngine Agent:
  - Generate reports and insights
  - Tools: matplotlib, seaborn, plotly

Supervisor Agent:
  - Coordinates all agents
  - Manages data flow
  - Handles error recovery
```

## 🎯 **Real-World Examples**

### 1. **PDF Processing System**
```python
request = AdvancedAgentRequest(
    task="Create PDF processor with text and image extraction",
    multiple_agent=False,
    use_existing_tools="select_intelligently",
    requirements=[
        "Handle multiple PDF formats",
        "Extract text and images separately",
        "Export to JSON format",
        "Progress tracking"
    ]
)
```

**Result:** Single optimized agent with `PyPDF2`, `pillow`, `pathlib`, `json`, `logging`

### 2. **Data Science Workflow**
```python
request = AdvancedAgentRequest(
    task="Build comprehensive data science workflow",
    multiple_agent=True,
    use_existing_tools="use_all",
    supervisor_pattern=True,
    max_agents=4
)
```

**Result:** Multi-agent system with:
- **DataIngestor**: Handles data loading (`pandas`, `sqlalchemy`)
- **DataCleaner**: Preprocessing and cleaning (`numpy`, `scipy`)
- **ModelTrainer**: ML training (`scikit-learn`, `xgboost`)
- **ReportGenerator**: Visualization (`matplotlib`, `plotly`)
- **Supervisor**: Orchestrates entire workflow

### 3. **Authentication API**
```python
request = AdvancedAgentRequest(
    task="Create high-performance authentication API",
    multiple_agent=False,
    use_existing_tools="none",
    custom_tools=["fastapi", "sqlalchemy", "redis-py", "jwt", "bcrypt"]
)
```

**Result:** Focused API agent with exactly specified tools

## 📊 **Result Structure**

```python
@dataclass
class AdvancedAgentResult:
    success: bool
    task_id: str
    
    # Single agent
    agent_code: str
    agent_type: str
    
    # Multi-agent
    agents: List[Dict[str, Any]]
    supervisor_code: str
    coordination_code: str
    
    # Metadata
    tools_used: List[str]
    tools_selected_by: str  # "llm", "all", "user"
    quality_score: float    # 0.0-1.0
    complexity_score: float # 0.0-1.0
    creation_time: float
    file_paths: List[str]
```

## ⚙️ **Configuration Options**

### AdvancedAgentRequest Parameters

```python
@dataclass
class AdvancedAgentRequest:
    task: str                           # Main task description
    api_key: str                        # OpenAI API key
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    complexity: str = "intermediate"    # basic, intermediate, advanced, expert
    
    # Core options
    multiple_agent: bool = False
    use_existing_tools: str = "select_intelligently"
    custom_tools: Optional[List[str]] = None
    requirements: Optional[List[str]] = None
    
    # Multi-agent options
    max_agents: int = 5
    supervisor_pattern: bool = True
    agent_collaboration: bool = True
```

### Complexity Levels

- **`basic`**: Simple functionality, minimal dependencies
- **`intermediate`**: Standard features, good structure
- **`advanced`**: Complex logic, multiple components
- **`expert`**: Highly sophisticated, production-grade

## 🎨 **Usage Patterns**

### Pattern 1: **Single Optimized Agent**
```python
# Perfect for focused tasks
request = AdvancedAgentRequest(
    task="Create a log analyzer",
    multiple_agent=False,
    use_existing_tools="select_intelligently"
)
```

### Pattern 2: **Multi-Agent System**
```python
# Perfect for complex workflows
request = AdvancedAgentRequest(
    task="Build complete ML pipeline",
    multiple_agent=True,
    supervisor_pattern=True,
    max_agents=4
)
```

### Pattern 3: **Custom Tool Control**
```python
# Perfect for specific requirements
request = AdvancedAgentRequest(
    task="Create FastAPI microservice",
    multiple_agent=False,
    use_existing_tools="none",
    custom_tools=["fastapi", "pydantic", "sqlalchemy"]
)
```

## 🚀 **Installation & Setup**

### Requirements
```bash
# Core dependencies
pip install openai langchain-openai

# Optional for enhanced memory
pip install redis

# Tool-specific dependencies (install as needed)
pip install pandas numpy matplotlib      # Data tools
pip install requests beautifulsoup4     # Web tools  
pip install scikit-learn tensorflow     # ML tools
pip install fastapi uvicorn             # API tools
```

### Environment Setup
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Or pass directly
coder = AdvancedUltimateCoderAgent(api_key="your-api-key")
```

## 📈 **Performance Benchmarks**

### Tool Selection Performance
- **Intelligent Selection**: ~2-3 seconds
- **Use All Tools**: ~1 second  
- **Custom Tools**: ~0.5 seconds

### Agent Creation Times
- **Single Agent**: 5-15 seconds
- **Multi-Agent (2-3 agents)**: 15-30 seconds
- **Multi-Agent (4-5 agents)**: 30-60 seconds

### Quality Scores
- **Single Optimized**: 0.8-0.95
- **Multi-Agent Average**: 0.75-0.90
- **Custom Tools**: 0.85-0.95

## 🎯 **Best Practices**

### 1. **Choose Right Mode**
```python
# Simple, focused task → Single agent
task = "Create a CSV parser"
multiple_agent = False

# Complex, multi-step workflow → Multi-agent
task = "Build complete data processing pipeline"
multiple_agent = True
```

### 2. **Tool Selection Strategy**
```python
# Don't know which tools → Intelligent selection
use_existing_tools = "select_intelligently"

# Want maximum capabilities → Use all
use_existing_tools = "use_all"

# Know exactly what you need → Custom
use_existing_tools = "none"
custom_tools = ["specific", "tools", "only"]
```

### 3. **Task Description**
```python
# ✅ Good: Specific and clear
task = "Create a web scraper for e-commerce product data with rate limiting and CSV export"

# ❌ Avoid: Vague and overly broad
task = "Create a web thing that does stuff"
```

## 🎉 **Quick Demo**

```bash
# Run interactive examples
python simple_advanced_usage.py

# Available examples:
#   1. Single Agent (Smart Tools)
#   2. Multi-Agent System  
#   3. Custom Tool Selection
#   4. Strategy Comparison
#   5. Interactive Creation
```

## 🔧 **Advanced Features**

### Batch Agent Creation
```python
# Create multiple related agents
tasks = [
    "Create data processor",
    "Create API service", 
    "Create reporting tool"
]

# All agents learn from each other during creation
results = []
for task in tasks:
    request = AdvancedAgentRequest(task=task, ...)
    result = coder.create_agent(request)
    results.append(result)
```

### Agent Coordination
```python
# Multi-agent result includes coordination code
if result.success and result.coordination_code:
    # Use coordination system to manage agents
    exec(result.coordination_code)
```

### File Management
```python
# All generated files are automatically saved
if result.success:
    for file_path in result.file_paths:
        print(f"Generated: {file_path}")
        
    # Files include:
    # - Individual agent codes
    # - Supervisor code (if multi-agent)
    # - Coordination system
    # - Documentation
```

## 🎊 **Summary**

**Advanced Ultimate Coder Agent** provides:

- ✅ **Flexible Agent Creation** - Single vs Multi-agent
- ✅ **Intelligent Tool Selection** - 60+ tools, AI-driven selection  
- ✅ **Supervisor Pattern** - Multi-agent coordination
- ✅ **Custom Tool Control** - Exact tool specification
- ✅ **Quality Assessment** - Automated code quality scoring
- ✅ **Production Ready** - Complete, runnable agent code
- ✅ **File Management** - Automatic saving with metadata
- ✅ **Scalable Architecture** - From simple scripts to complex systems

### **İstediğin seçeneklerle perfect agent yaratır:**

1. **`multiple_agent=False`** → Single optimized agent
2. **`multiple_agent=True`** → Multi-agent system + supervisor  
3. **`use_existing_tools="select_intelligently"`** → LLM picks best tools
4. **`use_existing_tools="use_all"`** → All tools available
5. **`use_existing_tools="none" + custom_tools`** → Exact tool control

**Task veriyorsun → İstediğin şekilde agent alıyorsun!** 🚀

---

*Advanced Ultimate Coder Agent ile AI-powered development workflow'una geç!*