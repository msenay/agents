# Core Agent Oluşturma Rehberi

## 🚀 Hızlı Başlangıç

Core Agent oluşturmak çok basit! 3 temel yöntem var:

## 1. En Basit Yöntem - Minimal Agent

```python
from core import CoreAgent, AgentConfig
from langchain_openai import ChatOpenAI

# Model
model = ChatOpenAI(model="gpt-4o-mini")

# Config
config = AgentConfig(
    name="MyAgent",
    model=model,
    system_prompt="Sen yardımsever bir asistansın."
)

# Agent oluştur
agent = CoreAgent(config)

# Kullan
response = agent.invoke("Merhaba!")
```

## 2. Tool'lu Agent

```python
from core import CoreAgent, AgentConfig
from langchain_core.tools import tool

# Custom tool tanımla
@tool
def calculator(expression: str) -> str:
    """Matematik işlemlerini yapar"""
    try:
        return str(eval(expression))
    except:
        return "Hata: Geçersiz işlem"

# Config
config = AgentConfig(
    name="MathAgent",
    model=model,
    system_prompt="Sen bir matematik asistanısın.",
    tools=[calculator]  # Tool'ları ekle
)

agent = CoreAgent(config)
```

## 3. Memory'li Agent

```python
config = AgentConfig(
    name="MemoryAgent",
    model=model,
    system_prompt="Sen hafızası olan bir asistansın.",
    
    # Memory ayarları
    enable_memory=True,
    memory_backend="inmemory",  # veya "redis", "postgres"
    memory_types=["short_term", "long_term"]
)

agent = CoreAgent(config)

# Thread memory kullanımı
response = agent.invoke(
    "Benim adım Ali",
    config={"configurable": {"thread_id": "user_123"}}
)
```

## 📋 Config Parametreleri (Sadece İhtiyacın Olanları Kullan!)

### Temel Parametreler
```python
config = AgentConfig(
    # Zorunlu
    name="AgentName",           # Agent ismi
    model=model,                # LLM model
    
    # Opsiyonel
    system_prompt="...",        # System prompt
    tools=[],                   # Tool listesi
    description="..."           # Agent açıklaması
)
```

### Memory Parametreleri
```python
config = AgentConfig(
    # Memory'yi aç
    enable_memory=True,
    
    # Backend seç (birini)
    memory_backend="inmemory",  # "redis", "postgres"
    
    # Memory türleri (istediğini seç)
    memory_types=["short_term", "long_term", "session", "semantic"],
    
    # Backend URL'leri (sadece gerekirse)
    redis_url="redis://localhost:6379",
    postgres_url="postgresql://user:pass@localhost:5432/db"
)
```

### Gelişmiş Özellikler
```python
config = AgentConfig(
    # Rate limiting
    enable_rate_limiting=True,
    requests_per_second=2.0,
    
    # Human feedback
    enable_human_feedback=True,
    interrupt_before=["tool_call"],
    
    # Streaming
    enable_streaming=True,
    
    # Message trimming
    enable_message_trimming=True,
    max_tokens=4000
)
```

## 🎯 Pratik Örnekler

### 1. Chatbot Agent
```python
def create_chatbot():
    return CoreAgent(AgentConfig(
        name="Chatbot",
        model=ChatOpenAI(model="gpt-4o-mini"),
        system_prompt="Sen samimi ve yardımsever bir chatbotsun.",
        enable_memory=True,
        memory_backend="inmemory"
    ))

chatbot = create_chatbot()
response = chatbot.invoke("Nasılsın?")
```

### 2. Coder Agent
```python
from core.tools import create_python_coding_tools

def create_coder():
    return CoreAgent(AgentConfig(
        name="PythonCoder",
        model=ChatOpenAI(model="gpt-4"),
        system_prompt="Sen uzman bir Python geliştiricisisin.",
        tools=create_python_coding_tools(),
        enable_memory=True,
        memory_types=["short_term", "long_term"]
    ))

coder = create_coder()
response = coder.invoke("Fibonacci fonksiyonu yaz")
```

### 3. Research Agent
```python
from langchain_community.tools import TavilySearchResults

def create_researcher():
    search_tool = TavilySearchResults()
    
    return CoreAgent(AgentConfig(
        name="Researcher",
        model=ChatOpenAI(model="gpt-4"),
        system_prompt="Sen detaylı araştırma yapan bir asistansın.",
        tools=[search_tool],
        enable_memory=True,
        memory_types=["short_term", "semantic"],  # Semantic search
        memory_backend="postgres",  # pgvector ile
        embedding_model="openai:text-embedding-3-small"
    ))
```

### 4. Multi-Agent System
```python
# Agent'ları oluştur
coder = create_coder()
tester = create_tester()

# Supervisor config
supervisor_config = AgentConfig(
    name="Supervisor",
    model=ChatOpenAI(model="gpt-4"),
    enable_supervisor=True,
    agents={
        "coder": coder,
        "tester": tester
    }
)

supervisor = CoreAgent(supervisor_config)
```

## 💡 İpuçları

### 1. Başlangıç için Minimal Config Kullan
```python
# ❌ Karmaşık
config = AgentConfig(
    name="MyAgent",
    model=model,
    enable_memory=True,
    memory_backend="redis",
    redis_url="...",
    enable_rate_limiting=True,
    requests_per_second=5,
    enable_evaluation=True,
    # ... 20 parametre daha
)

# ✅ Basit başla
config = AgentConfig(
    name="MyAgent",
    model=model,
    system_prompt="Yardımsever bir asistansın."
)
```

### 2. Factory Pattern Kullan
```python
class AgentFactory:
    @staticmethod
    def create_chatbot(name: str = "Chatbot"):
        return CoreAgent(AgentConfig(
            name=name,
            model=ChatOpenAI(model="gpt-4o-mini"),
            system_prompt="Samimi bir chatbotsun.",
            enable_memory=True
        ))
    
    @staticmethod
    def create_coder(name: str = "Coder"):
        return CoreAgent(AgentConfig(
            name=name,
            model=ChatOpenAI(model="gpt-4"),
            system_prompt="Python uzmanısın.",
            tools=create_python_coding_tools()
        ))

# Kullanım
chatbot = AgentFactory.create_chatbot()
coder = AgentFactory.create_coder()
```

### 3. Config'i JSON'dan Yükle
```python
import json

def create_agent_from_json(json_path: str):
    with open(json_path) as f:
        config_dict = json.load(f)
    
    # Model'i ayrı oluştur
    model = ChatOpenAI(model=config_dict.pop("model_name", "gpt-4o-mini"))
    
    config = AgentConfig(
        model=model,
        **config_dict
    )
    
    return CoreAgent(config)

# config.json
{
    "name": "MyAgent",
    "system_prompt": "Sen bir asistansın.",
    "enable_memory": true,
    "memory_backend": "inmemory"
}
```

## 🎓 Özet

1. **Basit başla**: Sadece `name`, `model`, `system_prompt` ile başla
2. **İhtiyaca göre ekle**: Memory, tools, rate limiting vs. sadece gerekirse
3. **Config karmaşık değil**: Sadece ihtiyacın olan parametreleri kullan
4. **Factory pattern**: Tekrar kullanılabilir agent creator'lar yaz
5. **Test et**: Önce inmemory backend ile test et, sonra production'a geç

## Daha Fazla Örnek

`core/test_core/simple_agent_creators.py` dosyasında 20+ hazır agent creator örneği var!