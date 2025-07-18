# CoreAgent Framework - Özet Dokümantasyon

Bu projeyi şu dökümanda belirtilen tüm özelliklerle birlikte kapsamlı bir **CoreAgent Framework** yaratıldı. Framework, LangGraph ekosisteminin tüm özelliklerini opsiyonel olarak içerir ve farklı prompt'lar, tool'lar ve konfigürasyonlarla yeni agenlar yaratmaya imkan sağlar.

## 🎯 Yaratılan Sistem

### Ana Bileşenler

1. **`core_agent.py`** - Ana framework dosyası
2. **`examples/basic_usage.py`** - Temel kullanım örnekleri
3. **`examples/specialized_agents.py`** - Özelleştirilmiş agent örnekleri
4. **`examples/multi_agent.py`** - Multi-agent pattern örnekleri (supervisor, swarm, handoff)
5. **`requirements.txt`** - Gerekli bağımlılıklar
6. **`README.md`** - Kapsamlı İngilizce dokümantasyon
7. **`SUMMARY.md`** - Türkçe özet dokümantasyon
8. **`test_framework.py`** - Test scripti

### İçerdiği Özellikler (Dökümanda Belirtilen)

✅ **1. Subgraph (graf-içinde-graf) kapsülleme** - Yeniden kullanılabilir çekirdek bileşenler
✅ **2. Kalıcı bellek için RedisSaver** - Çok oturumlu uzun hafıza
✅ **3. SupervisorGraph** - Hiyerarşik çok-ajan orkestratörü (3 pattern: supervisor, swarm, handoff)
✅ **4. langgraph-prebuilt** - Hazır bileşenler (aktif kullanılıyor)
✅ **5. langgraph-supervisor** - Supervisor agent'lar için araçlar  
✅ **6. langgraph-swarm** - Swarm multi-agent sistem araçları
✅ **7. langchain-mcp-adapters** - MCP server entegrasyonu
✅ **8. langmem** - Agent hafıza yönetimi
✅ **9. agentevals** - Agent performans değerlendirme

### 🆕 Multi-Agent Mimarileri

1. **Supervisor Pattern** - Merkezi koordinatör agent'lar arasında görev dağıtır
2. **Swarm Pattern** - Agent'lar dinamik olarak birbirlerine transfer yapar  
3. **Handoff Pattern** - Manuel agent transferleri (built-in olarak mevcut)

### Ek Özellikler

- **Human-in-the-Loop** - İnsan geri bildirimi ve müdahale
- **Streaming Support** - Gerçek zamanlı yanıt akışı
- **Memory Management** - Kısa ve uzun vadeli hafıza
- **Tool Integration** - Araç entegrasyonu
- **Configuration Management** - Konfigürasyon kaydetme/yükleme
- **Agent Evaluation** - Agent performans değerlendirme

## 🚀 Nasıl Kullanılır

### 1. Kurulum

```bash
# Virtual environment yaratın
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Temel bağımlılıkları yükleyin
pip install langgraph langchain langchain-core pydantic

# Opsiyonel ekstra özellikler için:
pip install langgraph-supervisor langgraph-swarm langchain-mcp-adapters langmem agentevals redis
```

### 2. Temel Agent Yaratma

```python
from core_agent import create_basic_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Tool tanımlayın
@tool
def calculate(expression: str) -> str:
    """Matematiksel hesaplama yapar."""
    return str(eval(expression))

# Temel agent yaratın
model = ChatOpenAI("gpt-4")
agent = create_basic_agent(model=model, tools=[calculate])

# Kullanın
result = agent.invoke("15 kere 7 kaçtır?")
print(result)
```

### 3. Özelleştirilmiş Agent Yaratma

```python
from core_agent import CoreAgent, AgentConfig

# Özelleştirilmiş konfigürasyon
config = AgentConfig(
    name="UzmanAsistan",
    description="Özelleştirilmiş uzman asistan",
    model=ChatOpenAI("gpt-4"),
    tools=[calculate, search_web],
    system_prompt="Sen Türkçe konuşan uzman bir asistansın.",
    enable_memory=True,
    memory_type="redis",
    redis_url="redis://localhost:6379",
    enable_supervisor=True,
    enable_evaluation=True,
    enable_streaming=True,
    enable_human_feedback=True
)

agent = CoreAgent(config)
```

### 4. Specialized Agent Sınıfları

Framework ile farklı amaçlar için özelleştirilmiş agent'lar yaratabilirsiniz:

```python
# Kod inceleme agent'ı
class KodİncelemeAgent(CoreAgent):
    def __init__(self, model):
        config = AgentConfig(
            name="KodİncelemeAgent",
            model=model,
            system_prompt="Sen uzman bir yazılım geliştiricisin...",
            tools=[kod_analiz_et, dökümantasyon_ara, test_çalıştır],
            evaluation_metrics=["teknik_doğruluk", "yardımseverlik"]
        )
        super().__init__(config)
    
    def kod_incele(self, kod: str, dil: str = "python"):
        return self.invoke(f"Bu {dil} kodunu incele: {kod}")

# Araştırma agent'ı
class AraştırmaAgent(CoreAgent):
    def __init__(self, model):
        config = AgentConfig(
            name="AraştırmaAgent", 
            model=model,
            system_prompt="Sen bir araştırma uzmanısın...",
            tools=[akademik_ara, doğrula, web_ara],
            enable_evaluation=True
        )
        super().__init__(config)

# Müşteri hizmetleri agent'ı
class MüşteriHizmetleriAgent(CoreAgent):
    def __init__(self, model):
        config = AgentConfig(
            name="MüşteriHizmetleriAgent",
            model=model,
            system_prompt="Sen müşteri hizmetleri uzmanısın...",
            tools=[müşteri_veritabanı, bilet_oluştur],
            enable_human_feedback=True,
            interrupt_before=["execute_tools"]
        )
        super().__init__(config)
```

### 5. Multi-Agent Koordinasyon

Framework 3 farklı multi-agent mimarisi destekler:

#### A) Supervisor Pattern (Merkezi Koordinasyon)
```python
# Özelleştirilmiş agent'lar yaratın
flight_agent = UçuşAgent(model)
hotel_agent = OtelAgent(model)

# Supervisor yaratın
from core_agent import create_supervisor_agent

agents = {"flight": flight_agent, "hotel": hotel_agent}
supervisor = create_supervisor_agent(model, agents)

# Merkezi koordinasyon
result = supervisor.coordinate_task("Uçuş ve otel rezervasyonu yap")
```

#### B) Swarm Pattern (Dinamik Transfer)
```python
from core_agent import create_swarm_agent

# Swarm sistemi yaratın
swarm = create_swarm_agent(model, agents, default_active_agent="flight")

# Dinamik agent değişimi
result = swarm.coordinate_task("Seyahat planla")
```

#### C) Handoff Pattern (Manuel Transfer)
```python
from core_agent import create_handoff_agent

# Handoff sistemi yaratın
handoff = create_handoff_agent(model, agents, default_active_agent="flight")

# Manuel transferler
result = handoff.coordinate_task("Yardım et")
```

### 6. Hafıza Yönetimi

```python
# Hafızada bilgi saklayın
agent.store_memory("kullanıcı_tercihleri", {"dil": "Türkçe", "stil": "resmi"})
agent.store_memory("sohbet_geçmişi", ["Önceki konuşma bağlamı"])

# Hafızadan bilgi alın
tercihler = agent.retrieve_memory("kullanıcı_tercihleri")
geçmiş = agent.retrieve_memory("sohbet_geçmişi")
```

### 7. Streaming ve Human-in-the-Loop

```python
# Streaming kullanın
for chunk in agent.stream("Bu veriyi analiz et"):
    print(chunk)

# Human-in-the-loop konfigürasyonu
config = AgentConfig(
    enable_human_feedback=True,
    interrupt_before=["execute_tools"],  # Araç çalıştırmadan önce duraklat
    interrupt_after=["generate_response"]  # Yanıt ürettikten sonra duraklat
)
```

## 🧪 Test ve Örnekler

### Testleri Çalıştırın

```bash
# Framework testleri
source venv/bin/activate
python test_framework.py

# Temel kullanım örnekleri
python examples/basic_usage.py

# Özelleştirilmiş agent örnekleri 
python examples/specialized_agents.py
```

### Mevcut Test Sonuçları

```
CoreAgent Framework Test Suite
==================================================
✓ Package availability
✓ Basic agent creation
✓ Custom configuration agent
✓ Memory functionality
✓ Subgraph functionality  
✓ Agent status
✓ Configuration save/load

Test Results: 7/7 tests passed
🎉 All tests passed! CoreAgent framework is working correctly.
```

## 📋 Konfigürasyon Seçenekleri

`AgentConfig` sınıfı ile kapsamlı konfigürasyon yapabilirsiniz:

```python
config = AgentConfig(
    # Temel ayarlar
    name="ÖzelAgent",
    description="Agent açıklaması",
    model=ChatOpenAI("gpt-4"),
    system_prompt="Sistem prompt'u",
    
    # Araçlar ve yetenekler
    tools=[araç1, araç2, araç3],
    tool_calling_enabled=True,
    pre_model_hook=model_öncesi_fonksiyon,
    post_model_hook=model_sonrası_fonksiyon,
    response_format=YanıtFormatı,
    
    # Hafıza ayarları
    enable_memory=True,
    memory_type="redis",  # "memory", "redis", veya "both"
    redis_url="redis://localhost:6379",
    
    # Gelişmiş özellikler
    enable_supervisor=True,
    enable_swarm=True,
    enable_mcp=True,
    enable_evaluation=True,
    
    # Human-in-the-loop
    enable_human_feedback=True,
    interrupt_before=["execute_tools"],
    interrupt_after=["generate_response"],
    
    # Diğer
    enable_streaming=True,
    enable_subgraphs=True,
    evaluation_metrics=["doğruluk", "yardımseverlik", "verimlilik"]
)
```

## 🏗️ Mimari Genel Bakış

### Ana Bileşenler

1. **CoreAgent** - Ana agent sınıfı
2. **AgentConfig** - Konfigürasyon dataclass'ı  
3. **CoreAgentState** - Agent durumu Pydantic modeli
4. **SubgraphManager** - Yeniden kullanılabilir subgraph yöneticisi
5. **MemoryManager** - Kısa ve uzun vadeli hafıza yöneticisi
6. **SupervisorManager** - Multi-agent koordinasyon yöneticisi
7. **EvaluationManager** - Agent performans değerlendirme yöneticisi

### Factory Fonksiyonları

- `create_basic_agent()` - Basit agent yaratır
- `create_advanced_agent()` - Gelişmiş özellikli agent yaratır
- `create_supervisor_agent()` - Multi-agent koordinasyon için supervisor yaratır

## 🎯 Sonuç

Bu CoreAgent framework ile:

1. ✅ **Dökümandaki tüm özellikleri** opsiyonel olarak kullanabilirsiniz
2. ✅ **Farklı prompt'lar** ile özelleştirilmiş agent'lar yaratabilirsiniz  
3. ✅ **Farklı tool'lar** ekleyerek agent yeteneklerini genişletebilirsiniz
4. ✅ **Farklı konfigürasyonlar** ile agent davranışlarını özelleştirebilirsiniz
5. ✅ **Multi-agent sistemler** kurabilirsiniz
6. ✅ **Memory, streaming, human-in-the-loop** gibi gelişmiş özellikleri kullanabilirsiniz

Framework esnek bir yapıya sahiptir ve eksik bağımlılıkları zarif bir şekilde handle eder. Sadece ihtiyacınız olan özellikleri aktifleştirerek kullanabilirsiniz.

**Başlamak için:** `python examples/basic_usage.py` komutunu çalıştırarak örnek kullanımları inceleyebilirsiniz!