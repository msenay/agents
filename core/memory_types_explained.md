# Short-term vs Long-term Memory Detaylı Açıklama

## 🧠 Memory Türleri ve Farkları

### 1. Short-term Memory (Kısa Süreli Hafıza)

**Ne Oluyor?**
- LangGraph'ın **checkpointer** sistemi kullanılıyor
- Her `thread_id` için ayrı konuşma geçmişi tutuluyor
- Mesajlar otomatik olarak state'e ekleniyor ve checkpoint'leniyor
- **invoke() çağrıldığında otomatik olarak yükleniyor**

**Nasıl Saklanıyor?**
```python
# Backend: InMemorySaver, RedisSaver, PostgresSaver
# Veri yapısı:
{
    "thread_id": "user_123",
    "checkpoint": {
        "messages": [
            HumanMessage("Merhaba, ben Ali"),
            AIMessage("Merhaba Ali!"),
            HumanMessage("Hava nasıl?"),
            AIMessage("Hava güzel...")
        ],
        "other_state": {...}
    }
}
```

**Otomatik Kullanım:**
```python
# İlk mesaj
agent.invoke("Benim adım Ali", config={"configurable": {"thread_id": "user_123"}})

# İkinci mesaj - ÖNCEKİ MESAJLAR OTOMATİK YÜKLENİR!
response = agent.invoke("Adımı hatırlıyor musun?", config={"configurable": {"thread_id": "user_123"}})
# Agent tüm konuşma geçmişini görür ve "Evet Ali" der
```

### 2. Long-term Memory (Uzun Süreli Hafıza)

**Ne Oluyor?**
- LangGraph'ın **store** sistemi kullanılıyor
- Key-value şeklinde kalıcı veri saklama
- Thread'lerden bağımsız, manuel yönetim gerekiyor
- **Manuel olarak save/load yapman gerekiyor**

**Nasıl Saklanıyor?**
```python
# Backend: InMemoryStore, RedisStore, PostgresStore
# Veri yapısı:
{
    "namespace": ("user_data",),
    "key": "user_123_profile",
    "value": {
        "name": "Ali",
        "preferences": {"theme": "dark", "language": "tr"},
        "history": ["login_2024_01_15", "purchase_item_x"]
    }
}
```

**Manuel Kullanım:**
```python
# Veri kaydetme
agent.memory_manager.store_long_term_memory(
    key="user_123_profile",
    data={"name": "Ali", "city": "İstanbul", "interests": ["coding", "AI"]}
)

# Veri okuma - MANUEL YAPMAN GEREK
user_data = agent.memory_manager.get_long_term_memory("user_123_profile")
```

### 3. Semantic Memory (Anlamsal Hafıza)

**Ne Oluyor?**
- Long-term memory'nin özel bir türü
- Veriler embedding'lerle vektör olarak saklanıyor
- Benzerlik araması yapılabiliyor

**Nasıl Saklanıyor?**
```python
# Text → Embedding → Vector DB
"Paris'te Eyfel Kulesi'ni gördüm" → [0.23, -0.45, 0.67, ...] → Vector DB

# Arama yapınca:
"Seyahat anıları" → [0.21, -0.43, 0.65, ...] → Benzer vektörleri bul
```

## 📊 Karşılaştırma Tablosu

| Özellik | Short-term Memory | Long-term Memory |
|---------|-------------------|------------------|
| **Ne Zaman Kullanılır** | Her invoke'da otomatik | Manuel save/load |
| **Veri Türü** | Konuşma mesajları | Her türlü veri |
| **Scope** | Thread bazlı | Global |
| **Otomatik mı?** | ✅ Evet | ❌ Hayır |
| **Kullanım** | Konuşma hafızası | Kalıcı bilgiler |
| **Örnek** | "Adım Ali" → "Adını hatırlıyor musun?" | Kullanıcı profili, notlar |

## 🔄 Nasıl Birlikte Çalışıyorlar?

```python
# Agent oluştur
config = AgentConfig(
    name="SmartAgent",
    model=model,
    enable_memory=True,
    memory_types=["short_term", "long_term"],
    memory_backend="redis"
)
agent = CoreAgent(config)

# 1. SHORT-TERM: Konuşma otomatik saklanır
response1 = agent.invoke(
    "Benim adım Ali, İstanbul'da yaşıyorum",
    config={"configurable": {"thread_id": "session_123"}}
)

# 2. LONG-TERM: Agent kendi karar verip saklayabilir
# (Agent'ın system prompt'unda bu yetenek tanımlanmışsa)
agent.memory_manager.store_long_term_memory(
    "user_ali_info",
    {"name": "Ali", "city": "İstanbul", "first_seen": "2024-01-15"}
)

# 3. SONRAKİ KULLANIM
# Short-term otomatik yüklenir
response2 = agent.invoke(
    "Adımı ve yaşadığım şehri hatırlıyor musun?",
    config={"configurable": {"thread_id": "session_123"}}
)
# Agent: "Evet, adın Ali ve İstanbul'da yaşıyorsun" (short-term'den)

# Long-term manuel yüklenir (veya agent tool kullanır)
user_info = agent.memory_manager.get_long_term_memory("user_ali_info")
```

## 🎯 Pratik Kullanım Senaryoları

### Senaryo 1: Chatbot
```python
config = AgentConfig(
    enable_memory=True,
    memory_types=["short_term"],  # Sadece konuşma hafızası
    memory_backend="redis"
)

# Her kullanıcı için ayrı thread
agent.invoke("Merhaba", config={"configurable": {"thread_id": "user_1"}})
agent.invoke("Merhaba", config={"configurable": {"thread_id": "user_2"}})
```

### Senaryo 2: Kişisel Asistan
```python
config = AgentConfig(
    enable_memory=True,
    memory_types=["short_term", "long_term", "semantic"],
    memory_backend="postgres"
)

# Short-term: Günlük konuşmalar
# Long-term: Kullanıcı tercihleri, önemli bilgiler
# Semantic: Notlar, dökümanlar için arama
```

### Senaryo 3: Öğrenen Agent
```python
# Agent her konuşmadan öğrenir
@tool
def save_learned_info(category: str, info: str):
    """Öğrenilen bilgiyi sakla"""
    agent.memory_manager.store_long_term_memory(
        f"learned_{category}",
        {"info": info, "date": datetime.now()}
    )
    return "Bilgi kaydedildi"

config = AgentConfig(
    tools=[save_learned_info],
    enable_memory=True,
    memory_types=["short_term", "long_term"],
    system_prompt="""
    Sen öğrenen bir asistansın. 
    Kullanıcıdan öğrendiğin önemli bilgileri save_learned_info tool'u ile sakla.
    """
)
```

## 🔑 Önemli Noktalar

1. **Short-term = Otomatik**: Her invoke'da thread_id'ye göre otomatik yüklenir
2. **Long-term = Manuel**: Sen (veya agent) manuel save/load yapmalı
3. **Semantic = Arama**: Embedding'lerle benzerlik araması
4. **Session = Paylaşım**: Multi-agent sistemlerde agent'lar arası veri paylaşımı

Thread_id ile invoke ettiğinde, short-term memory otomatik olarak tüm geçmiş konuşmayı yükler. Long-term memory'yi ise sen yönetirsin!