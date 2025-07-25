# PostgreSQL Memory Backend Detaylı Açıklama

## PostgreSQL Seçildiğinde Ne Oluyor?

PostgreSQL'i memory backend olarak seçtiğinizde, her memory türü için farklı implementasyonlar devreye giriyor:

## 1. Config Örneği

```python
config = AgentConfig(
    enable_memory=True,
    memory_backend="postgres",
    postgres_url="postgresql://user:pass@localhost:5432/agent_db",
    memory_types=["short_term", "long_term", "session", "semantic"],
    embedding_model="openai:text-embedding-3-small",  # Semantic için
    session_id="session_123"  # Session için
)
```

## 2. Her Memory Türü İçin Ne Oluyor?

### 🔵 **short_term** (Thread-based conversation memory)
- **Kullanılan Sınıf**: `PostgresSaver` (LangGraph'tan)
- **Ne Yapıyor**: Her thread_id için ayrı conversation state'i PostgreSQL'de saklanıyor
- **Tablo**: LangGraph otomatik olarak checkpoints tablosu oluşturuyor
- **Kullanım**:
```python
# Thread 1'de konuşma
agent.invoke("Merhaba, ben Ali", config={"configurable": {"thread_id": "thread_1"}})

# Thread 2'de farklı konuşma
agent.invoke("Selam, ben Ayşe", config={"configurable": {"thread_id": "thread_2"}})
```

### 🟢 **long_term** (Persistent key-value storage)
- **Kullanılan Sınıf**: `PostgresStore` (LangGraph'tan)
- **Ne Yapıyor**: Kalıcı veri depolama, thread'lerden bağımsız
- **Tablo**: LangGraph otomatik olarak store tablosu oluşturuyor
- **Kullanım**:
```python
# Veri kaydetme
memory_manager.store_long_term_memory("user_preferences", {"theme": "dark", "lang": "tr"})

# Veri okuma
prefs = memory_manager.get_long_term_memory("user_preferences")
```

### 🟡 **session** (Multi-agent shared memory)
- **Kullanılan Sınıf**: Custom implementation (Redis kullanıyor)
- **Ne Yapıyor**: Birden fazla agent arasında paylaşılan memory
- **ÖNEMLİ**: Session memory şu anda sadece Redis backend'i destekliyor!
- **PostgreSQL'de**: Session memory çalışmaz, Redis URL'si gerekir

### 🔴 **semantic** (Vector-based similarity search)
- **Kullanılan Sınıf**: `PostgresStore` with pgvector extension
- **Ne Yapıyor**: Embedding'lerle semantic arama
- **Gereksinimler**: 
  - PostgreSQL'de `pgvector` extension kurulu olmalı
  - `embedding_model` config'de belirtilmeli
- **Kullanım**:
```python
# Semantic veri kaydetme
memory_manager.store_long_term_memory("doc1", {"content": "Paris'te Eyfel Kulesi'ni gördüm"})
memory_manager.store_long_term_memory("doc2", {"content": "Python ile web scraping yaptım"})

# Semantic arama
results = memory_manager.search_memory("seyahat anıları", limit=5)
```

## 3. PostgreSQL Backend Özellikleri

### ✅ Desteklenenler:
- ✅ **short_term**: Thread-based conversation memory (PostgresSaver)
- ✅ **long_term**: Persistent key-value storage (PostgresStore)
- ✅ **semantic**: Vector search (pgvector extension ile)
- ❌ **session**: Multi-agent shared memory (Sadece Redis destekliyor)

### 🔧 Gereksinimler:
1. PostgreSQL 12+ 
2. pgvector extension (semantic search için)
3. Yeterli disk alanı
4. Connection string: `postgresql://user:pass@host:port/dbname`

### 📊 PostgreSQL vs Redis vs InMemory

| Özellik | PostgreSQL | Redis | InMemory |
|---------|------------|-------|----------|
| short_term | ✅ PostgresSaver | ✅ RedisSaver | ✅ InMemorySaver |
| long_term | ✅ PostgresStore | ✅ RedisStore | ✅ InMemoryStore |
| session | ❌ | ✅ Custom | ❌ |
| semantic | ✅ pgvector | ✅ RediSearch | ✅ Basit |
| TTL | ❌ | ✅ | ❌ |
| Persistent | ✅ | ✅ | ❌ |
| ACID | ✅ | ❌ | ❌ |

## 4. Örnek Senaryo

```python
# PostgreSQL backend ile agent oluşturma
config = AgentConfig(
    name="PostgresAgent",
    model=model,
    enable_memory=True,
    memory_backend="postgres",
    postgres_url="postgresql://agent_user:pass@localhost:5432/agent_db",
    memory_types=["short_term", "long_term", "semantic"],  # session yok!
    embedding_model="openai:text-embedding-3-small",
    embedding_dims=1536
)

agent = CoreAgent(config)

# 1. Short-term memory (conversation)
response1 = agent.invoke("Benim adım Mehmet", config={"configurable": {"thread_id": "user_123"}})
response2 = agent.invoke("Adımı hatırlıyor musun?", config={"configurable": {"thread_id": "user_123"}})

# 2. Long-term memory (persistent data)
mm = agent.memory_manager
mm.store_long_term_memory("user_info", {"name": "Mehmet", "city": "İstanbul"})

# 3. Semantic search
mm.store_long_term_memory("note1", {"content": "Bugün hava çok güzeldi, parka gittim"})
mm.store_long_term_memory("note2", {"content": "Python öğrenmeye başladım"})

similar_notes = mm.search_memory("doğa ve açık hava", limit=3)
```

## 5. Limitasyonlar

1. **Session Memory**: PostgreSQL backend'de çalışmıyor, Redis gerekiyor
2. **TTL**: PostgreSQL TTL desteklemiyor, manuel temizlik gerekir
3. **pgvector**: Semantic search için pgvector extension kurulu olmalı
4. **Performance**: Çok büyük vector aramaları için Redis daha hızlı olabilir

## Sonuç

PostgreSQL seçtiğinizde:
- ✅ Thread-based conversations (short_term)
- ✅ Persistent storage (long_term) 
- ✅ Semantic search (semantic) - pgvector ile
- ❌ Multi-agent shared memory (session) - Çalışmaz

Eğer session memory'ye ihtiyacınız varsa, ya Redis backend kullanın ya da hybrid setup yapın (PostgreSQL + Redis URL'si birlikte).