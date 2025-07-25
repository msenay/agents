# Core Agent Demos

Bu klasörde Core Agent'ın çeşitli özelliklerini gösteren demo'lar bulunur.

## 📁 Demo Listesi

### 1. Redis Memory Demo
**Dosya**: `redis_memory_demo.py`

Redis backend'inin tüm memory özelliklerini test eden kapsamlı demo.

**Test Edilen Özellikler**:
- ✅ Short-term memory (Thread-based conversations)
- ✅ Long-term memory (Key-value store)
- ✅ Semantic memory (Vector similarity search)
- ✅ Session memory (Multi-agent shared memory)
- ✅ TTL support (Auto-expiration)
- ✅ Advanced features (Trimming, namespaces, metadata)

**Çalıştırma**:
```bash
# 1. Redis'i başlat
cd /workspace
docker-compose up redis -d

# 2. Demo'yu çalıştır
cd core/demos
python redis_memory_demo.py
```

**Gereksinimler**:
- Redis Stack (docker-compose ile)
- OpenAI API key (semantic search için)
- Python dependencies (requirements.txt'te mevcut)

## 🚀 Yeni Demo Ekleme

Yeni bir demo eklemek için:

1. `core/demos/` klasörüne yeni bir Python dosyası oluştur
2. Demo'yu açıklayıcı bir şekilde yaz
3. Bu README'ye demo bilgilerini ekle

## 📝 Demo Standartları

Her demo şunları içermeli:
- Açıklayıcı docstring
- Gereksinimler listesi
- Adım adım test senaryoları
- Hata yakalama ve açıklayıcı mesajlar
- Temizlik (cleanup) seçeneği

## 🔗 İlgili Dokümantasyon

- [Core Agent Kullanım Rehberi](../agent_creation_guide.md)
- [Memory Sistemi Açıklaması](../memory_types_explained.md)
- [Basit Örnekler](../simple_examples.py)