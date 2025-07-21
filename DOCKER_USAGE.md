# Core Agent Docker Compose Kılavuzu

Bu rehber Core Agent'ınızı Redis, PostgreSQL ve MongoDB ile birlikte Docker ortamında çalıştırmanız için hazırlanmıştır.

## 🚀 Hızlı Başlangıç

### 1. Ön Gereksinimler
```bash
# Docker ve Docker Compose kurulu olmalı
docker --version
docker-compose --version
```

### 2. Environment Dosyası Oluşturma
```bash
# .env dosyasını oluşturun
cp .env.example .env

# API anahtarlarınızı düzenleyin
nano .env
```

### 3. Servisleri Başlatma

#### Tüm servisleri başlat (önerilen)
```bash
docker-compose up -d
```

#### Sadece veritabanlarını başlat
```bash
docker-compose up -d redis postgres mongodb
```

#### Sadece testleri çalıştır
```bash
docker-compose --profile test up test_runner
```

## 📊 Servis Detayları

### 🔴 Redis
- **Port**: 6379
- **UI**: http://localhost:8001 (RedisInsight)
- **Bağlantı**: `redis://:redis_password@localhost:6379/0`
- **Kullanım**: LangGraph checkpointing, caching

### 🐘 PostgreSQL
- **Port**: 5432
- **Veritabanı**: `core_agent_db`
- **Test DB**: `core_agent_test_db`
- **Kullanıcı**: `core_agent_user`
- **Şifre**: `postgres_password`

### 🍃 MongoDB
- **Port**: 27017
- **Veritabanı**: `core_agent_db`
- **Admin**: `admin:mongo_password`
- **App User**: `core_agent_user:mongo_app_password`

### 🐍 Core Agent App
- **Port**: 8000
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## 🧪 Test Endpoints

Core Agent uygulaması çalıştığında şu endpoint'leri kullanabilirsiniz:

```bash
# Health check
curl http://localhost:8000/health

# Config testi
curl -X POST http://localhost:8000/test/config \
  -H "Content-Type: application/json" \
  -d '{"backend": "redis", "enable_memory": true}'

# Redis testi
curl http://localhost:8000/test/redis

# PostgreSQL testi
curl http://localhost:8000/test/postgres

# MongoDB testi
curl http://localhost:8000/test/mongodb
```

## 🔧 Geliştirme Komutları

### Logları izleme
```bash
# Tüm servislerin logları
docker-compose logs -f

# Sadece app logları
docker-compose logs -f core_agent_app

# Sadece veritabanı logları
docker-compose logs -f redis postgres mongodb
```

### Container'a bağlanma
```bash
# App container'ına bash ile bağlan
docker-compose exec core_agent_app bash

# Redis CLI
docker-compose exec redis redis-cli -a redis_password

# PostgreSQL CLI
docker-compose exec postgres psql -U core_agent_user -d core_agent_db

# MongoDB CLI
docker-compose exec mongodb mongosh -u admin -p mongo_password
```

### Testleri çalıştırma
```bash
# Comprehensive testler
docker-compose --profile test up test_runner

# App container'ında manuel test
docker-compose exec core_agent_app python -m pytest core/test_core/ -v
```

### Verileri temizleme
```bash
# Tüm servisleri durdur
docker-compose down

# Verileri de sil (dikkat!)
docker-compose down -v

# Tamamen temizle (images dahil)
docker-compose down -v --rmi all
```

## 🎯 Production Kullanımı

### Environment Variables
Production için `.env` dosyasını güncellemeyi unutmayın:

```bash
# API Keys
OPENAI_API_KEY=sk-your-real-key
LANGCHAIN_API_KEY=your-real-key
LANGCHAIN_TRACING_V2=true

# Güvenlik için şifreleri değiştirin
REDIS_ARGS=--requirepass your_strong_redis_password
POSTGRES_PASSWORD=your_strong_postgres_password
MONGO_INITDB_ROOT_PASSWORD=your_strong_mongo_password
```

### Volume Backup
```bash
# PostgreSQL backup
docker-compose exec postgres pg_dump -U core_agent_user core_agent_db > backup.sql

# Redis backup (RDB)
docker-compose exec redis redis-cli -a redis_password --rdb dump.rdb

# MongoDB backup
docker-compose exec mongodb mongodump --uri="mongodb://admin:mongo_password@localhost:27017/core_agent_db?authSource=admin"
```

## 🐛 Sorun Giderme

### Port çakışması
```bash
# Hangi portların kullanıldığını kontrol et
sudo netstat -tlnp | grep :6379
sudo netstat -tlnp | grep :5432
sudo netstat -tlnp | grep :27017
sudo netstat -tlnp | grep :8000
```

### Memory issues
```bash
# Docker sistem bilgisi
docker system df

# Kullanılmayan container'ları temizle
docker system prune -f
```

### Bağlantı sorunları
```bash
# Network'ü kontrol et
docker network ls
docker network inspect workspace_core_agent_network
```

## 📋 Faydalı Komutlar

```bash
# Servislerin durumunu göster
docker-compose ps

# Sadece belirli servisleri yeniden başlat
docker-compose restart redis postgres

# Container resource kullanımı
docker stats

# Tüm Core Agent container'larını durdur
docker stop $(docker ps -q --filter "name=core_agent")
```

## 🎉 Başarılı Kurulum Kontrolü

Tüm servisler çalışıyorsa şu adımları test edin:

1. ✅ http://localhost:8000/health - Sağlık kontrolü
2. ✅ http://localhost:8000/docs - API dokümantasyonu  
3. ✅ http://localhost:8001 - RedisInsight UI
4. ✅ `docker-compose ps` - Tüm servisler "Up" durumunda
5. ✅ Test endpoint'leri çalışıyor

Artık Core Agent'ınız tüm veritabanları ile birlikte hazır! 🚀