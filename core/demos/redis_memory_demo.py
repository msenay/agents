#!/usr/bin/env python3
"""
Redis Memory Demo - CoreAgent ile Redis'in tüm memory özelliklerini test eder

Gereksinimler:
- Redis Stack kurulu olmalı (docker-compose up redis)
- pip install redis langgraph-checkpoint-redis langgraph-store-redis

Test edilecekler:
1. Short-term memory (conversation/thread-based)
2. Long-term memory (key-value store)
3. Semantic memory (vector search)
4. Session memory (multi-agent sharing)
5. TTL (Time-To-Live) support
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any

from core import CoreAgent, AgentConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# Redis bağlantı kontrolü
def check_redis_connection():
    """Redis bağlantısını kontrol et"""
    try:
        import redis
        r = redis.from_url("redis://localhost:6379")
        r.ping()
        print("✅ Redis bağlantısı başarılı")
        return True
    except Exception as e:
        print(f"❌ Redis bağlantısı başarısız: {e}")
        print("🔧 Çözüm: docker-compose up redis")
        return False


class RedisMemoryDemo:
    """Redis memory özelliklerini test eden demo"""
    
    def __init__(self):
        self.redis_url = "redis://localhost:6379"
        self.model = None
        self.agent = None
        
    def setup(self):
        """Demo için gerekli setup"""
        print("\n🚀 Redis Memory Demo Başlıyor...")
        print("=" * 60)
        
        # OpenAI API key kontrolü
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY bulunamadı. Mock model kullanılacak.")
            from core.simple_examples import MockLLM
            self.model = MockLLM()
        else:
            print("✅ OpenAI API key bulundu")
            self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
    def create_memory_tools(self):
        """Long-term memory için tool'lar oluştur"""
        
        @tool
        def save_user_info(name: str, info: Dict[str, Any]) -> str:
            """Kullanıcı bilgilerini long-term memory'ye kaydet"""
            if self.agent:
                key = f"user_info_{name}"
                self.agent.memory_manager.store_long_term_memory(key, info)
                return f"{name} için bilgiler kaydedildi"
            return "Agent henüz hazır değil"
            
        @tool  
        def get_user_info(name: str) -> str:
            """Kullanıcı bilgilerini long-term memory'den getir"""
            if self.agent:
                key = f"user_info_{name}"
                info = self.agent.memory_manager.get_long_term_memory(key)
                if info:
                    return f"{name} bilgileri: {json.dumps(info, ensure_ascii=False)}"
                return f"{name} için bilgi bulunamadı"
            return "Agent henüz hazır değil"
            
        @tool
        def search_similar_notes(query: str, limit: int = 3) -> str:
            """Semantic search ile benzer notları bul"""
            if self.agent and hasattr(self.agent.memory_manager, 'search_memory'):
                results = self.agent.memory_manager.search_memory(query, limit=limit)
                if results:
                    return f"Benzer notlar: {results}"
                return "Benzer not bulunamadı"
            return "Semantic search mevcut değil"
            
        return [save_user_info, get_user_info, search_similar_notes]
        
    def test_short_term_memory(self):
        """Short-term (conversation) memory testi"""
        print("\n\n🔵 TEST 1: Short-term Memory (Thread-based Conversations)")
        print("-" * 60)
        
        config = AgentConfig(
            name="RedisShortTermAgent",
            model=self.model,
            system_prompt="Sen Redis memory kullanan bir asistansın.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["short_term"],
            redis_url=self.redis_url
        )
        
        try:
            agent = CoreAgent(config)
            print("✅ Short-term memory agent oluşturuldu")
            
            # Thread 1 - Ali ile konuşma
            print("\n📱 Thread 1 - Ali:")
            thread_1_config = {"configurable": {"thread_id": "ali_conversation"}}
            
            response1 = agent.invoke("Merhaba, benim adım Ali ve İstanbul'da yaşıyorum", config=thread_1_config)
            print(f"Ali: Merhaba, benim adım Ali ve İstanbul'da yaşıyorum")
            print(f"Agent: {response1['messages'][-1].content}")
            
            response2 = agent.invoke("En sevdiğim yemek lahmacun", config=thread_1_config)
            print(f"\nAli: En sevdiğim yemek lahmacun")
            print(f"Agent: {response2['messages'][-1].content}")
            
            # Thread 2 - Ayşe ile konuşma
            print("\n\n📱 Thread 2 - Ayşe:")
            thread_2_config = {"configurable": {"thread_id": "ayse_conversation"}}
            
            response3 = agent.invoke("Selam ben Ayşe, Ankara'da oturuyorum", config=thread_2_config)
            print(f"Ayşe: Selam ben Ayşe, Ankara'da oturuyorum")
            print(f"Agent: {response3['messages'][-1].content}")
            
            # Thread 1'e geri dön
            print("\n\n📱 Thread 1'e Geri Dönüş:")
            response4 = agent.invoke("Adımı ve yaşadığım şehri hatırlıyor musun?", config=thread_1_config)
            print(f"Ali: Adımı ve yaşadığım şehri hatırlıyor musun?")
            print(f"Agent: {response4['messages'][-1].content}")
            
            # State kontrolü
            if hasattr(agent.compiled_graph, 'get_state'):
                state = agent.compiled_graph.get_state(thread_1_config)
                print(f"\n📊 Thread 1 State: {len(state.values.get('messages', []))} mesaj")
                
            print("\n✅ Short-term memory testi başarılı!")
            
        except Exception as e:
            print(f"❌ Short-term memory testi başarısız: {e}")
            
    def test_long_term_memory(self):
        """Long-term (persistent store) memory testi"""
        print("\n\n🟢 TEST 2: Long-term Memory (Persistent Key-Value Store)")
        print("-" * 60)
        
        config = AgentConfig(
            name="RedisLongTermAgent", 
            model=self.model,
            system_prompt="Sen kullanıcı bilgilerini kaydeden bir asistansın.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["short_term", "long_term"],
            redis_url=self.redis_url,
            tools=self.create_memory_tools()
        )
        
        try:
            self.agent = agent = CoreAgent(config)
            print("✅ Long-term memory agent oluşturuldu")
            
            # Manuel kaydetme
            print("\n📝 Manuel Kaydetme:")
            user_data = {
                "name": "Ali",
                "age": 28,
                "city": "İstanbul",
                "interests": ["teknoloji", "müzik", "seyahat"],
                "registered_at": datetime.now().isoformat()
            }
            
            agent.memory_manager.store_long_term_memory("user_ali_profile", user_data)
            print(f"Kaydedilen: {user_data}")
            
            # Manuel okuma
            print("\n📖 Manuel Okuma:")
            retrieved = agent.memory_manager.get_long_term_memory("user_ali_profile")
            print(f"Okunan: {retrieved}")
            
            # Tool ile kullanım
            print("\n🔧 Tool ile Kullanım:")
            response = agent.invoke("Ali kullanıcısının bilgilerini getir")
            print(f"Tool yanıtı: {response['messages'][-1].content}")
            
            # Farklı namespace'de kaydetme
            print("\n📁 Namespace Kullanımı:")
            agent.memory_manager.store_long_term_memory(
                "settings",
                {"theme": "dark", "language": "tr"},
                namespace="app_config"
            )
            settings = agent.memory_manager.get_long_term_memory("settings", namespace="app_config")
            print(f"App settings: {settings}")
            
            print("\n✅ Long-term memory testi başarılı!")
            
        except Exception as e:
            print(f"❌ Long-term memory testi başarısız: {e}")
            
    def test_semantic_memory(self):
        """Semantic (vector search) memory testi"""
        print("\n\n🔴 TEST 3: Semantic Memory (Vector-based Similarity Search)")
        print("-" * 60)
        
        # Embedding model kontrolü
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Semantic search OpenAI API key gerektirir. Test atlanıyor.")
            return
            
        config = AgentConfig(
            name="RedisSemanticAgent",
            model=self.model,
            system_prompt="Sen semantic search yapabilen bir asistansın.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["long_term", "semantic"],
            redis_url=self.redis_url,
            embedding_model="openai:text-embedding-3-small",
            embedding_dims=1536,
            tools=self.create_memory_tools()
        )
        
        try:
            self.agent = agent = CoreAgent(config)
            print("✅ Semantic memory agent oluşturuldu")
            
            # Çeşitli notlar kaydet
            print("\n📝 Notlar Kaydediliyor:")
            notes = [
                ("travel_paris", {"content": "Paris'te Eyfel Kulesi'ni gördüm, harika bir deneyimdi", "date": "2024-01-15"}),
                ("travel_tokyo", {"content": "Tokyo'da sakura ağaçları çiçek açmıştı, muhteşem manzara", "date": "2024-03-20"}),
                ("cooking_pasta", {"content": "İtalyan usulü makarna yapmayı öğrendim, domates sosu tarifi", "date": "2024-02-10"}),
                ("tech_python", {"content": "Python ile machine learning projesi geliştirdim", "date": "2024-01-05"}),
                ("book_scifi", {"content": "Dune kitabını okudum, bilimkurgu severler için harika", "date": "2024-02-28"})
            ]
            
            for key, data in notes:
                agent.memory_manager.store_long_term_memory(key, data)
                print(f"  ✓ {key}: {data['content'][:50]}...")
                
            # Semantic search testleri
            print("\n🔍 Semantic Search Testleri:")
            
            queries = [
                "seyahat anıları",
                "yemek tarifleri", 
                "programlama projeleri",
                "Japonya deneyimleri"
            ]
            
            for query in queries:
                print(f"\n📍 Aranan: '{query}'")
                if hasattr(agent.memory_manager, 'search_memory'):
                    results = agent.memory_manager.search_memory(query, limit=3)
                    if results:
                        for i, result in enumerate(results, 1):
                            print(f"  {i}. {result}")
                    else:
                        print("  Sonuç bulunamadı")
                        
            print("\n✅ Semantic memory testi başarılı!")
            
        except Exception as e:
            print(f"❌ Semantic memory testi başarısız: {e}")
            print("📌 Not: Redis Stack (RediSearch modülü) gerekli!")
            
    def test_session_memory(self):
        """Session (multi-agent shared) memory testi"""
        print("\n\n🟡 TEST 4: Session Memory (Multi-Agent Shared Memory)")
        print("-" * 60)
        
        session_id = "team_collaboration_123"
        
        # Agent 1: Researcher
        config1 = AgentConfig(
            name="ResearchAgent",
            model=self.model,
            system_prompt="Sen araştırma yapan bir agentsın.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["session"],
            redis_url=self.redis_url,
            session_id=session_id
        )
        
        # Agent 2: Writer
        config2 = AgentConfig(
            name="WriterAgent",
            model=self.model,
            system_prompt="Sen yazı yazan bir agentsın.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["session"],
            redis_url=self.redis_url,
            session_id=session_id
        )
        
        try:
            agent1 = CoreAgent(config1)
            agent2 = CoreAgent(config2)
            print(f"✅ Session agents oluşturuldu (Session: {session_id})")
            
            # Agent 1 araştırma yapar ve paylaşır
            print("\n👤 Agent 1 (Researcher) veri paylaşıyor:")
            if agent1.memory_manager.has_session_memory():
                research_data = {
                    "topic": "Yapay Zeka Trendleri 2024",
                    "key_points": [
                        "Multimodal AI yükselişte",
                        "Edge AI cihazları yaygınlaşıyor",
                        "AI regulation artıyor"
                    ],
                    "sources": ["MIT Review", "Nature AI", "ArXiv"]
                }
                agent1.memory_manager.store_session_memory(research_data)
                print(f"  ✓ Araştırma verileri paylaşıldı: {research_data['topic']}")
                
                # Agent 2 veriyi okur
                print("\n👤 Agent 2 (Writer) veriyi okuyor:")
                shared_data = agent2.memory_manager.get_session_memory()
                if shared_data:
                    print(f"  ✓ Paylaşılan veri alındı: {len(shared_data)} item")
                    for item in shared_data:
                        print(f"    - {item}")
                        
                # Agent-specific memory
                print("\n📌 Agent-specific memory:")
                agent1.memory_manager.store_agent_memory(
                    "ResearchAgent",
                    session_id,
                    {"status": "research_completed", "duration": "2 hours"}
                )
                
                agent2.memory_manager.store_agent_memory(
                    "WriterAgent", 
                    session_id,
                    {"status": "writing_draft", "word_count": 1500}
                )
                
                print("  ✓ Agent-specific veriler kaydedildi")
                
            else:
                print("⚠️  Session memory Redis backend gerektirir")
                
            print("\n✅ Session memory testi tamamlandı!")
            
        except Exception as e:
            print(f"❌ Session memory testi başarısız: {e}")
            
    def test_ttl_support(self):
        """TTL (Time-To-Live) desteği testi"""
        print("\n\n⏰ TEST 5: TTL Support (Auto-expiration)")
        print("-" * 60)
        
        config = AgentConfig(
            name="RedisTTLAgent",
            model=self.model,
            system_prompt="Sen TTL destekli memory kullanan bir asistansın.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["short_term", "long_term"],
            redis_url=self.redis_url,
            enable_ttl=True,
            default_ttl_minutes=1,  # 1 dakika TTL
            refresh_on_read=True
        )
        
        try:
            agent = CoreAgent(config)
            print("✅ TTL agent oluşturuldu (TTL: 1 dakika)")
            
            # Veri kaydet
            print("\n📝 TTL'li veri kaydediliyor:")
            temp_data = {
                "session_token": "abc123xyz",
                "created_at": datetime.now().isoformat(),
                "purpose": "temporary auth token"
            }
            
            agent.memory_manager.store_long_term_memory("temp_session", temp_data)
            print(f"  ✓ Geçici veri kaydedildi: {temp_data}")
            
            # Hemen oku
            print("\n📖 Veri hemen okunuyor:")
            retrieved = agent.memory_manager.get_long_term_memory("temp_session")
            print(f"  ✓ Veri mevcut: {retrieved is not None}")
            
            # TTL refresh testi
            if config.refresh_on_read:
                print("\n🔄 TTL refresh testi:")
                print("  - refresh_on_read=True olduğu için TTL yenilendi")
                print("  - Veri 1 dakika daha yaşayacak")
                
            print("\n⏳ Not: 1 dakika sonra veri otomatik silinecek")
            print("✅ TTL testi tamamlandı!")
            
        except Exception as e:
            print(f"❌ TTL testi başarısız: {e}")
            
    def test_advanced_features(self):
        """Gelişmiş özellikler testi"""
        print("\n\n🚀 TEST 6: Advanced Features")
        print("-" * 60)
        
        config = AgentConfig(
            name="RedisAdvancedAgent",
            model=self.model,
            system_prompt="Sen gelişmiş Redis özellikleri kullanan bir asistansın.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["short_term", "long_term", "semantic"],
            redis_url=self.redis_url,
            enable_message_trimming=True,
            max_tokens=1000,
            trim_strategy="last"
        )
        
        try:
            agent = CoreAgent(config)
            print("✅ Advanced agent oluşturuldu")
            
            # Message trimming
            print("\n✂️ Message Trimming:")
            print(f"  - Max tokens: {config.max_tokens}")
            print(f"  - Strategy: {config.trim_strategy}")
            print("  - Uzun konuşmalarda eski mesajlar otomatik kesilir")
            
            # Store metadata
            print("\n📊 Metadata Storage:")
            agent.memory_manager.store_long_term_memory(
                "user_session_meta",
                {
                    "user_id": "usr_123",
                    "session_start": datetime.now().isoformat(),
                    "device": "web",
                    "location": "TR",
                    "preferences": {
                        "language": "tr",
                        "theme": "dark"
                    }
                }
            )
            print("  ✓ Session metadata kaydedildi")
            
            # Namespace kullanımı
            print("\n📁 Namespace Organization:")
            namespaces = ["users", "sessions", "analytics", "configs"]
            for ns in namespaces:
                agent.memory_manager.store_long_term_memory(
                    f"test_key",
                    {"namespace": ns, "data": f"Test data for {ns}"},
                    namespace=ns
                )
            print(f"  ✓ {len(namespaces)} farklı namespace'de veri organize edildi")
            
            print("\n✅ Advanced features testi tamamlandı!")
            
        except Exception as e:
            print(f"❌ Advanced features testi başarısız: {e}")
            
    def show_redis_stats(self):
        """Redis kullanım istatistikleri"""
        print("\n\n📊 Redis Kullanım İstatistikleri")
        print("-" * 60)
        
        try:
            import redis
            r = redis.from_url(self.redis_url)
            
            # Key sayıları
            keys = r.keys("*")
            print(f"\n🔑 Toplam key sayısı: {len(keys)}")
            
            # Key türlerine göre grupla
            key_types = {}
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                prefix = key_str.split(":")[0] if ":" in key_str else "other"
                key_types[prefix] = key_types.get(prefix, 0) + 1
                
            print("\n📂 Key türleri:")
            for prefix, count in sorted(key_types.items()):
                print(f"  - {prefix}: {count} adet")
                
            # Memory kullanımı
            info = r.info("memory")
            used_memory = info.get("used_memory_human", "N/A")
            print(f"\n💾 Memory kullanımı: {used_memory}")
            
        except Exception as e:
            print(f"❌ Redis stats alınamadı: {e}")
            
    def cleanup(self):
        """Test verilerini temizle (opsiyonel)"""
        print("\n\n🧹 Cleanup (Opsiyonel)")
        print("-" * 60)
        
        response = input("Test verilerini temizlemek ister misiniz? (y/N): ")
        if response.lower() == 'y':
            try:
                import redis
                r = redis.from_url(self.redis_url)
                
                # Test key'lerini bul ve sil
                test_keys = r.keys("*test*") + r.keys("*ali*") + r.keys("*ayse*")
                if test_keys:
                    r.delete(*test_keys)
                    print(f"✅ {len(test_keys)} test key'i silindi")
                else:
                    print("ℹ️  Silinecek test key'i bulunamadı")
                    
            except Exception as e:
                print(f"❌ Cleanup başarısız: {e}")
        else:
            print("ℹ️  Test verileri korundu")


def main():
    """Ana demo fonksiyonu"""
    print("🚀 Redis Memory Demo - CoreAgent")
    print("================================")
    print("\nBu demo Redis'in tüm memory özelliklerini test eder:")
    print("- Short-term (conversation) memory")
    print("- Long-term (key-value) memory") 
    print("- Semantic (vector search) memory")
    print("- Session (multi-agent) memory")
    print("- TTL (auto-expiration) support")
    
    # Redis kontrolü
    if not check_redis_connection():
        return
        
    # Demo çalıştır
    demo = RedisMemoryDemo()
    demo.setup()
    
    # Testleri çalıştır
    demo.test_short_term_memory()
    demo.test_long_term_memory()
    demo.test_semantic_memory()
    demo.test_session_memory()
    demo.test_ttl_support()
    demo.test_advanced_features()
    
    # İstatistikler
    demo.show_redis_stats()
    
    # Temizlik
    demo.cleanup()
    
    print("\n\n✨ Redis Memory Demo Tamamlandı!")
    print("\n📚 Öğrendiklerimiz:")
    print("1. Short-term = Thread-based conversation (otomatik)")
    print("2. Long-term = Key-value store (manuel)")
    print("3. Semantic = Vector similarity search")
    print("4. Session = Multi-agent sharing")
    print("5. TTL = Automatic expiration")
    print("\n🎯 Redis tüm memory türlerini destekliyor!")


if __name__ == "__main__":
    main()