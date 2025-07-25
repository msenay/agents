#!/usr/bin/env python3
"""
Memory Flow Example - Short-term vs Long-term Memory Akışı
Gerçek kullanım senaryosu ile memory'nin nasıl çalıştığını gösterir
"""

from core import CoreAgent, AgentConfig
from langchain_core.tools import tool
from datetime import datetime


def demo_memory_flow():
    """Memory akışını gösteren demo"""
    
    # 1. Memory'li agent oluştur
    print("🤖 Memory'li Agent Oluşturuluyor...")
    print("=" * 60)
    
    # Long-term memory tool'u
    @tool
    def remember_important_info(category: str, info: str) -> str:
        """Önemli bilgileri long-term memory'ye kaydet"""
        # Gerçek implementasyonda agent.memory_manager kullanılır
        print(f"💾 Long-term memory'ye kaydediliyor: {category} -> {info}")
        return f"'{info}' bilgisi '{category}' kategorisinde kaydedildi"
    
    @tool
    def recall_info(category: str) -> str:
        """Long-term memory'den bilgi getir"""
        # Gerçek implementasyonda agent.memory_manager.get_long_term_memory kullanılır
        print(f"🔍 Long-term memory'den aranıyor: {category}")
        return f"{category} hakkında kayıtlı bilgi: ..."
    
    config = AgentConfig(
        name="SmartAssistant",
        model=None,  # Demo için None
        system_prompt="""Sen akıllı bir asistansın. 
        Önemli bilgileri remember_important_info tool'u ile sakla.
        Gerektiğinde recall_info ile hatırla.""",
        enable_memory=True,
        memory_types=["short_term", "long_term"],
        memory_backend="inmemory",
        tools=[remember_important_info, recall_info]
    )
    
    # Not: Gerçek kullanımda agent = CoreAgent(config) yapılır
    print("✅ Agent oluşturuldu")
    print(f"   - Short-term memory: ✓ (Otomatik)")
    print(f"   - Long-term memory: ✓ (Manuel)")
    print(f"   - Tools: {[t.name for t in config.tools]}")
    
    # 2. İlk Konuşma - Thread 1
    print("\n\n📱 KULLANICI 1 - İlk Konuşma")
    print("-" * 60)
    
    thread_1 = "user_ali_session"
    
    print("USER: Merhaba, benim adım Ali. 25 yaşındayım ve İstanbul'da yaşıyorum.")
    print("🧠 SHORT-TERM: Mesaj otomatik olarak thread'e kaydedildi")
    print(f"   Thread ID: {thread_1}")
    print("   Kaydedilen: [HumanMessage, AIMessage]")
    
    print("\nAGENT: Merhaba Ali! Tanıştığımıza memnun oldum.")
    print("      [Agent remember_important_info tool'unu kullanıyor...]")
    print("💾 LONG-TERM: user_profile -> {name: Ali, age: 25, city: İstanbul}")
    
    # 3. Aynı Thread'de Devam
    print("\n\n📱 KULLANICI 1 - Devam Eden Konuşma")
    print("-" * 60)
    
    print("USER: En sevdiğim yemek mantı. Bunu da not alır mısın?")
    print("🧠 SHORT-TERM: Önceki mesajlar otomatik yüklendi:")
    print("   - Merhaba, benim adım Ali...")
    print("   - Merhaba Ali!...")
    print("   - En sevdiğim yemek mantı...")
    
    print("\nAGENT: Tabii Ali, favori yemeğinin mantı olduğunu not aldım.")
    print("💾 LONG-TERM: user_preferences -> {favorite_food: mantı}")
    
    # 4. Farklı Thread - Başka Kullanıcı
    print("\n\n📱 KULLANICI 2 - Yeni Konuşma")
    print("-" * 60)
    
    thread_2 = "user_ayse_session"
    
    print("USER: Selam, ben Ayşe. Ankara'da oturuyorum.")
    print("🧠 SHORT-TERM: Yeni thread, boş geçmiş")
    print(f"   Thread ID: {thread_2}")
    print("   ❌ Ali'nin mesajları YOK (farklı thread)")
    
    print("\nAGENT: Merhaba Ayşe! Ankara'da yaşıyorsun demek.")
    print("💾 LONG-TERM: user_profile -> {name: Ayşe, city: Ankara}")
    
    # 5. İlk Kullanıcıya Dönüş
    print("\n\n📱 KULLANICI 1 - Ertesi Gün")
    print("-" * 60)
    
    print("USER: Merhaba, dün konuşmuştuk. Adımı hatırlıyor musun?")
    print("🧠 SHORT-TERM: Thread history otomatik yüklendi")
    print(f"   Thread ID: {thread_1}")
    print("   ✅ Tüm eski mesajlar mevcut")
    
    print("\nAGENT: Elbette hatırlıyorum Ali! Dün tanışmıştık.")
    print("      İstanbul'da yaşıyorsun ve en sevdiğin yemek mantı.")
    print("      (Short-term'den otomatik, long-term'den tool ile)")
    
    # 6. Memory Akış Özeti
    print("\n\n📊 MEMORY AKIŞ ÖZETİ")
    print("=" * 60)
    
    print("\n🔵 SHORT-TERM MEMORY (Otomatik):")
    print("├─ Thread bazlı konuşma geçmişi")
    print("├─ invoke() ile otomatik yüklenir")
    print("├─ Her thread ayrı")
    print("└─ Mesajlar sıralı saklanır")
    
    print("\n🟢 LONG-TERM MEMORY (Manuel):")
    print("├─ Global key-value storage")
    print("├─ Thread'lerden bağımsız")
    print("├─ Manuel save/load gerekir")
    print("└─ Tool'larla veya direkt erişimle")
    
    print("\n🔄 NASIL BİRLİKTE ÇALIŞIRLAR:")
    print("1. User mesaj gönderir → Short-term'e otomatik kaydedilir")
    print("2. Agent yanıt verir → Short-term'e otomatik eklenir")
    print("3. Agent önemli bilgi görürse → Tool ile long-term'e kaydeder")
    print("4. Sonraki invoke'da → Short-term otomatik, long-term manuel yüklenir")
    
    # 7. Pratik Kod Örneği
    print("\n\n💻 GERÇEK KOD ÖRNEĞİ")
    print("=" * 60)
    
    code_example = '''
# Agent oluştur
agent = CoreAgent(AgentConfig(
    name="Assistant",
    model=ChatOpenAI(),
    enable_memory=True,
    memory_types=["short_term", "long_term"],
    memory_backend="redis"
))

# SENARYO 1: İlk kullanıcı
response1 = agent.invoke(
    "Benim adım Ali, 25 yaşındayım",
    config={"configurable": {"thread_id": "ali_chat"}}
)
# Short-term: Otomatik kaydedildi ✓

# Long-term: Manuel kaydet
agent.memory_manager.store_long_term_memory(
    "user_ali",
    {"name": "Ali", "age": 25}
)

# SENARYO 2: Ali geri geliyor
response2 = agent.invoke(
    "Yaşımı hatırlıyor musun?",
    config={"configurable": {"thread_id": "ali_chat"}}
)
# Short-term: Önceki mesajlar otomatik yüklendi ✓
# Agent görür: "Benim adım Ali, 25 yaşındayım"

# SENARYO 3: Farklı thread'de Ali'yi hatırlamak
ali_info = agent.memory_manager.get_long_term_memory("user_ali")
# Long-term'den manuel çekiyoruz
'''
    
    print(code_example)


if __name__ == "__main__":
    print("🧠 MEMORY FLOW DEMONSTRATION")
    print("Short-term vs Long-term Memory Nasıl Çalışır?\n")
    
    demo_memory_flow()
    
    print("\n\n✨ Demo Tamamlandı!")
    print("\n📌 ÖNEMLİ NOKTALAR:")
    print("1. Short-term = Thread bazlı, otomatik")
    print("2. Long-term = Global, manuel")
    print("3. thread_id aynıysa → Eski mesajlar otomatik yüklenir")
    print("4. thread_id farklıysa → Yeni konuşma başlar")
    print("5. Long-term her zaman erişilebilir (manuel)")