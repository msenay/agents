#!/usr/bin/env python3
"""
Core Agent - Basit Kullanım Örnekleri
Direkt çalıştırılabilir örnekler
"""

from core import CoreAgent, AgentConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from typing import List, Any


# Mock model for testing (gerçek kullanımda ChatOpenAI kullan)
class MockLLM(BaseChatModel):
    def _generate(self, messages: List[BaseMessage], **kwargs) -> Any:
        user_msg = messages[-1].content if messages else ""
        return AIMessage(content=f"Merhaba! Mesajını aldım: '{user_msg}'")
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def invoke(self, input: Any, config=None, **kwargs) -> BaseMessage:
        if isinstance(input, list):
            return self._generate(input)
        return AIMessage(content=f"Yanıt: {input}")


# ============================================================
# ÖRNEK 1: En Basit Agent
# ============================================================
def example_1_minimal_agent():
    """En minimal agent örneği"""
    print("=" * 50)
    print("ÖRNEK 1: Minimal Agent")
    print("=" * 50)
    
    # Config - sadece zorunlu parametreler
    config = AgentConfig(
        name="MinimalAgent",
        model=MockLLM()
    )
    
    # Agent oluştur
    agent = CoreAgent(config)
    
    # Kullan
    response = agent.invoke("Merhaba dünya!")
    print(f"Yanıt: {response['messages'][-1].content}")


# ============================================================
# ÖRNEK 2: System Prompt'lu Agent
# ============================================================
def example_2_with_prompt():
    """System prompt'lu agent"""
    print("\n" + "=" * 50)
    print("ÖRNEK 2: System Prompt'lu Agent")
    print("=" * 50)
    
    config = AgentConfig(
        name="AsistanAgent",
        model=MockLLM(),
        system_prompt="Sen yardımsever bir asistansın. Her zaman nazik ve profesyonel ol."
    )
    
    agent = CoreAgent(config)
    response = agent.invoke("Bugün hava nasıl?")
    print(f"Yanıt: {response['messages'][-1].content}")


# ============================================================
# ÖRNEK 3: Tool'lu Agent
# ============================================================
def example_3_with_tools():
    """Tool kullanan agent"""
    print("\n" + "=" * 50)
    print("ÖRNEK 3: Tool'lu Agent")
    print("=" * 50)
    
    # Basit bir tool tanımla
    @tool
    def hesap_makinesi(islem: str) -> str:
        """Basit matematik işlemleri yapar. Örnek: '2 + 2' veya '10 * 5'"""
        try:
            sonuc = eval(islem)
            return f"Sonuç: {sonuc}"
        except:
            return "Hata: Geçersiz işlem"
    
    @tool
    def tarih_saat() -> str:
        """Şu anki tarih ve saati döndürür"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Config
    config = AgentConfig(
        name="ToolAgent",
        model=MockLLM(),
        system_prompt="Sen matematik ve zaman konularında yardımcı olan bir asistansın.",
        tools=[hesap_makinesi, tarih_saat]
    )
    
    agent = CoreAgent(config)
    print(f"Kullanılabilir tool'lar: {[t.name for t in config.tools]}")


# ============================================================
# ÖRNEK 4: Memory'li Agent
# ============================================================
def example_4_with_memory():
    """Memory kullanan agent"""
    print("\n" + "=" * 50)
    print("ÖRNEK 4: Memory'li Agent")
    print("=" * 50)
    
    config = AgentConfig(
        name="MemoryAgent",
        model=MockLLM(),
        system_prompt="Sen hafızası olan bir asistansın.",
        
        # Memory ayarları
        enable_memory=True,
        memory_backend="inmemory",
        memory_types=["short_term"]  # Thread-based conversation memory
    )
    
    agent = CoreAgent(config)
    
    # Thread 1'de konuşma
    print("\nThread 1:")
    agent.invoke("Benim adım Ahmet", config={"configurable": {"thread_id": "user_1"}})
    agent.invoke("Adımı hatırlıyor musun?", config={"configurable": {"thread_id": "user_1"}})
    
    # Thread 2'de farklı konuşma
    print("\nThread 2:")
    agent.invoke("Ben Mehmet", config={"configurable": {"thread_id": "user_2"}})
    agent.invoke("Kim olduğumu biliyor musun?", config={"configurable": {"thread_id": "user_2"}})


# ============================================================
# ÖRNEK 5: Streaming Agent
# ============================================================
def example_5_streaming():
    """Streaming destekli agent"""
    print("\n" + "=" * 50)
    print("ÖRNEK 5: Streaming Agent")
    print("=" * 50)
    
    config = AgentConfig(
        name="StreamingAgent",
        model=MockLLM(),
        enable_streaming=True
    )
    
    agent = CoreAgent(config)
    
    # Stream kullanımı
    print("Streaming yanıt:")
    for chunk in agent.stream("Uzun bir hikaye anlat"):
        # Gerçek kullanımda chunk'lar parça parça gelir
        print(".", end="", flush=True)
    print("\nStreaming tamamlandı!")


# ============================================================
# ÖRNEK 6: Rate Limited Agent
# ============================================================
def example_6_rate_limited():
    """Rate limiting'li agent"""
    print("\n" + "=" * 50)
    print("ÖRNEK 6: Rate Limited Agent")
    print("=" * 50)
    
    config = AgentConfig(
        name="RateLimitedAgent",
        model=MockLLM(),
        
        # Rate limiting
        enable_rate_limiting=True,
        requests_per_second=2.0,  # Saniyede max 2 istek
        max_bucket_size=5.0
    )
    
    agent = CoreAgent(config)
    print(f"Rate limit: {config.requests_per_second} istek/saniye")


# ============================================================
# FACTORY PATTERN ÖRNEĞİ
# ============================================================
class AgentFactory:
    """Agent oluşturmak için factory pattern"""
    
    @staticmethod
    def create_chatbot(name: str = "Chatbot") -> CoreAgent:
        """Basit bir chatbot oluştur"""
        return CoreAgent(AgentConfig(
            name=name,
            model=MockLLM(),
            system_prompt="Sen samimi ve yardımsever bir chatbotsun.",
            enable_memory=True,
            memory_backend="inmemory"
        ))
    
    @staticmethod
    def create_coder(name: str = "Coder") -> CoreAgent:
        """Kod yazan agent oluştur"""
        
        @tool
        def python_runner(code: str) -> str:
            """Python kodunu çalıştırır"""
            return "Kod çalıştırıldı (simülasyon)"
        
        return CoreAgent(AgentConfig(
            name=name,
            model=MockLLM(),
            system_prompt="Sen uzman bir Python geliştiricisisin.",
            tools=[python_runner],
            enable_memory=True
        ))
    
    @staticmethod
    def create_researcher(name: str = "Researcher") -> CoreAgent:
        """Araştırma yapan agent oluştur"""
        
        @tool
        def web_search(query: str) -> str:
            """Web'de arama yapar"""
            return f"'{query}' için arama sonuçları (simülasyon)"
        
        return CoreAgent(AgentConfig(
            name=name,
            model=MockLLM(),
            system_prompt="Sen detaylı araştırma yapan bir asistansın.",
            tools=[web_search],
            enable_memory=True,
            memory_types=["short_term", "long_term"]
        ))


# ============================================================
# ÇALIŞTIR
# ============================================================
if __name__ == "__main__":
    print("🚀 Core Agent Basit Örnekler\n")
    
    # Tüm örnekleri çalıştır
    example_1_minimal_agent()
    example_2_with_prompt()
    example_3_with_tools()
    example_4_with_memory()
    example_5_streaming()
    example_6_rate_limited()
    
    # Factory pattern örneği
    print("\n" + "=" * 50)
    print("FACTORY PATTERN ÖRNEĞİ")
    print("=" * 50)
    
    factory = AgentFactory()
    
    # Hazır agent'lar oluştur
    chatbot = factory.create_chatbot("SohbetBotu")
    coder = factory.create_coder("KodYazıcı")
    researcher = factory.create_researcher("Araştırmacı")
    
    print(f"✅ {chatbot.config.name} oluşturuldu")
    print(f"✅ {coder.config.name} oluşturuldu")
    print(f"✅ {researcher.config.name} oluşturuldu")
    
    print("\n✨ Tüm örnekler tamamlandı!")
    print("\n📌 Gerçek kullanım için:")
    print("   - MockLLM yerine ChatOpenAI kullan")
    print("   - Gerçek tool'lar ekle")
    print("   - Production için Redis/PostgreSQL backend kullan")