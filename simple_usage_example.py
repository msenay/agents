#!/usr/bin/env python3
"""
🚀 Ultimate Coder Agent - Simple Usage Example
==============================================

Bu dosya Ultimate Coder Agent'ın nasıl kullanılacağını gösterir.
Real API key ile production-ready agent'lar yaratabilirsiniz.

Requirements:
- OpenAI API key (opsiyonel - mock ile de çalışır)
- Redis (opsiyonel - local memory fallback var)
"""

from ultimate_coder_agent import UltimateCoderAgent


def example_1_simple_agent():
    """Basit agent yaratma örneği"""
    print("🎯 ÖRNEK 1: Basit Agent Yaratma")
    print("=" * 50)
    
    # Ultimate Coder Agent oluştur
    coder = UltimateCoderAgent(
        api_key="your-real-openai-api-key-here",  # Gerçek key girin
        model="gpt-4o-mini"  # veya "gpt-4o"
    )
    
    # Tek agent yaratın
    result = coder.create_agent(
        task_description="Create a password generator that creates secure passwords with customizable rules",
        requirements=[
            "Support different length options",
            "Include/exclude special characters", 
            "Generate multiple passwords at once",
            "Validate password strength"
        ],
        tools=["secrets", "string", "re"],
        complexity="intermediate"
    )
    
    # Sonuçları göster
    if result.success:
        print(f"✅ Agent başarıyla yaratıldı!")
        print(f"   📊 Kalite Skoru: {result.quality_score:.2f}")
        print(f"   🔧 Karmaşıklık: {result.complexity_score:.2f}")
        print(f"   📝 Kod Uzunluğu: {result.code_length} karakter")
        print(f"   ⏱️ Yaratım Süresi: {result.creation_time:.2f} saniye")
        
        # Agent'ı dosyaya kaydet
        file_path = coder.save_agent_to_file(result, "password generator")
        print(f"   💾 Kaydedildi: {file_path}")
        
        # Kod preview
        print(f"\n📄 KOD ÖNİZLEME:")
        print("-" * 30)
        print(result.agent_code[:500] + "..." if len(result.agent_code) > 500 else result.agent_code)
        
    else:
        print(f"❌ Agent yaratılamadı: {result.errors}")


def example_2_multiple_agents():
    """Çoklu agent yaratma örneği"""
    print("\n🎨 ÖRNEK 2: Çoklu Agent Yaratma")
    print("=" * 50)
    
    coder = UltimateCoderAgent(
        api_key="your-real-openai-api-key-here",
        model="gpt-4o-mini"
    )
    
    # Birden fazla agent task'ı tanımla
    tasks = [
        "Create a file organizer that sorts files by type and date",
        "Build a system monitor that tracks CPU and memory usage",
        "Make a log analyzer that finds errors and patterns",
        "Create a backup utility that compresses and archives files"
    ]
    
    # Batch olarak yaratın (cross-learning ile)
    results = coder.create_multiple_agents(tasks)
    
    # Sonuçları analiz et
    successful = sum(1 for r in results if r.success)
    average_quality = sum(r.quality_score for r in results if r.success) / successful if successful > 0 else 0
    
    print(f"📊 BATCH SONUÇLARI:")
    print(f"   ✅ Başarılı: {successful}/{len(results)}")
    print(f"   📈 Ortalama Kalite: {average_quality:.2f}")
    
    # Her agent için detay
    for i, (task, result) in enumerate(zip(tasks, results), 1):
        status = "✅" if result.success else "❌"
        quality = f"({result.quality_score:.2f})" if result.success else "(failed)"
        print(f"   {i}. {status} {task[:40]}... {quality}")


def example_3_learning_from_feedback():
    """Feedback ile öğrenme örneği"""
    print("\n📚 ÖRNEK 3: Feedback ile Öğrenme")
    print("=" * 50)
    
    coder = UltimateCoderAgent(
        api_key="your-real-openai-api-key-here",
        model="gpt-4o-mini"
    )
    
    # Agent yaratın
    result = coder.create_agent(
        "Create a JSON data validator with schema support",
        requirements=["Validate JSON structure", "Support custom schemas", "Detailed error messages"],
        complexity="advanced"
    )
    
    if result.success:
        print(f"Agent yaratıldı: Task ID {result.task_id}")
        
        # Simulated feedback (gerçekte kullanıcıdan gelir)
        feedbacks = [
            ("Great code structure, easy to understand", 0.9),
            ("Good error handling but could be more efficient", 0.75),
            ("Perfect documentation and examples", 0.95)
        ]
        
        # Feedback'leri öğret
        for feedback, score in feedbacks:
            coder.learn_from_feedback(result.task_id, feedback, score)
            print(f"📝 Feedback öğrenildi: {feedback[:30]}... (skor: {score})")


def example_4_agent_statistics():
    """Agent istatistikleri örneği"""
    print("\n📊 ÖRNEK 4: Agent İstatistikleri")
    print("=" * 50)
    
    coder = UltimateCoderAgent(
        api_key="your-real-openai-api-key-here",
        model="gpt-4o-mini"
    )
    
    # Birkaç agent yaratın
    simple_tasks = [
        "Create a calculator with basic operations",
        "Build a todo list manager",
        "Make a simple text editor"
    ]
    
    results = coder.create_multiple_agents(simple_tasks)
    
    # İstatistikleri al
    stats = coder.get_agent_statistics()
    
    print(f"🔍 CODER İSTATİSTİKLERİ:")
    print(f"   Session ID: {stats['session_id']}")
    print(f"   Toplam Task: {stats['total_tasks_completed']}")
    print(f"   Başarı Oranı: {stats['success_rate']:.1%}")
    print(f"   Ortalama Kalite: {stats['average_quality_score']:.2f}")
    print(f"   Öğrenilen Pattern'ler: {stats['total_patterns_learned']}")
    print(f"   Task Türleri: {', '.join(stats['task_types_handled'])}")
    print(f"   Memory Backend: {stats['memory_backend']}")
    print(f"   AI Backend: {stats['ai_backend']}")


def example_5_real_api_usage():
    """Gerçek API kullanımı template'i"""
    print("\n🔑 ÖRNEK 5: Gerçek API Kullanımı")
    print("=" * 50)
    
    # Gerçek API key kullanımı için template
    api_key = input("OpenAI API Key girin (boş bırakabilirsiniz demo için): ").strip()
    
    if not api_key:
        api_key = "demo-key"  # Mock kullanır
        print("Demo mode: Mock responses kullanılacak")
    else:
        print("Production mode: Gerçek OpenAI API kullanılacak")
    
    coder = UltimateCoderAgent(
        api_key=api_key,
        model="gpt-4o-mini",  # Veya "gpt-4o" daha güçlü model için
        session_id="my_coding_session"  # Kendi session ID'niz
    )
    
    # Özel task
    task = input("Hangi agent'ı yaratmak istiyorsunuz? ").strip()
    if not task:
        task = "Create a URL shortener with analytics"
    
    result = coder.create_agent(
        task_description=task,
        complexity="intermediate"
    )
    
    if result.success:
        print(f"🎉 '{task}' için agent başarıyla yaratıldı!")
        
        # Agent'ı kaydet
        file_path = coder.save_agent_to_file(result, task)
        print(f"💾 Agent kaydedildi: {file_path}")
        
        # Kullanıcıya göster
        print(f"\n📋 AGENT DETAYLARI:")
        print(f"   Task ID: {result.task_id}")
        print(f"   Kalite: {result.quality_score:.2f}/1.00")
        print(f"   Karmaşıklık: {result.complexity_score:.2f}/1.00")
        print(f"   Yaratım Süresi: {result.creation_time:.1f} saniye")
        
        # Çalıştırabilir mi kontrol et
        try:
            exec(compile(result.agent_code, '<agent>', 'exec'))
            print("✅ Agent kodu hata olmadan çalışabilir")
        except Exception as e:
            print(f"⚠️  Agent kodu syntax hatası içeriyor: {e}")
    else:
        print(f"❌ Agent yaratılamadı: {result.errors}")


def main():
    """Ana demo fonksiyonu"""
    print("🚀 ULTIMATE CODER AGENT - KULLANIM ÖRNEKLERİ")
    print("=" * 60)
    print("Bu agent her task'ı hatırlar, pattern'leri öğrenir ve sürekli gelişir!")
    print()
    
    examples = [
        ("1", "Basit Agent Yaratma", example_1_simple_agent),
        ("2", "Çoklu Agent Yaratma", example_2_multiple_agents),
        ("3", "Feedback ile Öğrenme", example_3_learning_from_feedback),
        ("4", "Agent İstatistikleri", example_4_agent_statistics),
        ("5", "Gerçek API Kullanımı", example_5_real_api_usage),
        ("A", "Tüm Örnekleri Çalıştır", None)
    ]
    
    print("Mevcut örnekler:")
    for code, name, _ in examples:
        print(f"  {code}: {name}")
    
    choice = input(f"\nHangi örneği çalıştırmak istiyorsunuz? (1-5, A, veya Enter = 1): ").strip().upper()
    
    if not choice:
        choice = "1"
    
    if choice == "A":
        # Tüm örnekleri çalıştır
        for code, name, func in examples:
            if func:  # A seçeneği None
                print(f"\n{'='*60}")
                print(f"🔄 {name} çalıştırılıyor...")
                print(f"{'='*60}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ Örnek {name} başarısız oldu: {e}")
                print("\n⏱️  Devam etmek için Enter'a basın...")
                input()
    else:
        # Seçilen örneği çalıştır
        for code, name, func in examples:
            if code == choice and func:
                print(f"\n{'='*60}")
                print(f"🔄 {name} çalıştırılıyor...")
                print(f"{'='*60}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ Örnek {name} başarısız oldu: {e}")
                break
        else:
            print("❌ Geçersiz seçim")
    
    print(f"\n🎊 Örnekler tamamlandı!")
    print(f"\n💡 Sonraki adımlar:")
    print(f"  1. Gerçek OpenAI API key'i ile deneyin")
    print(f"  2. Kendi task'larınız için agent yaratın")
    print(f"  3. Feedback ile agent'ın öğrenmesini sağlayın")
    print(f"  4. Multi-agent workflow'ları kurun")
    print(f"  5. Production environment'ta deploy edin!")


if __name__ == "__main__":
    main()