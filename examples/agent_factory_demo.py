#!/usr/bin/env python3
"""
🏭 Agent Factory Demo
====================

Agent Factory sisteminin demo'su. Farklı task'lar için otomatik agent yaratımı.

Usage:
    python examples/agent_factory_demo.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_factory import AgentFactory, AgentRequest


def demo_sentiment_analysis_agent():
    """Demo: Sentiment Analysis Agent yaratımı"""
    print("🎭 SENTIMENT ANALYSIS AGENT CREATION")
    print("=" * 50)
    
    factory = AgentFactory()
    
    request = AgentRequest(
        task="Create a sentiment analysis agent that analyzes text and returns positive, negative, or neutral sentiment with confidence scores",
        api_key="mock-api-key",  # Demo için mock key
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        tools=["text_processing", "sentiment_classification"],
        requirements=[
            "Handle multiple languages",
            "Return confidence scores (0-1)",
            "Support batch processing",
            "Robust error handling"
        ]
    )
    
    print(f"Task: {request.task}")
    print(f"Required Tools: {request.tools}")
    print(f"Requirements: {request.requirements}")
    print()
    
    try:
        result = factory.create_agent(request)
        
        print("\n📊 CREATION RESULTS:")
        print(f"✅ Success: {result.success}")
        print(f"📝 Agent Code: {len(result.agent_code)} characters")
        print(f"🧪 Test Code: {len(result.test_code)} characters")
        print(f"💾 Saved to: {result.file_path}")
        
        if result.agent_code:
            print("\n👨‍💻 AGENT CODE PREVIEW:")
            print("-" * 30)
            print(result.agent_code[:800] + "..." if len(result.agent_code) > 800 else result.agent_code)
        
        if result.test_code:
            print("\n🧪 TEST CODE PREVIEW:")
            print("-" * 30)
            print(result.test_code[:400] + "..." if len(result.test_code) > 400 else result.test_code)
        
        if result.review_feedback:
            print("\n🔍 REVIEW FEEDBACK:")
            print("-" * 30)
            print(result.review_feedback[:600] + "..." if len(result.review_feedback) > 600 else result.review_feedback)
        
        if result.test_results:
            print("\n⚡ AGENT TEST RESULTS:")
            print("-" * 30)
            print(result.test_results[:400] + "..." if len(result.test_results) > 400 else result.test_results)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        factory.cleanup()


def demo_data_processor_agent():
    """Demo: Data Processing Agent yaratımı"""
    print("\n📊 DATA PROCESSOR AGENT CREATION")
    print("=" * 50)
    
    factory = AgentFactory()
    
    request = AgentRequest(
        task="Create a data processing agent that can clean, transform, and analyze CSV data files",
        api_key="mock-api-key",
        model="gpt-4o-mini",
        tools=["csv_processing", "data_cleaning", "statistical_analysis"],
        requirements=[
            "Handle missing values",
            "Data type inference",
            "Basic statistical summaries",
            "Export cleaned data"
        ]
    )
    
    print(f"Task: {request.task}")
    print(f"Tools: {request.tools}")
    print()
    
    try:
        result = factory.create_agent(request)
        
        print(f"✅ Success: {result.success}")
        print(f"📄 Agent saved to: {result.file_path}")
        
        if result.agent_code:
            print("\n👨‍💻 DATA PROCESSOR CODE PREVIEW:")
            print(result.agent_code[:600] + "...")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        factory.cleanup()


def demo_web_scraper_agent():
    """Demo: Web Scraper Agent yaratımı"""
    print("\n🕷️ WEB SCRAPER AGENT CREATION")
    print("=" * 50)
    
    factory = AgentFactory()
    
    request = AgentRequest(
        task="Create a web scraper agent that can extract data from e-commerce websites and save to JSON",
        api_key="mock-api-key",
        model="gpt-4o-mini",
        tools=["web_scraping", "html_parsing", "json_processing"],
        requirements=[
            "Respect robots.txt",
            "Handle dynamic content",
            "Rate limiting",
            "Export to JSON format"
        ]
    )
    
    print(f"Task: {request.task}")
    print(f"Tools: {request.tools}")
    print()
    
    try:
        result = factory.create_agent(request)
        
        print(f"✅ Success: {result.success}")
        print(f"📄 Agent saved to: {result.file_path}")
        
        if result.agent_code:
            print("\n🕷️ WEB SCRAPER CODE PREVIEW:")
            print(result.agent_code[:600] + "...")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        factory.cleanup()


def demo_agent_comparison():
    """Demo: Multiple agent yaratıp karşılaştırma"""
    print("\n🏆 AGENT COMPARISON DEMO")
    print("=" * 50)
    
    tasks = [
        {
            "name": "Email Classifier",
            "task": "Create an email classification agent that categorizes emails as spam, important, or normal",
            "tools": ["text_classification", "email_processing"]
        },
        {
            "name": "Password Generator",
            "task": "Create a secure password generator agent with customizable rules",
            "tools": ["cryptography", "random_generation"]
        },
        {
            "name": "Image Processor",
            "task": "Create an image processing agent that can resize, crop, and apply filters",
            "tools": ["image_processing", "file_handling"]
        }
    ]
    
    results = []
    
    for task_info in tasks:
        print(f"\n🔨 Creating {task_info['name']}...")
        
        factory = AgentFactory()
        
        request = AgentRequest(
            task=task_info['task'],
            api_key="mock-api-key",
            tools=task_info['tools']
        )
        
        try:
            result = factory.create_agent(request)
            results.append({
                "name": task_info['name'],
                "success": result.success,
                "code_length": len(result.agent_code),
                "test_length": len(result.test_code),
                "file_path": result.file_path
            })
            print(f"  ✅ {task_info['name']}: {result.success}")
        except Exception as e:
            results.append({
                "name": task_info['name'],
                "success": False,
                "error": str(e)
            })
            print(f"  ❌ {task_info['name']}: Failed")
        finally:
            factory.cleanup()
    
    # Summary
    print("\n📊 COMPARISON RESULTS:")
    print("-" * 60)
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        if result['success']:
            print(f"{result['name']:<20} {status} - Code: {result['code_length']} chars")
        else:
            print(f"{result['name']:<20} {status} - Error: {result.get('error', 'Unknown')}")


def interactive_agent_creator():
    """Interactive agent yaratıcı"""
    print("\n🤖 INTERACTIVE AGENT CREATOR")
    print("=" * 50)
    
    print("Create your custom agent!")
    print("Enter the details below:")
    
    # Get user input
    task = input("\n📝 What should your agent do? ")
    if not task.strip():
        task = "Create a hello world agent that greets users"
    
    tools_input = input("🛠️ Required tools (comma-separated, optional): ")
    tools = [t.strip() for t in tools_input.split(",") if t.strip()] if tools_input.strip() else None
    
    requirements_input = input("📋 Special requirements (comma-separated, optional): ")
    requirements = [r.strip() for r in requirements_input.split(",") if r.strip()] if requirements_input.strip() else None
    
    print(f"\n🏭 Creating agent for task: '{task}'")
    if tools:
        print(f"🛠️ Tools: {tools}")
    if requirements:
        print(f"📋 Requirements: {requirements}")
    
    factory = AgentFactory()
    
    request = AgentRequest(
        task=task,
        api_key="mock-api-key",
        tools=tools,
        requirements=requirements
    )
    
    try:
        result = factory.create_agent(request)
        
        print(f"\n✅ Agent creation completed!")
        print(f"Success: {result.success}")
        print(f"File: {result.file_path}")
        
        if result.agent_code:
            print(f"\n👨‍💻 Your agent code ({len(result.agent_code)} characters):")
            print("=" * 60)
            print(result.agent_code)
        
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
    finally:
        factory.cleanup()


def main():
    """Ana demo fonksiyonu"""
    print("🏭 AGENT FACTORY SYSTEM DEMO")
    print("=" * 60)
    print("Otomatik agent geliştirme pipeline'ı demonstrasyonu")
    print()
    
    demos = [
        ("1", "Sentiment Analysis Agent", demo_sentiment_analysis_agent),
        ("2", "Data Processor Agent", demo_data_processor_agent),
        ("3", "Web Scraper Agent", demo_web_scraper_agent),
        ("4", "Agent Comparison", demo_agent_comparison),
        ("5", "Interactive Creator", interactive_agent_creator),
        ("A", "Run All Demos", None)
    ]
    
    print("Available demos:")
    for code, name, _ in demos:
        print(f"  {code}: {name}")
    
    choice = input("\nSelect demo (1-5, A for all, or Enter for #1): ").strip().upper()
    
    if not choice:
        choice = "1"
    
    if choice == "A":
        # Run all demos except interactive
        for code, name, func in demos:
            if func and code != "5":
                print(f"\n{'='*60}")
                print(f"Running {name}...")
                print(f"{'='*60}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ Demo {name} failed: {e}")
    else:
        # Run selected demo
        for code, name, func in demos:
            if code == choice and func:
                print(f"\n{'='*60}")
                print(f"Running {name}...")
                print(f"{'='*60}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ Demo {name} failed: {e}")
                break
        else:
            print("❌ Invalid selection")
    
    print("\n🎉 Demo completed!")
    print("\n💡 Next steps:")
    print("  1. Replace 'mock-api-key' with your real OpenAI API key")
    print("  2. Customize the AgentRequest parameters")
    print("  3. Run the generated agents in your environment")
    print("  4. Integrate the Agent Factory into your workflow!")


if __name__ == "__main__":
    main()