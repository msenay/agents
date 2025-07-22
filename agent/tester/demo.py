#!/usr/bin/env python3
"""
Tester Agent Demo
================

Demonstrates TesterAgent's capabilities for unit test generation.
"""

import sys
sys.path.insert(0, '/workspace')

from agent.tester.tester import TesterAgent


def demo_basic_unit_tests():
    """Demo: Generate basic unit tests"""
    print("\n" + "="*60)
    print("📝 DEMO 1: Basic Unit Test Generation")
    print("="*60)
    
    # Sample code to test
    sample_code = """
def calculate_discount(price, discount_percent):
    '''Calculate discounted price'''
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")
    
    discount_amount = price * (discount_percent / 100)
    return price - discount_amount

def parse_user_data(user_string):
    '''Parse user data from string format'''
    parts = user_string.split(',')
    if len(parts) != 3:
        raise ValueError("Invalid user data format")
    
    return {
        'name': parts[0].strip(),
        'age': int(parts[1].strip()),
        'email': parts[2].strip()
    }
"""
    
    tester = TesterAgent()
    result = tester.generate_tests(sample_code)
    
    if result["success"]:
        print(f"✅ Tests generated successfully!")
        print(f"📊 Framework: {result['framework']}")
        print(f"\n📄 Generated Tests (preview):")
        print(result["tests"][:1000] + "..." if len(result["tests"]) > 1000 else result["tests"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_framework_specific_tests():
    """Demo: Generate tests for specific framework"""
    print("\n" + "="*60)
    print("🛠️ DEMO 2: Framework-Specific Test Generation")
    print("="*60)
    
    sample_code = """
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def divide(self, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self):
        return self.history.copy()
"""
    
    tester = TesterAgent()
    
    # Generate unittest tests
    print("\n🔧 Generating unittest tests...")
    result = tester.generate_tests(sample_code, framework="unittest")
    
    if result["success"]:
        print(f"✅ unittest tests generated!")
        print(f"\n📄 Generated Tests (preview):")
        print(result["tests"][:800] + "..." if len(result["tests"]) > 800 else result["tests"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_test_fixtures():
    """Demo: Generate test fixtures and mocks"""
    print("\n" + "="*60)
    print("🏗️ DEMO 3: Test Fixtures and Mocks Generation")
    print("="*60)
    
    sample_code = """
import requests
from datetime import datetime

class WeatherService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.weather.com/v1"
    
    def get_current_weather(self, city):
        response = requests.get(
            f"{self.base_url}/current",
            params={"city": city, "key": self.api_key}
        )
        response.raise_for_status()
        return response.json()
    
    def get_forecast(self, city, days=5):
        response = requests.get(
            f"{self.base_url}/forecast",
            params={"city": city, "days": days, "key": self.api_key}
        )
        response.raise_for_status()
        data = response.json()
        data['retrieved_at'] = datetime.now().isoformat()
        return data
"""
    
    tester = TesterAgent()
    result = tester.generate_fixtures(sample_code)
    
    if result["success"]:
        print("✅ Fixtures and mocks generated!")
        print("\n📄 Generated Fixtures:")
        print(result["fixtures"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_coverage_analysis():
    """Demo: Analyze test coverage"""
    print("\n" + "="*60)
    print("📊 DEMO 4: Test Coverage Analysis")
    print("="*60)
    
    original_code = """
def fibonacci(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    elif n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True
"""
    
    existing_tests = """
import pytest

def test_fibonacci_base_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1

def test_fibonacci_recursive():
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55
"""
    
    tester = TesterAgent()
    result = tester.analyze_coverage(original_code, existing_tests)
    
    if result["success"]:
        print("✅ Coverage analysis complete!")
        print("\n📊 Analysis Results:")
        print(result["analysis"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_advanced_test_types():
    """Demo: Generate different types of tests"""
    print("\n" + "="*60)
    print("🚀 DEMO 5: Advanced Test Types")
    print("="*60)
    
    sample_code = """
def process_batch(items, batch_size=100):
    '''Process items in batches'''
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        # Simulate processing
        results.extend([item * 2 for item in batch])
    return results

def search_sorted_array(arr, target):
    '''Binary search implementation'''
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
"""
    
    # Use all tools for advanced demos
    tester = TesterAgent(use_all_tools=True)
    
    # Generate parameterized tests
    print("\n🔧 Generating parameterized tests...")
    result = tester.generate_tests(sample_code, test_type="parameterized")
    
    if result["success"]:
        print("✅ Parameterized tests generated!")
        print(f"\n📄 Generated Tests (preview):")
        print(result["tests"][:600] + "..." if len(result["tests"]) > 600 else result["tests"])
    else:
        print(f"❌ Error: {result['error']}")
    
    # Generate performance tests
    print("\n\n⚡ Generating performance tests...")
    result = tester.generate_tests(sample_code, test_type="performance")
    
    if result["success"]:
        print("✅ Performance tests generated!")
        print(f"\n📄 Generated Tests (preview):")
        print(result["tests"][:600] + "..." if len(result["tests"]) > 600 else result["tests"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_interactive_chat():
    """Demo: Interactive chat with TesterAgent"""
    print("\n" + "="*60)
    print("💬 DEMO 6: Interactive Chat")
    print("="*60)
    
    tester = TesterAgent(use_all_tools=True)
    
    # Example request
    request = """I have a function that validates email addresses:

def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

Can you generate comprehensive tests including edge cases?"""
    
    print(f"\n🗨️ Request: {request}")
    
    full_response = ""
    for chunk in tester.stream(request):
        if isinstance(chunk, dict):
            # Extract the actual content from the chunk
            if "agent" in chunk and "messages" in chunk["agent"]:
                for msg in chunk["agent"]["messages"]:
                    if hasattr(msg, "content"):
                        full_response += msg.content
            elif "messages" in chunk:
                for msg in chunk["messages"]:
                    if hasattr(msg, "content"):
                        full_response += msg.content
        else:
            full_response += str(chunk)
    print(f"\n🤖 Response:\n{full_response[:1000]}...")


if __name__ == "__main__":
    print("🚀 TESTER AGENT DEMO - Comprehensive Unit Test Generation")
    print("="*80)
    print("TesterAgent generates high-quality tests with:")
    print("- Multiple framework support (pytest, unittest, nose2)")
    print("- Test fixtures and mocks")
    print("- Coverage analysis")
    print("- Various test types (unit, integration, performance, parameterized)")
    
    try:
        demo_basic_unit_tests()
        demo_framework_specific_tests()
        demo_test_fixtures()
        demo_coverage_analysis()
        demo_advanced_test_types()
        demo_interactive_chat()
        
        print("\n" + "="*80)
        print("✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()