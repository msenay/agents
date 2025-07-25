#!/usr/bin/env python3
"""
Test new Redis version with updated packages
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.checkpoint.redis import RedisSaver

REDIS_URL = "redis://:redis_password@localhost:6379"

def test_redis_saver():
    """Test if RedisSaver works with new version"""
    print("Testing RedisSaver with new version...")
    
    try:
        # Try direct initialization
        saver = RedisSaver.from_conn_string(REDIS_URL)
        print(f"✅ RedisSaver type: {type(saver)}")
        
        # Test basic operation
        config = {"configurable": {"thread_id": "test_thread"}}
        
        # Try to get a checkpoint (should create index if needed)
        result = saver.get_tuple(config)
        print(f"✅ Get tuple result: {result}")
        
        print("\n✅ New version seems to work!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e)}")
        
        # Check if it's still a context manager
        try:
            with RedisSaver.from_conn_string(REDIS_URL) as saver:
                print(f"✅ Works as context manager: {type(saver)}")
        except Exception as e2:
            print(f"❌ Context manager also failed: {e2}")


if __name__ == "__main__":
    test_redis_saver()