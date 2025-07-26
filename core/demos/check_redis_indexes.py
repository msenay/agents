#!/usr/bin/env python3
"""
Check and list Redis indexes
"""

import redis
import sys

REDIS_URL = "redis://:redis_password@localhost:6379"


def check_indexes():
    """Check Redis indexes"""
    
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Connected to Redis")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False
    
    print("\n📋 Current Redis indexes:")
    try:
        indexes = r.execute_command("FT._LIST")
        if not indexes:
            print("   ⚠️  No indexes found!")
            print("\n💡 This is why you're getting 'No such index' errors.")
            print("   The setup() method will create these indexes.")
        else:
            print(f"   Found {len(indexes)} index(es):")
            for idx in indexes:
                idx_name = idx.decode() if isinstance(idx, bytes) else idx
                print(f"   - {idx_name}")
                
                # Get index info
                try:
                    info = r.execute_command("FT.INFO", idx_name)
                    # Convert info list to dict
                    info_dict = {}
                    for i in range(0, len(info), 2):
                        key = info[i].decode() if isinstance(info[i], bytes) else info[i]
                        val = info[i+1]
                        if isinstance(val, bytes):
                            val = val.decode()
                        info_dict[key] = val
                    
                    print(f"     - Number of docs: {info_dict.get('num_docs', 'unknown')}")
                    print(f"     - Index options: {info_dict.get('index_options', 'unknown')}")
                except:
                    pass
                    
    except Exception as e:
        print(f"   Error listing indexes: {e}")
        print("\n⚠️  RediSearch might not be enabled.")
        print("   Make sure you're using redis/redis-stack:latest")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("🔍 REDIS INDEX CHECKER")
    print("="*60)
    
    check_indexes()
    
    print("\n" + "="*60)
    print("📝 Next steps:")
    print("1. If no indexes found, run your agent - setup() will create them")
    print("2. If RediSearch not enabled, use redis/redis-stack:latest")
    print("3. After setup(), run this script again to verify")
    print("="*60)