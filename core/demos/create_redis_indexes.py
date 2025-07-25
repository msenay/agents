#!/usr/bin/env python3
"""
Helper script to create Redis indexes manually
Use this if automatic index creation fails
"""

import redis
import sys

REDIS_URL = "redis://:redis_password@localhost:6379"


def create_indexes():
    """Create required indexes for LangGraph Redis backend"""
    
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Connected to Redis")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False
    
    indexes_created = 0
    
    # 1. Checkpoints index for short-term memory
    try:
        r.execute_command("FT.INFO", "checkpoints")
        print("✅ Index 'checkpoints' already exists")
    except:
        try:
            r.execute_command(
                "FT.CREATE", "checkpoints",
                "ON", "HASH",
                "PREFIX", "1", "checkpoint:",
                "SCHEMA",
                "thread_id", "TAG",
                "checkpoint_id", "TAG", 
                "thread_ts", "NUMERIC",
                "parent_ts", "NUMERIC",
                "channel_values", "TEXT",
                "channel_versions", "TEXT",
                "versions_seen", "TEXT",
                "pending_sends", "TEXT",
                "metadata", "TEXT"
            )
            print("✅ Created 'checkpoints' index")
            indexes_created += 1
        except Exception as e:
            print(f"❌ Failed to create checkpoints index: {e}")
    
    # 2. Store index for long-term memory
    try:
        r.execute_command("FT.INFO", "store")
        print("✅ Index 'store' already exists")
    except:
        try:
            r.execute_command(
                "FT.CREATE", "store",
                "ON", "HASH",
                "PREFIX", "1", "store:",
                "SCHEMA",
                "namespace", "TAG",
                "key", "TAG",
                "value", "TEXT",
                "created_at", "NUMERIC",
                "updated_at", "NUMERIC"
            )
            print("✅ Created 'store' index")
            indexes_created += 1
        except Exception as e:
            print(f"❌ Failed to create store index: {e}")
    
    # 3. List all indexes
    print("\n📋 Current Redis indexes:")
    try:
        indexes = r.execute_command("FT._LIST")
        for idx in indexes:
            print(f"   - {idx.decode() if isinstance(idx, bytes) else idx}")
    except Exception as e:
        print(f"   Error listing indexes: {e}")
    
    print(f"\n✅ Created {indexes_created} new index(es)")
    print("\n💡 You can now run redis_memory_demo.py")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("🔧 REDIS INDEX CREATOR")
    print("="*60)
    print("\nThis script creates indexes required by LangGraph Redis backend")
    
    if create_indexes():
        print("\n✅ Success!")
    else:
        print("\n❌ Failed!")
        sys.exit(1)