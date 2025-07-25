#!/usr/bin/env python3
"""
Fix Redis indexes for LangGraph checkpointer
"""

import os
import redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema

REDIS_URL = os.getenv("REDIS_URL", "redis://:redis_password@localhost:6379")


def create_checkpoint_index():
    """Create the checkpoints index that LangGraph expects"""
    
    # Connect to Redis
    r = redis.from_url(REDIS_URL)
    
    # Check if index already exists
    try:
        r.execute_command("FT.INFO", "checkpoints")
        print("✅ Index 'checkpoints' already exists")
        return
    except:
        print("📝 Creating 'checkpoints' index...")
    
    # Define schema for checkpoints index
    schema = {
        "index": {
            "name": "checkpoints",
            "prefix": "checkpoint:",
            "storage_type": "hash"
        },
        "fields": [
            {"name": "thread_id", "type": "tag"},
            {"name": "checkpoint_id", "type": "tag"},
            {"name": "thread_ts", "type": "numeric"},
            {"name": "parent_ts", "type": "numeric", "optional": True},
            {"name": "channel_values", "type": "text"},
            {"name": "channel_versions", "type": "text"},
            {"name": "versions_seen", "type": "text"},
            {"name": "pending_sends", "type": "text", "optional": True},
            {"name": "metadata", "type": "text", "optional": True}
        ]
    }
    
    # Create index using Redis command
    try:
        # Build FT.CREATE command
        cmd = [
            "FT.CREATE", "checkpoints",
            "ON", "HASH",
            "PREFIX", "1", "checkpoint:",
            "SCHEMA"
        ]
        
        # Add fields
        for field in schema["fields"]:
            cmd.extend([field["name"], "TAG" if field["type"] == "tag" else field["type"].upper()])
            if field.get("optional"):
                cmd.append("NOINDEX")
        
        r.execute_command(*cmd)
        print("✅ Index 'checkpoints' created successfully")
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")


def create_store_index():
    """Create the store index for long-term memory"""
    
    # Connect to Redis
    r = redis.from_url(REDIS_URL)
    
    # Check if index already exists
    try:
        r.execute_command("FT.INFO", "store")
        print("✅ Index 'store' already exists")
        return
    except:
        print("📝 Creating 'store' index...")
    
    # Create store index
    try:
        cmd = [
            "FT.CREATE", "store",
            "ON", "HASH",
            "PREFIX", "1", "store:",
            "SCHEMA",
            "namespace", "TAG",
            "key", "TAG",
            "value", "TEXT",
            "created_at", "NUMERIC",
            "updated_at", "NUMERIC"
        ]
        
        r.execute_command(*cmd)
        print("✅ Index 'store' created successfully")
        
    except Exception as e:
        print(f"❌ Error creating store index: {e}")


def main():
    print("\n" + "="*60)
    print("🔧 REDIS INDEX FIX")
    print("="*60)
    
    # Test Redis connection
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("\nPlease start Redis with: docker-compose up redis")
        return
    
    # Create indexes
    print("\n📋 Creating required indexes...")
    create_checkpoint_index()
    create_store_index()
    
    # List all indexes
    print("\n📋 Current indexes:")
    try:
        indexes = r.execute_command("FT._LIST")
        for idx in indexes:
            print(f"   - {idx.decode() if isinstance(idx, bytes) else idx}")
    except Exception as e:
        print(f"   Error listing indexes: {e}")
    
    print("\n✅ Index setup complete!")
    print("\nYou can now run redis_memory_demo.py")


if __name__ == "__main__":
    main()