# MongoDB Support for CoreAgent Framework

The CoreAgent framework now includes comprehensive MongoDB support for both short-term and long-term memory, providing a scalable and flexible memory backend for AI agents.

## Features

✅ **Short-term Memory** - MongoDBSaver for conversation persistence  
✅ **Long-term Memory** - MongoDBStore for cross-session data storage  
✅ **TTL Support** - Automatic data expiration  
✅ **Session-based Memory** - Multi-agent collaboration  
✅ **Semantic Search** - Vector similarity search with embeddings  
✅ **Performance Optimized** - Fast read/write operations  
✅ **Production Ready** - Works with MongoDB Atlas and local instances  

## Quick Start

### 1. Install Dependencies

```bash
# Install MongoDB Python driver
pip install pymongo

# Install LangGraph MongoDB packages
pip install langgraph-checkpoint-mongodb langgraph-store-mongodb

# Optional: Install MongoDB server locally
# See: https://docs.mongodb.com/manual/installation/
```

### 2. Start MongoDB

**Local MongoDB:**
```bash
# Start MongoDB service
mongod --dbpath /path/to/data/directory

# Or use Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

**MongoDB Atlas (Cloud):**
1. Create account at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create cluster and get connection string
3. Replace `mongodb://localhost:27017` with your Atlas URL

### 3. Basic Usage

```python
from core_agent import create_memory_agent
from langchain_openai import ChatOpenAI

# Create agent with MongoDB memory
model = ChatOpenAI(model="gpt-3.5-turbo")

agent = create_memory_agent(
    model=model,
    name="MongoAgent",
    short_term_memory="mongodb",      # Use MongoDB for conversations
    long_term_memory="mongodb",       # Use MongoDB for persistent data
    mongodb_url="mongodb://localhost:27017/coreagent",
    system_prompt="You are an AI assistant with MongoDB memory."
)

# Use the agent
config = {"configurable": {"thread_id": "my_conversation"}}
response = agent.invoke({
    "messages": [{"role": "user", "content": "Remember that I prefer Python programming"}]
}, config)
```

## Memory Types

### 1. Short-term Memory (MongoDBSaver)
**Purpose:** Conversation persistence within threads  
**Use case:** Multi-turn conversations, context continuity

```python
from core_agent import create_short_term_memory_agent

agent = create_short_term_memory_agent(
    model=model,
    memory_backend="mongodb",
    mongodb_url="mongodb://localhost:27017/coreagent"
)
```

### 2. Long-term Memory (MongoDBStore)
**Purpose:** Cross-session data persistence  
**Use case:** User profiles, knowledge base, persistent information

```python
from core_agent import create_long_term_memory_agent

agent = create_long_term_memory_agent(
    model=model,
    memory_backend="mongodb",
    mongodb_url="mongodb://localhost:27017/coreagent",
    enable_semantic_search=True  # Enable vector search
)
```

### 3. Comprehensive Memory
**Purpose:** Both short and long-term memory  
**Use case:** Full-featured AI assistants

```python
from core_agent import create_memory_agent

agent = create_memory_agent(
    model=model,
    short_term_memory="mongodb",
    long_term_memory="mongodb",
    mongodb_url="mongodb://localhost:27017/coreagent"
)
```

### 4. TTL Memory (Auto-expiration)
**Purpose:** Temporary data that expires automatically  
**Use case:** Privacy compliance, temporary sessions

```python
from core_agent import create_ttl_memory_agent

agent = create_ttl_memory_agent(
    model=model,
    memory_backend="mongodb",
    ttl_minutes=1440,  # 24 hours
    mongodb_url="mongodb://localhost:27017/coreagent"
)
```

### 5. Session-based Memory
**Purpose:** Multi-agent collaboration with shared memory  
**Use case:** Team workflows, agent collaboration

```python
from core_agent import create_session_agent

# Agent 1
coder_agent = create_session_agent(
    model=model,
    session_id="team_session_123",
    name="CoderAgent",
    memory_namespace="coder",
    mongodb_url="mongodb://localhost:27017/coreagent"
)

# Agent 2 (shares memory with Agent 1)
reviewer_agent = create_session_agent(
    model=model,
    session_id="team_session_123",  # Same session ID
    name="ReviewerAgent", 
    memory_namespace="reviewer",
    mongodb_url="mongodb://localhost:27017/coreagent"
)
```

## Configuration Options

### Connection String Formats

```python
# Local MongoDB
mongodb_url = "mongodb://localhost:27017/database_name"

# MongoDB with authentication
mongodb_url = "mongodb://username:password@localhost:27017/database_name"

# MongoDB Atlas (cloud)
mongodb_url = "mongodb+srv://username:password@cluster.mongodb.net/database_name"

# MongoDB with options
mongodb_url = "mongodb://localhost:27017/database_name?retryWrites=true&w=majority"
```

### AgentConfig Parameters

```python
from core_agent import AgentConfig, CoreAgent

config = AgentConfig(
    name="MongoAgent",
    model=model,
    
    # Memory configuration
    enable_short_term_memory=True,
    short_term_memory_type="mongodb",
    enable_long_term_memory=True,
    long_term_memory_type="mongodb",
    
    # MongoDB connection
    mongodb_url="mongodb://localhost:27017/coreagent",
    
    # TTL configuration
    enable_ttl=True,
    default_ttl_minutes=1440,  # 24 hours
    refresh_on_read=True,
    
    # Semantic search
    enable_semantic_search=True,
    embedding_model="openai:text-embedding-3-small",
    
    # Session memory
    session_id="optional_session_id",
    enable_shared_memory=True,
    memory_namespace="agent_namespace"
)

agent = CoreAgent(config)
```

## Advanced Features

### Semantic Search with Embeddings

```python
from core_agent import create_semantic_search_agent

agent = create_semantic_search_agent(
    model=model,
    memory_backend="mongodb",
    mongodb_url="mongodb://localhost:27017/coreagent",
    embedding_model="openai:text-embedding-3-small",
    embedding_dims=1536,
    distance_type="cosine"  # "cosine", "euclidean", "dot_product"
)

# Store and search information
config = {"configurable": {"thread_id": "knowledge_base"}}

# Store information
agent.invoke({
    "messages": [{"role": "user", "content": "Store: Python is a high-level programming language"}]
}, config)

# Search by semantic similarity
agent.invoke({
    "messages": [{"role": "user", "content": "Find information about programming languages"}]
}, config)
```

### Multi-backend Support

```python
# Different backends for different memory types
agent = create_memory_agent(
    model=model,
    short_term_memory="redis",        # Redis for conversations
    long_term_memory="mongodb",       # MongoDB for persistent data
    redis_url="redis://localhost:6379",
    mongodb_url="mongodb://localhost:27017/coreagent"
)
```

## Testing

### Run MongoDB Tests

```bash
# Make sure MongoDB is running
mongod

# Run basic tests
python test_mongodb_memory.py

# Run usage examples
python example_mongodb_usage.py
```

### Test Output Example

```
🚀 Starting MongoDB Memory Tests for CoreAgent Framework
============================================================

🔄 Testing MongoDB connection...
✅ MongoDB connection successful
   📍 Connected to: mongodb://localhost:27017/coreagent_test
   📊 Database: coreagent_test
   🔗 Ping: Success

🔄 Testing MongoDB short-term memory (checkpointer)...
✅ MongoDB short-term memory test successful
   📝 Thread ID: mongo_test_1703123456
   💾 Checkpointer: MongoDBSaver
   🔄 Memory persistence: Working

🔄 Testing MongoDB comprehensive memory...
✅ MongoDB comprehensive memory test successful
   📝 Thread ID: mongo_comprehensive_1703123456
   💾 Short-term: MongoDBSaver
   🗃️ Long-term: MongoDBStore
   🔧 Tools: Calculator

============================================================
📊 MongoDB Memory Test Summary:
============================================================
   connection     : ✅ PASS
   short_term     : ✅ PASS
   long_term      : ✅ PASS
   comprehensive  : ✅ PASS
   ttl            : ✅ PASS
   session        : ✅ PASS
   performance    : ✅ PASS
----------------------------------------
   Total: 7/7 tests passed (100.0%)

🎉 All MongoDB memory tests passed!
   MongoDB integration is working perfectly!
```

## Performance

### MongoDB vs Other Backends

| Backend    | Read Speed | Write Speed | Scalability | TTL Support | Vector Search |
|------------|------------|-------------|-------------|-------------|---------------|
| InMemory   | ⚡⚡⚡       | ⚡⚡⚡        | ❌          | ❌          | ❌            |
| Redis      | ⚡⚡        | ⚡⚡         | ✅          | ✅          | ❌            |
| PostgreSQL | ⚡         | ⚡          | ✅          | ❌          | ✅            |
| **MongoDB**| ⚡⚡        | ⚡⚡         | ✅          | ✅          | ✅            |

### Performance Tips

1. **Use MongoDB Atlas** for production deployments
2. **Enable indexing** for frequently queried fields
3. **Use TTL** for automatic cleanup of old data
4. **Configure read/write concerns** based on your consistency needs
5. **Use connection pooling** for high-traffic applications

## Production Deployment

### MongoDB Atlas Setup

1. **Create Atlas Account:** [MongoDB Atlas](https://www.mongodb.com/atlas)
2. **Create Cluster:** Choose your cloud provider and region
3. **Configure Access:** Set up database user and network access
4. **Get Connection String:** Copy the connection string
5. **Update Configuration:**

```python
# Production configuration
MONGODB_URL = "mongodb+srv://username:password@cluster.mongodb.net/production_db"

agent = create_memory_agent(
    model=model,
    short_term_memory="mongodb",
    long_term_memory="mongodb", 
    mongodb_url=MONGODB_URL,
    enable_ttl=True,
    default_ttl_minutes=10080  # 7 days
)
```

### Environment Variables

```bash
# Set in your environment
export MONGODB_URL="mongodb+srv://username:password@cluster.mongodb.net/production_db"
export OPENAI_API_KEY="your_openai_api_key"

# Use in your application
import os
mongodb_url = os.getenv("MONGODB_URL")
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   Error: MongoDB connection failed
   Solution: Check if MongoDB is running and connection string is correct
   ```

2. **Authentication Error**
   ```
   Error: Authentication failed
   Solution: Verify username/password in connection string
   ```

3. **Import Error**
   ```
   Error: No module named 'langgraph.checkpoint.mongodb'
   Solution: pip install langgraph-checkpoint-mongodb langgraph-store-mongodb
   ```

4. **TTL Not Working**
   ```
   Issue: Data not expiring
   Solution: TTL indexes need time to process; check MongoDB logs
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable MongoDB logging
import pymongo
pymongo.set_logging_level(logging.DEBUG)
```

## Migration

### From Redis to MongoDB

```python
# Old Redis configuration
old_agent = create_memory_agent(
    model=model,
    short_term_memory="redis",
    redis_url="redis://localhost:6379"
)

# New MongoDB configuration
new_agent = create_memory_agent(
    model=model,
    short_term_memory="mongodb",
    mongodb_url="mongodb://localhost:27017/coreagent"
)
```

### Data Migration Script

```python
# Migration utility (basic example)
import pymongo
import redis

def migrate_redis_to_mongodb():
    # Connect to Redis
    redis_client = redis.Redis.from_url("redis://localhost:6379")
    
    # Connect to MongoDB
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
    db = mongo_client["coreagent"]
    collection = db["migrated_data"]
    
    # Migrate data (implement based on your data structure)
    for key in redis_client.scan_iter():
        value = redis_client.get(key)
        doc = {"redis_key": key, "value": value}
        collection.insert_one(doc)
```

## Support

- **Documentation:** [LangGraph MongoDB](https://langchain-mongodb.readthedocs.io/)
- **Issues:** Create an issue in the CoreAgent repository
- **Community:** Join the LangChain Discord for MongoDB-related questions

## Roadmap

🔄 **In Development:**
- MongoDB GridFS support for large files
- Sharding configuration helpers
- Advanced indexing strategies
- Performance monitoring dashboard

✅ **Completed:**
- Basic MongoDB integration
- TTL support
- Session-based memory
- Semantic search with vectors
- Production deployment guides