// MongoDB Initialization Script for Core Agent
// This script sets up the database and collections for LangGraph

// Switch to the main database
db = db.getSiblingDB('core_agent_db');

// Create a user specifically for the core agent application
db.createUser({
  user: 'core_agent_user',
  pwd: 'mongo_app_password',
  roles: [
    { role: 'readWrite', db: 'core_agent_db' },
    { role: 'readWrite', db: 'core_agent_test_db' }
  ]
});

// Create collections for LangGraph checkpointing
db.createCollection('checkpoints', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['thread_id', 'checkpoint'],
      properties: {
        thread_id: { bsonType: 'string' },
        checkpoint: { bsonType: 'object' },
        created_at: { bsonType: 'date' },
        updated_at: { bsonType: 'date' }
      }
    }
  }
});

// Create collections for LangGraph storage
db.createCollection('memories', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['namespace', 'key', 'value'],
      properties: {
        namespace: { bsonType: 'string' },
        key: { bsonType: 'string' },
        value: { bsonType: 'object' },
        created_at: { bsonType: 'date' },
        updated_at: { bsonType: 'date' },
        ttl: { bsonType: 'date' }
      }
    }
  }
});

// Create indexes for better performance
db.checkpoints.createIndex({ thread_id: 1 });
db.checkpoints.createIndex({ created_at: 1 });
db.checkpoints.createIndex({ updated_at: 1 });

db.memories.createIndex({ namespace: 1, key: 1 }, { unique: true });
db.memories.createIndex({ namespace: 1 });
db.memories.createIndex({ ttl: 1 }, { expireAfterSeconds: 0 }); // TTL index

// Switch to test database and setup
db = db.getSiblingDB('core_agent_test_db');

// Create the same collections for test database
db.createCollection('checkpoints');
db.createCollection('memories');

// Create indexes for test database
db.checkpoints.createIndex({ thread_id: 1 });
db.memories.createIndex({ namespace: 1, key: 1 }, { unique: true });
db.memories.createIndex({ ttl: 1 }, { expireAfterSeconds: 0 });

// Log successful initialization
print('MongoDB databases and collections initialized successfully for Core Agent');