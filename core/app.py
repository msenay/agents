#!/usr/bin/env python3
"""
Core Agent Application Server
============================

Simple FastAPI server to run and test Core Agent with all backends.
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Core Agent API",
    description="Core Agent with Redis, PostgreSQL, and MongoDB support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test data models
class TestConfigRequest(BaseModel):
    backend: str = "inmemory"
    enable_memory: bool = True
    memory_types: list = ["short_term"]

class AgentRequest(BaseModel):
    message: str
    config: Optional[TestConfigRequest] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Core Agent API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "test_config": "/test/config",
            "test_redis": "/test/redis",
            "test_postgres": "/test/postgres",
            "test_mongodb": "/test/mongodb"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic import
        from core.config import AgentConfig
        
        return {
            "status": "healthy",
            "services": {
                "config": "available",
                "redis_url": os.getenv("REDIS_URL", "not_set"),
                "postgres_url": os.getenv("POSTGRES_URL", "not_set")[:50] + "..." if os.getenv("POSTGRES_URL") else "not_set",
                "mongodb_url": os.getenv("MONGODB_URL", "not_set")[:50] + "..." if os.getenv("MONGODB_URL") else "not_set"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/test/config")
async def test_config(request: TestConfigRequest):
    """Test basic config creation"""
    try:
        from core.config import AgentConfig
        
        config = AgentConfig(
            name="TestAgent",
            enable_memory=request.enable_memory,
            memory_backend=request.backend,
            memory_types=request.memory_types
        )
        
        return {
            "status": "success",
            "config": {
                "name": config.name,
                "memory_enabled": config.enable_memory,
                "memory_backend": config.memory_backend,
                "memory_types": config.memory_types
            }
        }
    except Exception as e:
        logger.error(f"Config test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Config test failed: {str(e)}")

@app.get("/test/redis")
async def test_redis():
    """Test Redis connection"""
    try:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            raise ValueError("REDIS_URL not configured")
            
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        
        # Test basic operations
        r.set("test_key", "test_value", ex=60)
        value = r.get("test_key")
        
        return {
            "status": "success",
            "redis_url": redis_url[:30] + "...",
            "test_result": value.decode() if value else None
        }
    except Exception as e:
        logger.error(f"Redis test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Redis test failed: {str(e)}")

@app.get("/test/postgres")
async def test_postgres():
    """Test PostgreSQL connection"""
    try:
        postgres_url = os.getenv("POSTGRES_URL")
        if not postgres_url:
            raise ValueError("POSTGRES_URL not configured")
            
        import psycopg2
        conn = psycopg2.connect(postgres_url)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()
        cur.close()
        conn.close()
        
        return {
            "status": "success",
            "postgres_url": postgres_url[:50] + "...",
            "version": version[0] if version else None
        }
    except Exception as e:
        logger.error(f"PostgreSQL test failed: {e}")
        raise HTTPException(status_code=500, detail=f"PostgreSQL test failed: {str(e)}")

@app.get("/test/mongodb")
async def test_mongodb():
    """Test MongoDB connection"""
    try:
        mongodb_url = os.getenv("MONGODB_URL")
        if not mongodb_url:
            raise ValueError("MONGODB_URL not configured")
            
        from pymongo import MongoClient
        client = MongoClient(mongodb_url)
        
        # Test connection
        admin_db = client.admin
        server_info = admin_db.command("ping")
        
        # Test database operations
        db = client.core_agent_db
        collection = db.test_collection
        
        # Insert test document
        result = collection.insert_one({"test": "value", "timestamp": "now"})
        
        # Clean up
        collection.delete_one({"_id": result.inserted_id})
        client.close()
        
        return {
            "status": "success",
            "mongodb_url": mongodb_url[:50] + "...",
            "ping_result": server_info
        }
    except Exception as e:
        logger.error(f"MongoDB test failed: {e}")
        raise HTTPException(status_code=500, detail=f"MongoDB test failed: {str(e)}")

@app.post("/agent/chat")
async def chat_with_agent(request: AgentRequest):
    """Simple chat endpoint for testing Core Agent"""
    try:
        # This would use the actual Core Agent when all dependencies are available
        return {
            "status": "success",
            "message": f"Received: {request.message}",
            "note": "This is a mock response. Full agent integration requires all LangGraph dependencies."
        }
    except Exception as e:
        logger.error(f"Agent chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent chat failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Core Agent API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)