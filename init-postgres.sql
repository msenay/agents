-- PostgreSQL Initialization Script for Core Agent
-- This script sets up the database and tables for LangGraph checkpointing and storage

-- Create test database
CREATE DATABASE core_agent_test_db OWNER core_agent_user;

-- Connect to main database
\c core_agent_db;

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector" CASCADE;  -- For vector operations if needed

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE core_agent_db TO core_agent_user;
GRANT ALL PRIVILEGES ON DATABASE core_agent_test_db TO core_agent_user;

-- Create schema for LangGraph
CREATE SCHEMA IF NOT EXISTS langgraph AUTHORIZATION core_agent_user;

-- Set default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA langgraph GRANT ALL ON TABLES TO core_agent_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA langgraph GRANT ALL ON SEQUENCES TO core_agent_user;

-- Connect to test database and setup
\c core_agent_test_db;

-- Enable extensions for test database too
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector" CASCADE;

-- Create schema for LangGraph in test DB
CREATE SCHEMA IF NOT EXISTS langgraph AUTHORIZATION core_agent_user;

-- Set default privileges for test DB
ALTER DEFAULT PRIVILEGES IN SCHEMA langgraph GRANT ALL ON TABLES TO core_agent_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA langgraph GRANT ALL ON SEQUENCES TO core_agent_user;

-- Log successful initialization
SELECT 'PostgreSQL databases initialized successfully for Core Agent' AS status;