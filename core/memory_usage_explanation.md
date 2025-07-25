# Memory Usage in Core Agent - How It Actually Works

## 🎯 The Key Question: How is memory used with LLM?

### 1. **Short-term Memory (Conversation Memory)**

**How it's stored:**
- Uses LangGraph's checkpointer system
- Automatically saves conversation state after each turn

**How it's used with LLM:**
- ✅ **AUTOMATICALLY loaded and sent to LLM**
- When you call `invoke()` with a `thread_id`, LangGraph:
  1. Loads all previous messages from that thread
  2. Adds them to the current state
  3. Sends the ENTIRE conversation history to the LLM

```python
# Example flow:
agent.invoke("My name is John", config={"configurable": {"thread_id": "user_123"}})
# LLM sees: [HumanMessage("My name is John")]

agent.invoke("What's my name?", config={"configurable": {"thread_id": "user_123"}})
# LLM sees: [
#   HumanMessage("My name is John"),
#   AIMessage("Hello John!"),
#   HumanMessage("What's my name?")
# ]
```

**Code proof:**
```python
# In core_agent.py
kwargs['checkpointer'] = self.memory_manager.get_checkpointer()
self.compiled_graph = create_react_agent(**kwargs)
```

The checkpointer is built into the graph compilation, so it automatically manages state.

### 2. **Long-term Memory (Knowledge Store)**

**How it's stored:**
- Uses LangGraph's store system
- Manual save/load operations
- Key-value pairs

**How it's used with LLM:**
- ❌ **NOT automatically sent to LLM**
- ✅ **Available through tools or manual retrieval**

Three ways to use it:

#### Option A: Through Memory Tools
```python
# If enable_memory_tools=True, agent gets these tools:
- manage_memory: Save information
- search_memory: Retrieve information

# The agent can CHOOSE to use these tools
agent.invoke("Remember that my favorite color is blue")
# Agent might call: manage_memory(key="user_prefs", value={"color": "blue"})
```

#### Option B: Manual in System Prompt
```python
# Retrieve manually and add to prompt
user_data = agent.memory_manager.get_long_term_memory("user_123")
enriched_prompt = f"User info: {user_data}\n\nQuery: What's my favorite color?"
agent.invoke(enriched_prompt)
```

#### Option C: Custom Tool
```python
@tool
def get_user_context(user_id: str) -> str:
    """Get user context from long-term memory"""
    data = agent.memory_manager.get_long_term_memory(f"user_{user_id}")
    return f"User context: {data}"

# Agent can use this tool when needed
```

### 3. **Semantic Memory (Vector Search)**

**How it's stored:**
- Stored as embeddings in vector database
- Part of long-term memory with vector index

**How it's used with LLM:**
- ❌ **NOT automatically sent to LLM**
- ✅ **Available through search**

```python
# Search for relevant information
results = agent.memory_manager.search_memory("travel experiences", limit=3)

# Option 1: Use in tool
@tool
def search_knowledge(query: str) -> str:
    results = agent.memory_manager.search_memory(query, limit=5)
    return f"Relevant information: {results}"

# Option 2: Manually add to context
context = agent.memory_manager.search_memory("user preferences", limit=3)
agent.invoke(f"Context: {context}\n\nUser query: {query}")
```

## 📊 Summary Table

| Memory Type | Automatic to LLM? | How to Access | Use Case |
|-------------|------------------|---------------|----------|
| **Short-term** | ✅ YES | Automatic via thread_id | Conversations |
| **Long-term** | ❌ NO | Tools or manual | Persistent data |
| **Semantic** | ❌ NO | Search function | Knowledge retrieval |
| **Session** | ❌ NO | Manual between agents | Multi-agent sharing |

## 🔍 Detailed Flow Example

```python
# 1. SETUP
agent = CoreAgent(AgentConfig(
    name="Assistant",
    model=model,
    enable_memory=True,
    memory_types=["short_term", "long_term", "semantic"],
    memory_backend="redis",
    enable_memory_tools=True  # Gives agent access to memory tools
))

# 2. FIRST INTERACTION
response = agent.invoke(
    "My name is Alice and I love hiking",
    config={"configurable": {"thread_id": "alice_chat"}}
)
# Short-term: Automatically saved
# Long-term: Agent might use tool to save {"name": "Alice", "interests": ["hiking"]}

# 3. SECOND INTERACTION - Same Thread
response = agent.invoke(
    "What did I tell you about myself?",
    config={"configurable": {"thread_id": "alice_chat"}}
)
# LLM sees ALL previous messages automatically
# Can also use memory tools to search long-term memory

# 4. DIFFERENT THREAD - No automatic context
response = agent.invoke(
    "Who am I?",
    config={"configurable": {"thread_id": "different_thread"}}
)
# LLM sees ONLY this message (no previous context)
# But can still use memory tools to search long-term memory
```

## 🎓 Key Insights

1. **Only short-term memory is automatic** - It's the conversation history
2. **Long-term and semantic require explicit access** - Through tools or manual code
3. **This is by design** - Not all stored data should go to every LLM call
4. **Context window management** - Sending everything would be expensive and hit limits
5. **Agent autonomy** - The agent can decide when to search long-term memory

## 💡 Best Practices

1. **For chatbots**: Rely mainly on short-term memory (automatic)
2. **For assistants**: Combine short-term + give access to memory tools
3. **For knowledge systems**: Heavy use of semantic search tools
4. **For context-heavy apps**: Manually retrieve and inject relevant long-term data

The key is: **Short-term = automatic conversation context, Long-term = on-demand knowledge**