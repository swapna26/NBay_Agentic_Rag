# Agentic RAG Backend

Production-ready FastAPI backend for Retrieval-Augmented Generation with integrated conversation memory and CrewAI agents. Features OpenWebUI compatibility for seamless chat interface integration and intelligent context-aware responses.

## Architecture Overview

The backend implements an intelligent chatbot system with conversation memory and multi-modal RAG processing:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OpenWebUI     │    │   FastAPI        │    │   PostgreSQL    │
│   Chat Client   │◄──►│   Backend        │◄──►│   + PGVector    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Conversation     │
                       │ Memory Engine    │
                       └──────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ Chat Engine  │ │  CrewAI      │ │ Query Engine │
            │ (Primary)    │ │  Agents      │ │ (Fallback)  
            └──────────────┘ └──────────────┘ └──────────────┘
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                       ┌──────────────────┐
                       │   Ollama LLM     │
                       │   + Embeddings   │
                       └──────────────────┘
```

## 🚀 New Conversation Memory System

### Key Features

- **Automatic Context Management**: LLM automatically understands conversation flow without manual topic detection
- **LlamaIndex Chat Engine**: Built-in conversation memory with token management
- **Context-Aware Responses**: Follow-up questions are answered with full conversation context
- **No Topic Detection Overhead**: Removed complex topic change detection logic

### How It Works

1. **Message History Processing**: OpenWebUI conversation history is automatically converted to LlamaIndex ChatMemoryBuffer
2. **Context Population**: Each request populates the chat engine with previous conversation messages
3. **Intelligent Retrieval**: LLM determines what documents to retrieve based on conversation context
4. **Memory Management**: Token-limited conversation memory prevents context overflow

## Core Components

### 🎯 RAG Service (`services/rag_service.py`)

The central orchestrator with conversation-aware processing:

**New Methods:**
- `chat()` - **Primary method with conversation memory**
- `query()` - Legacy method (redirects to chat with empty history)
- `stream_chat()` - Streaming with conversation context
- `_populate_memory_from_history()` - Converts OpenWebUI history to LlamaIndex memory

**Processing Flow:**
1. **Chat Engine (Primary)** - Uses LlamaIndex conversation memory for context-aware responses
2. **CrewAI Agents (Fallback)** - Multi-agent processing when chat engine fails
3. **Query Engine (Final Fallback)** - Direct RAG without conversation context

### 🧠 Conversation Memory Architecture

**Memory Components:**
- **ChatMemoryBuffer**: LlamaIndex conversation state management
- **Chat Engine**: Context-aware document retrieval and response generation
- **PostgreSQL Storage**: Persistent conversation history across sessions

**Memory Flow:**
```
OpenWebUI Messages → Conversation History → ChatMemoryBuffer → Chat Engine → Context-Aware Response
```

### 🤖 CrewAI Agents (`agents/crew_agents.py`)

Multi-agent system for complex query processing:

**Agent Roles:**
- **Query Analyzer** - Analyzes conversation context and determines search strategy
- **Document Retrieval Specialist** - Context-aware document search
- **Information Extractor** - Synthesizes responses based on conversation flow
- **Response Formatter** - Ensures proper formatting and completeness

### 🌐 Chat Router (`routers/chat.py`)

**Updated Features:**
- **Conversation History Extraction**: Automatically extracts message history from OpenWebUI format
- **Memory Integration**: Passes conversation context to RAG service
- **Simplified Routing**: Removed complex topic change detection logic

## 🔄 Request Flow

### Conversation-Aware Chat Flow

```
1. OpenWebUI Request
   │
   ▼
2. Extract Conversation History
   ├── Current Message: "Give me a summary"
   └── Previous Context: [{"role": "user", "content": "What is procurement?"}, ...]
   │
   ▼
3. Populate Chat Memory
   └── ChatMemoryBuffer.put(previous_messages)
   │
   ▼
4. Context-Aware Processing
   ├── Chat Engine: Uses memory for context-aware retrieval
   ├── CrewAI Fallback: Agents understand conversation flow
   └── Query Engine: Direct processing without context
   │
   ▼
5. Intelligent Response
   └── LLM automatically relates current question to previous context
```

### Processing Modes

| Mode | Trigger | Memory | Speed | Use Case |
|------|---------|--------|-------|----------|
| Chat Engine | Default | ✅ Full | Fast (1-5s) | Conversational Q&A |
| CrewAI Agents | Engine fails | ✅ Full | Medium (10-30s) | Complex analysis |
| Query Engine | All fail | ❌ None | Fast (1-2s) | Simple retrieval |

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- PostgreSQL with PGVector extension
- Ollama with required models

### Required Ollama Models

```bash
ollama pull llama3.2:1b              # Main LLM
ollama pull nomic-embed-text:v1.5    # Embedding model
```

### Environment Variables

Create `.env` file:

```env
# Database
DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/agentic_rag

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*

# RAG Settings
SIMILARITY_TOP_K=5
MAX_TOKENS=4000
TEMPERATURE=0.1

# Conversation Memory
CONVERSATION_TOKEN_LIMIT=2000
MEMORY_CLEANUP_INTERVAL=100

# CrewAI Settings
CREW_VERBOSE=false
CREW_MEMORY=false

# Phoenix Observability (Optional)
PHOENIX_BASE_URL=http://localhost:6006
PHOENIX_PROJECT_NAME=agentic_rag_backend
```

### Running the Backend

```bash
# Using Docker Compose
docker-compose up rag_backend

# Local Development
pip install -r requirements.txt
python main.py
```

## 📡 API Endpoints

### Chat Completions (OpenWebUI Compatible)

```bash
POST /api/chat/completions
```

**Conversation Example:**
```json
{
  "model": "agentic-rag-ollama",
  "messages": [
    {"role": "user", "content": "What is procurement?"},
    {"role": "assistant", "content": "Procurement is the process of..."},
    {"role": "user", "content": "Give me a 3-line summary"}
  ],
  "stream": false
}
```

The system automatically:
1. Extracts the current question: "Give me a 3-line summary"
2. Loads previous context: "What is procurement?" + previous response
3. Understands the user wants a summary of procurement (not a generic summary)
4. Returns a 3-line summary of procurement specifically

### Conversation Management

```bash
# Get conversation history
GET /api/chat/conversations/{conversation_id}

# Clear conversation memory
DELETE /api/chat/conversations/{conversation_id}

# Chat service health (includes memory status)
GET /api/chat/health
```

## 💬 Conversation Examples

### Example 1: Follow-up Questions

**User:** "What are the main procurement policies?"
**Assistant:** *Provides detailed procurement policy information*

**User:** "Can you summarize that in 5 bullet points?"
**Assistant:** *Automatically understands "that" refers to procurement policies and provides 5 bullet points*

### Example 2: Context Switching

**User:** "Tell me about data protection requirements"
**Assistant:** *Provides data protection information*

**User:** "How does this relate to procurement?"
**Assistant:** *Understands context and explains relationship between data protection and procurement*

### Example 3: Clarification Requests

**User:** "What is the approval process?"
**Assistant:** *Asks for clarification about which approval process*

**User:** "For procurement requests over $10,000"
**Assistant:** *Provides specific procurement approval process information*

## ⚙️ Configuration

### Memory Management

```env
# Conversation memory settings
MAX_TOKENS=4000                    # Total token limit
CONVERSATION_TOKEN_LIMIT=2000      # Reserve half for conversation memory
MEMORY_CLEANUP_THRESHOLD=10        # Clean memory after N conversations

# Chat engine settings
CHAT_MODE=context                  # Use context-aware chat mode
CHAT_ENGINE_VERBOSE=true           # Enable detailed logging
```

### Performance Optimization

```python
# Fast conversation responses
SIMILARITY_TOP_K=3
CONVERSATION_TOKEN_LIMIT=1000

# Balanced performance
SIMILARITY_TOP_K=5
CONVERSATION_TOKEN_LIMIT=2000

# Maximum context retention
SIMILARITY_TOP_K=10
CONVERSATION_TOKEN_LIMIT=3000
```

## 🔧 Development

### Testing Conversation Memory

```bash
# Test conversation flow
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agentic-rag-ollama",
    "messages": [
      {"role": "user", "content": "What is procurement?"},
      {"role": "assistant", "content": "Procurement is..."},
      {"role": "user", "content": "Give me a brief summary"}
    ]
  }'
```

### Custom Memory Integration

```python
from services.rag_service import RAGService

# Direct conversation usage
rag_service = RAGService(config)
await rag_service.initialize()

# Chat with conversation history
conversation_history = [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous response"}
]

result = await rag_service.chat(
    "Current question",
    conversation_history,
    "conversation_id"
)
```

### Adding Custom Memory Features

```python
# Custom memory population
async def populate_custom_memory(self, custom_history):
    for msg in custom_history:
        chat_msg = ChatMessage(
            role=msg["role"],
            content=msg["content"]
        )
        self.memory.put(chat_msg)
```

## 📊 Monitoring & Observability

### Conversation Metrics

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "event": "conversation_processed",
  "conversation_id": "conv_123",
  "message_count": 3,
  "has_conversation_context": true,
  "processing_mode": "chat_engine",
  "memory_token_usage": 1250,
  "response_time_ms": 1500
}
```

### Memory Usage Tracking

- **Memory Population**: Track conversation history loading
- **Token Usage**: Monitor memory token consumption
- **Context Success Rate**: Measure conversation understanding accuracy
- **Fallback Frequency**: Track chat engine vs agent usage

## 🚨 Troubleshooting

### Conversation Issues

**Missing Context:**
- Check conversation history extraction in chat router
- Verify ChatMemoryBuffer population
- Review memory token limits

**Incorrect Follow-ups:**
- Chat engine may have failed, check CrewAI agent fallback
- Verify conversation ID consistency
- Check memory reset between different conversations

**Memory Overflow:**
- Reduce `CONVERSATION_TOKEN_LIMIT`
- Implement conversation history truncation
- Monitor token usage in logs

### Debug Mode

```env
LOG_LEVEL=DEBUG
CHAT_ENGINE_VERBOSE=true
MEMORY_DEBUG=true
```

## 🎯 Key Improvements Over Previous Architecture

1. **Simplified Logic**: Removed complex topic change detection - LLM handles context naturally
2. **Better Context Understanding**: Full conversation history provides richer context
3. **Automatic Memory Management**: Token-limited memory prevents overflow
4. **Intelligent Fallback**: Graceful degradation from chat engine → agents → query engine
5. **Real Chatbot Experience**: Natural conversation flow like ChatGPT with your documents

## 🔗 Integration

### OpenWebUI Setup

1. **Add Backend URL**: `http://localhost:8000`
2. **Select Model**: `agentic-rag-ollama`
3. **Start Conversation**: Natural back-and-forth chat with document context
4. **Follow-up Questions**: Ask clarifications, request different formats, dive deeper

The system now provides a true chatbot experience where you can have natural conversations with your documents, ask follow-up questions, and the AI will maintain context throughout the conversation.

## 📄 License

Part of the Agentic RAG System. See main project for license information.

---

**Built for natural document conversations with enterprise-grade accuracy and intelligent memory management**