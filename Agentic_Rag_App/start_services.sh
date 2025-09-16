#!/bin/bash

echo "🚀 Starting Agentic RAG Core Services..."

# Start core services first
echo "Starting PostgreSQL, Ollama, and Phoenix..."
docker-compose up -d postgres ollama phoenix

# Wait for services to be healthy
echo "Waiting for PostgreSQL to be ready..."
while ! docker-compose exec postgres pg_isready -U raguser -d agentic_rag > /dev/null 2>&1; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done

echo "✅ PostgreSQL is ready"

# Check if Ollama is responding
echo "Checking Ollama service..."
until curl -f http://localhost:11434/api/version > /dev/null 2>&1; do
    echo "Waiting for Ollama..."
    sleep 2
done

echo "✅ Ollama is ready"

# Start OpenWebUI
echo "Starting OpenWebUI..."
docker-compose up -d openwebui

echo "🎉 Core services are running!"
echo "📊 Phoenix UI: http://localhost:6006"
echo "🤖 OpenWebUI: http://localhost:3000"
echo "🗄️ PostgreSQL: localhost:5433"
echo "🦙 Ollama API: http://localhost:11434"

echo ""
echo "Next steps:"
echo "1. Pull a model: docker-compose exec ollama ollama pull llama3.1:8b"
echo "2. Pull embedding model: docker-compose exec ollama ollama pull nomic-embed-text"