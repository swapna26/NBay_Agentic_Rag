"""Conversation models for PostgreSQL storage."""

from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

Base = declarative_base()


class Conversation(Base):
    """Conversation table to store conversation metadata."""

    __tablename__ = "conversations"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)  # For future user management
    title = Column(String(500), nullable=True)  # Auto-generated title
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    extra_metadata = Column(Text, nullable=True)  # JSON metadata

    # Indexes for performance
    __table_args__ = (
        Index('idx_conversations_user_id', 'user_id'),
        Index('idx_conversations_created_at', 'created_at'),
        Index('idx_conversations_updated_at', 'updated_at'),
    )


class ConversationMessage(Base):
    """Messages within conversations."""

    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Response metadata for assistant messages
    sources = Column(Text, nullable=True)  # JSON array of sources
    processing_mode = Column(String(50), nullable=True)  # 'crew_ai', 'regular_rag', 'greeting'
    model_used = Column(String(100), nullable=True)
    token_count = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index('idx_messages_conversation_id', 'conversation_id'),
        Index('idx_messages_created_at', 'created_at'),
        Index('idx_messages_role', 'role'),
    )


class ConversationSummary(Base):
    """Periodic summaries of conversations for efficient context retrieval."""

    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    message_count = Column(Integer, nullable=False)  # Number of messages summarized
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Indexes
    __table_args__ = (
        Index('idx_summaries_conversation_id', 'conversation_id'),
        Index('idx_summaries_created_at', 'created_at'),
    )