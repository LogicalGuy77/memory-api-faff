import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from urllib.parse import urlparse

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    # Render provides postgres:// but psycopg2 expects postgresql://
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

@contextmanager
def get_db():
    """Get database connection with proper error handling"""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def init_database():
    """Initialize PostgreSQL database with required tables"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            # Create chat_messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    sender TEXT,
                    content TEXT,
                    chat_id TEXT
                )
            """)
            
            # Create memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT,
                    memory_type TEXT,
                    confidence REAL,
                    source_messages JSONB,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    chat_id TEXT,
                    content_hash TEXT,
                    extraction_method TEXT DEFAULT 'rules',
                    reasoning TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages(chat_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_chat_id ON memories(chat_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_method ON memories(extraction_method)")
            
        conn.commit()
        print("Database initialized successfully")