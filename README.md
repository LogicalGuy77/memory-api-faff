# Technical Overview: Memory Extraction and Management System

This system is designed to extract, store, and manage meaningful "memories" from chat conversations. It employs a hybrid approach, combining rule-based methods with advanced Large Language Model (LLM) capabilities for nuanced understanding and intelligent memory handling. The system is built using Python, FastAPI, and Google's Gemini LLM, with a PostgreSQL database backend (as suggested by `database.py` and psycopg2 usage, though not explicitly detailed in `main.py`).

## Core Functionality

The system revolves around the following key processes:

1.  **Chat Message Ingestion**: Accepts and stores chat messages, forming the basis for memory extraction.
2.  **Memory Extraction**: Identifies and extracts pertinent pieces of information (memories) from the conversations.
3.  **Intelligent Memory Management**: Stores, updates, and deduplicates memories dynamically, ensuring relevance and accuracy.
4.  **Memory Retrieval**: Allows querying of stored memories based on various criteria.

## 1. Memory Extraction Engine (`EnhancedMemoryExtractor`)

The system uses an `EnhancedMemoryExtractor` that employs a two-pronged strategy for extracting memories:

* **Rule-Based Extraction**:
    * Utilizes predefined keywords and regular expression patterns (`MEMORY_TYPES` configuration) to identify potential memories related to categories like food preferences, travel habits, personal info, etc.
    * This method serves as a quick first pass and a reliable fallback.
    * Confidence scores are calculated based on explicit keywords and the presence of temporal indicators or questions.

* **LLM-Powered Extraction (Gemini)**:
    * Messages are processed in batches (with overlap for context retention) and sent to the Gemini LLM (`gemini-2.0-flash-lite` model).
    * A structured prompt guides the LLM to analyze conversation snippets and extract memories based on defined categories (`MEMORY_TYPES`). The prompt instructs the LLM to focus on persistent facts, preferences, and habits, assigning a memory type, content, and a confidence score.
    * The LLM is also asked to provide a `reasoning` for why a piece of information is considered a valuable memory.
    * The LLM's response, expected in JSON format, is parsed to create structured memory objects.
    * If LLM extraction fails for a batch, the system gracefully falls back to rule-based extraction for that batch.

## 2. LLM Reasoning and Semantic Understanding

The Gemini LLM plays a crucial role beyond simple extraction:

* **Contextual Understanding**: It analyzes the conversational context to infer meaning and identify information likely to be a long-term memory.
* **Confidence Scoring**: The LLM assigns a confidence level (0.1 to 1.0) to each extracted memory, indicating its perceived reliability.
* **Semantic Deduplication**:
    * Before final storage, extracted memories (from both rule-based and LLM methods) are grouped by type.
    * Within each type, another LLM call is made to compare the textual content of memories for semantic similarity.
    * The LLM identifies groups of similar or duplicate memories, suggests a representative index, and can provide merged content. This helps in consolidating redundant information and choosing the most comprehensive version.
    * A simpler hash-based deduplication acts as a fallback if semantic deduplication encounters issues.
* **Conflict Resolution in Updates**: When a newly extracted memory might conflict with an existing one, the LLM is prompted to decide the best course of action:
    * `replace`: If the new memory is an update (e.g., a new address).
    * `merge`: If the memories can be combined for more complete information.
    * `keep_both` (or create new): If they are distinct enough.
    * Special logic is included for address updates, often favoring the newer information.

## 3. Dynamic Database Interaction (`MemoryUpdateManager` & `MemoryManager`)

The system interacts with a database (managed via functions in `database.py`) to persist and manage memories.

* **Storing Chat Messages**: Uploaded chat messages are stored in a `chat_messages` table.
* **Storing Memories**: Extracted memories are stored in a `memories` table, including fields like `memory_id`, `content`, `memory_type`, `confidence`, `source_messages` (linking back to original chat messages), timestamps, `chat_id`, `extraction_method`, and `reasoning`.

The `MemoryUpdateManager` is responsible for intelligently integrating newly extracted memories with existing ones:

* **Fetching Existing Memories**: Retrieves current memories for a given `chat_id`.
* **Processing New Memories**: For each newly extracted memory:
    * **Similarity Detection**: It first filters existing memories by `memory_type`. Then, it employs an "enhanced" similarity check:
        * **Address-Specific Logic**: If the new memory is `personal_info` and contains address keywords, it specifically looks for existing address memories, assigning a high similarity score to facilitate updates.
        * **LLM-Based Similarity**: For other cases or as a fallback, it prompts the LLM to compare the new memory content with existing ones of the same type, providing a similarity score and reasoning.
    * **Decision Logic**:
        * **High Similarity (>0.7)**: Triggers conflict resolution (using the LLM, as described above) to decide whether to update an existing memory (e.g., replace content, merge), or if the new memory should still be created separately.
        * **Low Similarity**: The new memory is considered distinct and is created.
    * **Database Operations**: Based on the decision, it performs:
        * `_create_memory()`: Inserts a new memory record.
        * `_update_memory()`: Modifies an existing memory record (e.g., content, confidence, `source_messages`, `updated_at` timestamp).
        * `_delete_similar_memories()`: Removes redundant memories after an update/merge to maintain a clean dataset.

## 4. Detecting and Updating Memories

The core of the system's intelligence lies in its ability to recognize when a new piece of information relates to, updates, or duplicates an existing memory.

1.  **Extraction**: New information is extracted as potential memories (as detailed in Section 1).
2.  **Comparison**: Each new potential memory is compared against the existing memory bank for the specific user/chat (`_find_similar_memories_enhanced`):
    * It prioritizes matching by `memory_type`.
    * It uses special heuristics for common update scenarios like addresses.
    * For general cases, it leverages the LLM to assess semantic similarity between the new memory's content and existing memories' content.
3.  **Conflict Resolution (`_resolve_conflict`)**: If a sufficiently similar existing memory is found, the LLM is invoked to analyze both the existing and the new memory. The LLM considers their content, confidence, and creation/extraction details to recommend an action:
    * **Replace**: If the new memory is deemed a more current or accurate version (e.g., "My new address is..." vs. "My old address was..."). The system updates the existing memory's content, confidence, and source messages.
    * **Merge**: If the new memory adds complementary information to the existing one. The content might be combined, and confidence/sources updated.
    * **Keep Existing / Create New**: If the LLM deems them distinct despite some similarity, or if the new memory is preferred due to higher confidence or recency and the existing one is less relevant.
4.  **Database Update**: The `MemoryUpdateManager` then executes the chosen action, either creating a new memory entry, updating an existing one, or in some cases, deleting redundant older entries. Timestamps (`updated_at`) are crucial for tracking memory evolution.

This dynamic process ensures that the memory store evolves with the conversation, reflecting the most current and relevant information while minimizing redundancy.

## API Endpoints

The system exposes several FastAPI endpoints for interaction:

* `/api/chat/upload`: To upload new chat messages.
* `/api/memories/extract/{chat_id}`: To trigger memory extraction and intelligent updating for a specific chat.
* `/api/memories/{chat_id}`: To retrieve all memories for a chat.
* `/api/memories/query`: To search for memories based on query text, chat ID, and memory types.
* `/api/chats`: To get a list of all chats and their metadata.
* `/api/memory-types`: To get the list of configured memory types.
* `/api/memories/cleanup/{chat_id}`: A specific endpoint to clean up duplicate address memories by keeping the most recent one.
* `/health`: A health check endpoint.

## Key Technologies

* **Python**: Core programming language.
* **FastAPI**: For building the asynchronous API.
* **Pydantic**: For data validation and settings management.
* **Google Gemini LLM (`gemini-2.0-flash-lite`)**: For natural language understanding, memory extraction, semantic similarity, and conflict resolution.
* **Psycopg2 (implied for database)**: For interacting with a PostgreSQL database.
* **Uvicorn**: ASGI server for running the FastAPI application.
* **Dotenv**: For managing environment variables.
