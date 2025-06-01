from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import json
import uuid
import re
from dataclasses import dataclass
import hashlib
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn

# Import database functions
from database import get_db, init_database

# Load environment variables
load_dotenv()

app = FastAPI(title="Memory Extraction API")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "https://your-frontend-domain.onrender.com",  # Add your frontend URL
        "https://*.onrender.com"  # Allow all Render domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ChatMessage(BaseModel):
    message_id: str
    timestamp: datetime
    sender: str
    content: str
    chat_id: str

class Memory(BaseModel):
    memory_id: str
    content: str
    memory_type: str
    confidence: float
    source_messages: List[str]
    created_at: datetime
    updated_at: datetime
    chat_id: str

class MemoryQuery(BaseModel):
    query: str
    chat_id: Optional[str] = None
    memory_types: Optional[List[str]] = None

# Memory Types Configuration
MEMORY_TYPES = {
    "food_preference": {
        "keywords": ["prefer", "like", "love", "hate", "dislike", "favorite", "allergic", "vegetarian", "vegan"],
        "patterns": [r"prefer\s+(\w+)", r"don't\s+like\s+(\w+)", r"allergic\s+to\s+(\w+)"]
    },
    "travel_preference": {
        "keywords": ["flight", "seat", "premium", "economy", "first class", "window", "aisle"],
        "patterns": [r"prefer\s+(\w+)\s+seat", r"always\s+book\s+(\w+)"]
    },
    "personal_info": {
        "keywords": ["address", "phone", "email", "birthday", "age"],
        "patterns": [r"my\s+address\s+is\s+(.+)", r"phone\s+number\s+is\s+(\d+)"]
    },
    "delivery_instruction": {
        "keywords": ["deliver", "delivery", "guard", "doorman", "lobby", "building"],
        "patterns": [r"leave\s+with\s+(.+)", r"delivery\s+(.+)"]
    },
    "hobby_interest": {
        "keywords": ["play", "hobby", "interest", "music", "instrument", "sport"],
        "patterns": [r"play\s+the\s+(\w+)", r"hobby\s+is\s+(\w+)"]
    },
    "routine_timing": {
        "keywords": ["usually", "always", "typically", "every", "daily", "routine"],
        "patterns": [r"usually\s+(.+)\s+at\s+(\d+)", r"always\s+(.+)\s+around\s+(\d+)"]
    }
}

class EnhancedMemoryExtractor:
    def __init__(self):
        self.batch_size = 16
        self.overlap_size = 4
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
    def extract_memories_from_messages(self, messages: List[ChatMessage]) -> List[Dict]:
        """Extract memories using both rule-based and LLM approaches"""
        # First, get rule-based memories
        rule_based_memories = self._extract_rule_based_memories(messages)
        
        # Then, enhance with LLM extraction
        llm_memories = self._extract_llm_memories(messages)
        
        # Combine and deduplicate
        all_memories = rule_based_memories + llm_memories
        return self._deduplicate_memories(all_memories)
    
    def _extract_rule_based_memories(self, messages: List[ChatMessage]) -> List[Dict]:
        """Original rule-based extraction (fallback)"""
        memories = []
        
        for i in range(0, len(messages), self.batch_size - self.overlap_size):
            batch = messages[i:i + self.batch_size]
            batch_memories = self._process_message_batch_rules(batch)
            memories.extend(batch_memories)
        
        return memories
    
    def _extract_llm_memories(self, messages: List[ChatMessage]) -> List[Dict]:
        """LLM-based memory extraction using Gemini"""
        memories = []
        
        for i in range(0, len(messages), self.batch_size - self.overlap_size):
            batch = messages[i:i + self.batch_size]
            try:
                batch_memories = self._process_message_batch_llm(batch)
                memories.extend(batch_memories)
            except Exception as e:
                print(f"LLM extraction failed for batch: {e}")
                # Fallback to rule-based for this batch
                batch_memories = self._process_message_batch_rules(batch)
                memories.extend(batch_memories)
        
        return memories
    
    def _process_message_batch_llm(self, messages: List[ChatMessage]) -> List[Dict]:
        """Process message batch using Gemini LLM"""
        if not messages:
            return []
        
        # Prepare conversation context
        conversation_text = "\n".join([
            f"{msg.sender}: {msg.content}" for msg in messages
        ])
        
        prompt = self._create_extraction_prompt(conversation_text)
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_llm_response(response.text, messages[0].chat_id, messages)
        except Exception as e:
            print(f"Gemini API error: {e}")
            return []
    
    def _create_extraction_prompt(self, conversation_text: str) -> str:
        """Create a structured prompt for memory extraction"""
        memory_types_desc = {
            "food_preference": "Food likes, dislikes, allergies, dietary restrictions",
            "travel_preference": "Travel habits, seat preferences, booking patterns",
            "personal_info": "Personal details like address, phone, email, birthday",
            "delivery_instruction": "Delivery preferences, building access, special instructions",
            "hobby_interest": "Hobbies, interests, skills, activities they enjoy",
            "routine_timing": "Daily routines, regular schedules, timing preferences"
        }
        
        prompt = f"""
Analyze the following conversation and extract important memories about the person. Focus on persistent preferences, habits, and facts that would be useful to remember in future interactions.

CONVERSATION:
{conversation_text}

MEMORY CATEGORIES:
{json.dumps(memory_types_desc, indent=2)}

Extract memories and return them in this JSON format:
{{
  "memories": [
    {{
      "content": "The specific memory content",
      "memory_type": "category from above",
      "confidence": 0.8,
      "reasoning": "Why this is a valuable memory to store"
    }}
  ]
}}

Rules:
1. Only extract facts that are likely to remain true over time
2. Ignore temporary states or one-time events
3. Focus on preferences, habits, personal information, and consistent patterns
4. Assign confidence scores from 0.1 to 1.0 based on how certain you are
5. Don't extract obviously false or joking statements
6. Prefer specific details over general statements

Return valid JSON only:
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str, chat_id: str, messages: List[ChatMessage]) -> List[Dict]:
        """Parse LLM response into memory objects"""
        try:
            # Clean the response (remove markdown formatting if present)
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3]
            
            data = json.loads(cleaned_response)
            memories = []
            
            for memory_data in data.get("memories", []):
                memory = {
                    "memory_id": str(uuid.uuid4()),
                    "content": memory_data["content"],
                    "memory_type": memory_data["memory_type"],
                    "confidence": float(memory_data["confidence"]),
                    "source_messages": [msg.message_id for msg in messages],
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "chat_id": chat_id,
                    "extraction_method": "llm",
                    "reasoning": memory_data.get("reasoning", "")
                }
                memories.append(memory)
            
            return memories
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            print(f"Response was: {response_text}")
            return []
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []
    
    def _process_message_batch_rules(self, messages: List[ChatMessage]) -> List[Dict]:
        """Original rule-based processing"""
        memories = []
        
        for message in messages:
            content = message.content.lower()
            
            for memory_type, config in MEMORY_TYPES.items():
                if any(keyword in content for keyword in config["keywords"]):
                    memory = self._extract_memory_by_type_rules(message, memory_type, config)
                    if memory:
                        memory["extraction_method"] = "rules"
                        memories.append(memory)
        
        return memories
    
    def _extract_memory_by_type_rules(self, message: ChatMessage, memory_type: str, config: Dict) -> Dict:
        """Original rule-based extraction method"""
        content = message.content
        
        # Pattern matching for structured extraction
        for pattern in config.get("patterns", []):
            matches = re.findall(pattern, content.lower())
            if matches:
                return {
                    "memory_id": str(uuid.uuid4()),
                    "content": self._clean_memory_content(content, matches),
                    "memory_type": memory_type,
                    "confidence": self._calculate_confidence(content, memory_type),
                    "source_messages": [message.message_id],
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "chat_id": message.chat_id
                }
        
        # Fallback to keyword-based extraction
        if any(keyword in content.lower() for keyword in config["keywords"]):
            return {
                "memory_id": str(uuid.uuid4()),
                "content": self._extract_relevant_sentence(content, config["keywords"]),
                "memory_type": memory_type,
                "confidence": self._calculate_confidence(content, memory_type),
                "source_messages": [message.message_id],
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "chat_id": message.chat_id
            }
        
        return None

    def _clean_memory_content(self, content: str, matches: List) -> str:
        """Clean and format extracted memory content"""
        # Remove timestamps, sender info, etc.
        cleaned = re.sub(r'\[\d{2}:\d{2}\]', '', content)
        cleaned = re.sub(r'^[^:]+:', '', cleaned).strip()
        return cleaned[:200]  # Limit length
    
    def _extract_relevant_sentence(self, content: str, keywords: List[str]) -> str:
        """Extract the most relevant sentence containing keywords"""
        sentences = content.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                return sentence.strip()[:200]
        return content[:200]
    
    def _calculate_confidence(self, content: str, memory_type: str) -> float:
        """Calculate confidence score for extracted memory"""
        base_confidence = 0.5
        
        # Explicit indicators increase confidence
        explicit_words = ["prefer", "always", "never", "hate", "love"]
        if any(word in content.lower() for word in explicit_words):
            base_confidence += 0.3
        
        # Question marks decrease confidence
        if "?" in content:
            base_confidence -= 0.2
        
        # Temporal words decrease confidence for preferences
        temporal_words = ["today", "now", "currently", "right now"]
        if any(word in content.lower() for word in temporal_words):
            base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
    
    def _deduplicate_memories(self, memories: List[Dict]) -> List[Dict]:
        """Enhanced deduplication considering semantic similarity"""
        if not memories:
            return []
        
        # Group by memory type for better comparison
        grouped_memories = {}
        for memory in memories:
            memory_type = memory["memory_type"]
            if memory_type not in grouped_memories:
                grouped_memories[memory_type] = []
            grouped_memories[memory_type].append(memory)
        
        deduplicated = []
        for memory_type, type_memories in grouped_memories.items():
            # Use LLM for semantic deduplication within each type
            deduplicated.extend(self._semantic_deduplicate(type_memories))
        
        return deduplicated
    
    def _semantic_deduplicate(self, memories: List[Dict]) -> List[Dict]:
        """Use LLM to identify and merge semantically similar memories"""
        if len(memories) <= 1:
            return memories
        
        try:
            memory_contents = [mem["content"] for mem in memories]
            prompt = f"""
Compare these memories and identify which ones are duplicates or very similar:

MEMORIES:
{json.dumps(memory_contents, indent=2)}

Return a JSON object with groups of similar memories:
{{
  "groups": [
    {{
      "representative_index": 0,
      "similar_indices": [0, 2, 5],
      "merged_content": "Best version combining the information"
    }}
  ]
}}

Rules:
- Group memories that refer to the same fact or preference
- Choose the most detailed/accurate version as representative
- If memories conflict, choose the one with higher confidence
"""
            
            response = self.model.generate_content(prompt)
            
            # Better response cleaning
            cleaned_response = response.text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()
            
            # Additional cleaning for common issues
            if not cleaned_response:
                print("Empty response from LLM")
                return self._simple_deduplicate(memories)
            
            try:
                result = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Cleaned response: {cleaned_response[:200]}...")
                return self._simple_deduplicate(memories)
            
            final_memories = []
            processed_indices = set()
            
            for group in result.get("groups", []):
                rep_idx = group.get("representative_index", 0)
                similar_indices = group.get("similar_indices", [rep_idx])
                
                if rep_idx not in processed_indices and rep_idx < len(memories):
                    # Use the representative memory as base
                    final_memory = memories[rep_idx].copy()
                    
                    # Merge source messages from all similar memories
                    all_sources = []
                    max_confidence = 0
                    
                    for idx in similar_indices:
                        if idx < len(memories):
                            # Handle source_messages properly
                            sources = memories[idx].get("source_messages", [])
                            if isinstance(sources, str):
                                try:
                                    sources = json.loads(sources)
                                except:
                                    sources = []
                            all_sources.extend(sources)
                            max_confidence = max(max_confidence, memories[idx]["confidence"])
                            processed_indices.add(idx)
                    
                    final_memory["source_messages"] = list(set(all_sources))
                    final_memory["confidence"] = max_confidence
                    final_memory["content"] = group.get("merged_content", final_memory["content"])
                    
                    final_memories.append(final_memory)
            
            # Add memories that weren't part of any group
            for i, memory in enumerate(memories):
                if i not in processed_indices:
                    final_memories.append(memory)
            
            return final_memories
            
        except Exception as e:
            print(f"Semantic deduplication failed: {e}")
            # Fallback to simple hash-based deduplication
            return self._simple_deduplicate(memories)
    
    def _simple_deduplicate(self, memories: List[Dict]) -> List[Dict]:
        """Simple hash-based deduplication (fallback)"""
        seen_hashes = set()
        unique_memories = []
        
        for memory in memories:
            content_hash = hashlib.md5(memory["content"].encode()).hexdigest()
            if content_hash not in seen_hashes:
                memory["content_hash"] = content_hash
                seen_hashes.add(content_hash)
                unique_memories.append(memory)
        
        return unique_memories

class MemoryUpdateManager:
    def __init__(self, extractor):
        self.extractor = extractor
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    def _get_existing_memories(self, chat_id: str) -> List[Dict]:
        """Get all existing memories for a chat"""
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM memories WHERE chat_id = %s", (chat_id,))
                rows = cursor.fetchall()
                memories = []
                for row in rows:
                    memory = dict(row)
                    # Parse JSON strings back to lists/objects
                    try:
                        memory["source_messages"] = json.loads(memory["source_messages"]) if memory["source_messages"] else []
                    except (json.JSONDecodeError, TypeError):
                        memory["source_messages"] = []
                    memories.append(memory)
                return memories
        
    def update_or_create_memories(self, new_memories: List[Dict], chat_id: str) -> Dict:
        """Smart memory update system - updates existing memories or creates new ones"""
        existing_memories = self._get_existing_memories(chat_id)
        
        update_results = {
            "updated": 0,
            "created": 0,
            "conflicts_resolved": 0,
            "operations": []
        }
        
        for new_memory in new_memories:
            result = self._process_single_memory(new_memory, existing_memories, chat_id)
            update_results["operations"].append(result)
            
            if result["action"] == "update":
                update_results["updated"] += 1
            elif result["action"] == "create":
                update_results["created"] += 1
            elif result["action"] == "merge":
                update_results["conflicts_resolved"] += 1
            
            # Update existing_memories list to reflect changes for subsequent iterations
            if result["action"] in ["update", "merge"]:
                # Remove old memory and add updated one
                memory_id = result["memory_id"]
                existing_memories = [m for m in existing_memories if m["memory_id"] != memory_id]
                # Add the updated memory back
                updated_mem = self._get_memory_by_id(memory_id)
                if updated_mem:
                    existing_memories.append(updated_mem)
        
        return update_results
    
    def _get_memory_by_id(self, memory_id: str) -> Dict:
        """Get a specific memory by ID"""
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM memories WHERE memory_id = %s", (memory_id,))
                row = cursor.fetchone()
                if row:
                    memory = dict(row)
                    try:
                        memory["source_messages"] = json.loads(memory["source_messages"]) if memory["source_messages"] else []
                    except (json.JSONDecodeError, TypeError):
                        memory["source_messages"] = []
                    return memory
        return None
    
    def _process_single_memory(self, new_memory: Dict, existing_memories: List[Dict], chat_id: str) -> Dict:
        """Process a single memory - decide whether to update, create, or merge"""
        memory_type = new_memory["memory_type"]
        
        # Find similar memories using enhanced similarity detection
        similar_memories = self._find_similar_memories_enhanced(new_memory, existing_memories)
        
        if not similar_memories:
            # No similar memory found - create new
            memory_id = self._create_memory(new_memory)
            return {
                "action": "create",
                "memory_id": memory_id,
                "content": new_memory["content"],
                "reasoning": "No similar memory found"
            }
        
        # Found similar memories - decide what to do
        best_match = similar_memories[0]
        similarity_score = best_match["similarity_score"]
        
        if similarity_score > 0.7:  # Lowered threshold for address updates
            # High similarity - check if it's an update scenario
            conflict_resolution = self._resolve_conflict(best_match["memory"], new_memory)
            
            if conflict_resolution["action"] == "replace":
                # Check if result key exists
                if "result" in conflict_resolution:
                    updated_memory = conflict_resolution["result"]
                    memory_id = self._update_memory(updated_memory)
                    
                    # Delete other similar memories to avoid duplicates
                    self._delete_similar_memories(similar_memories[1:])
                    
                    return {
                        "action": "update",
                        "memory_id": memory_id,
                        "old_content": best_match["memory"]["content"],
                        "new_content": updated_memory["content"],
                        "reasoning": conflict_resolution["reasoning"]
                    }
                else:
                    # Fallback: update the existing memory directly
                    existing_memory = best_match["memory"]
                    existing_sources = existing_memory.get("source_messages", [])
                    if isinstance(existing_sources, str):
                        try:
                            existing_sources = json.loads(existing_sources)
                        except:
                            existing_sources = []
                    
                    new_sources = new_memory.get("source_messages", [])
                    
                    updated_memory = existing_memory.copy()
                    updated_memory.update({
                        "content": new_memory["content"],
                        "confidence": max(existing_memory["confidence"], new_memory["confidence"]),
                        "updated_at": datetime.now(),
                        "source_messages": list(set(existing_sources + new_sources)),
                        "reasoning": "Direct update - similar memory found"
                    })
                    
                    memory_id = self._update_memory(updated_memory)
                    self._delete_similar_memories(similar_memories[1:])
                    
                    return {
                        "action": "update",
                        "memory_id": memory_id,
                        "old_content": best_match["memory"]["content"],
                        "new_content": updated_memory["content"],
                        "reasoning": "Direct update - similar memory found"
                    }
            
            elif conflict_resolution["action"] == "keep_existing":
                # Don't create a new memory, just return existing
                return {
                    "action": "keep_existing",
                    "memory_id": best_match["memory"]["memory_id"],
                    "content": best_match["memory"]["content"],
                    "reasoning": conflict_resolution["reasoning"]
                }
            
            else:  # create_new or any other action
                # Create new memory if they're different enough
                memory_id = self._create_memory(new_memory)
                return {
                    "action": "create",
                    "memory_id": memory_id,
                    "content": new_memory["content"],
                    "reasoning": conflict_resolution.get("reasoning", "Different enough to warrant separate memory")
                }
        
        else:
            # Low similarity - create new memory
            memory_id = self._create_memory(new_memory)
            return {
                "action": "create",
                "memory_id": memory_id,
                "content": new_memory["content"],
                "reasoning": f"Low similarity ({similarity_score}) - creating new memory"
            }
    
    def _find_similar_memories_enhanced(self, new_memory: Dict, existing_memories: List[Dict]) -> List[Dict]:
        """Enhanced similarity detection with address-specific logic"""
        if not existing_memories:
            return []
        
        # Filter by memory type first
        same_type_memories = [
            mem for mem in existing_memories 
            if mem["memory_type"] == new_memory["memory_type"]
        ]
        
        if not same_type_memories:
            return []
        
        # Special handling for address updates
        if new_memory["memory_type"] == "personal_info":
            address_keywords = ["address", "street", "avenue", "road", "apt", "apartment", "building"]
            new_has_address = any(keyword in new_memory["content"].lower() for keyword in address_keywords)
            
            if new_has_address:
                # Look for existing address memories
                address_memories = []
                for mem in same_type_memories:
                    if any(keyword in mem["content"].lower() for keyword in address_keywords):
                        address_memories.append({
                            "memory": mem,
                            "similarity_score": 0.9,  # High similarity for address type
                            "reasoning": "Both contain address information"
                        })
                
                if address_memories:
                    return address_memories
        
        # Fallback to LLM-based similarity
        try:
            prompt = f"""
Compare the new memory with existing memories and rate their similarity.

NEW MEMORY:
{new_memory["content"]}

EXISTING MEMORIES:
{json.dumps([{"id": i, "content": mem["content"]} for i, mem in enumerate(same_type_memories)], indent=2)}

Rate similarity from 0.0 to 1.0 for each existing memory:
{{
  "similarities": [
    {{
      "memory_index": 0,
      "similarity_score": 0.9,
      "reasoning": "Both refer to the same type of information"
    }}
  ]
}}

Consider:
- 0.9-1.0: Same fact/preference type (e.g., both addresses, both phone numbers)
- 0.7-0.8: Related but might have different details
- 0.5-0.6: Same category but different specifics
- 0.0-0.4: Different facts/preferences

Return valid JSON:
"""
            
            response = self.model.generate_content(prompt)
            cleaned_response = response.text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3]
            
            similarities = json.loads(cleaned_response)
            
            # Create result list with memory objects and scores
            similar_memories = []
            for sim in similarities["similarities"]:
                if sim["similarity_score"] > 0.5:  # Only consider moderately similar or higher
                    memory_idx = sim["memory_index"]
                    if memory_idx < len(same_type_memories):
                        similar_memories.append({
                            "memory": same_type_memories[memory_idx],
                            "similarity_score": sim["similarity_score"],
                            "reasoning": sim["reasoning"]
                        })
            
            # Sort by similarity score (highest first)
            similar_memories.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_memories
            
        except Exception as e:
            print(f"Error finding similar memories: {e}")
            return []
    
    def _resolve_conflict(self, existing_memory: Dict, new_memory: Dict) -> Dict:
        """Use LLM to resolve conflicts between memories with enhanced address detection"""
        try:
            # Special handling for address updates
            address_keywords = ["address", "street", "avenue", "road", "apt", "apartment", "building"]
            existing_has_address = any(keyword in existing_memory["content"].lower() for keyword in address_keywords)
            new_has_address = any(keyword in new_memory["content"].lower() for keyword in address_keywords)
            
            if existing_has_address and new_has_address:
                # This is likely an address update - favor the new address
                # Handle source_messages properly
                existing_sources = existing_memory.get("source_messages", [])
                if isinstance(existing_sources, str):
                    try:
                        existing_sources = json.loads(existing_sources)
                    except:
                        existing_sources = []
                
                new_sources = new_memory.get("source_messages", [])
                if isinstance(new_sources, str):
                    try:
                        new_sources = json.loads(new_sources)
                    except:
                        new_sources = []
                
                updated_memory = existing_memory.copy()
                updated_memory.update({
                    "content": new_memory["content"],
                    "confidence": max(existing_memory["confidence"], new_memory["confidence"]),
                    "updated_at": datetime.now(),
                    "source_messages": list(set(existing_sources + new_sources)),
                    "reasoning": "Address update detected - replacing with new address"
                })
                
                return {
                    "action": "replace",
                    "result": updated_memory,
                    "reasoning": "Address update detected - replacing with new address"
                }
            
            prompt = f"""
    Two memories conflict or overlap. Decide how to resolve this:

    EXISTING MEMORY:
    Content: {existing_memory["content"]}
    Confidence: {existing_memory["confidence"]}
    Created: {existing_memory["created_at"]}
    Method: {existing_memory.get("extraction_method", "unknown")}

    NEW MEMORY:
    Content: {new_memory["content"]}
    Confidence: {new_memory["confidence"]}
    Method: {new_memory.get("extraction_method", "unknown")}

    Analyze and decide:
    {{
    "action": "replace|merge|keep_both",
    "result_content": "The final memory content",
    "confidence": 0.8,
    "reasoning": "Why this decision was made"
    }}

    Guidelines:
    - "replace": New memory is more accurate/recent (address change, preference update)
    - "merge": Combine both memories into more complete information
    - "keep_both": Memories are different enough to warrant separate entries

    For address information, always prefer "replace" with the newer address.

    Return valid JSON:
    """
            
            response = self.model.generate_content(prompt)
            cleaned_response = response.text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3]
            
            result = json.loads(cleaned_response)
            
            # Handle source_messages properly for both memories
            existing_sources = existing_memory.get("source_messages", [])
            if isinstance(existing_sources, str):
                try:
                    existing_sources = json.loads(existing_sources)
                except:
                    existing_sources = []
            
            new_sources = new_memory.get("source_messages", [])
            if isinstance(new_sources, str):
                try:
                    new_sources = json.loads(new_sources)
                except:
                    new_sources = []
            
            if result.get("action") == "replace":
                # Create updated memory with new content
                updated_memory = existing_memory.copy()
                updated_memory.update({
                    "content": result.get("result_content", new_memory["content"]),
                    "confidence": result.get("confidence", max(existing_memory["confidence"], new_memory["confidence"])),
                    "updated_at": datetime.now(),
                    "source_messages": list(set(existing_sources + new_sources)),
                    "reasoning": result.get("reasoning", "Memory replaced")
                })
                return {
                    "action": "replace",
                    "result": updated_memory,
                    "reasoning": result.get("reasoning", "Memory replaced")
                }
            
            elif result.get("action") == "merge":
                # Merge memories
                updated_memory = existing_memory.copy()
                updated_memory.update({
                    "content": result.get("result_content", f"{existing_memory['content']} | {new_memory['content']}"),
                    "confidence": max(existing_memory["confidence"], new_memory["confidence"]),
                    "updated_at": datetime.now(),
                    "source_messages": list(set(existing_sources + new_sources)),
                    "reasoning": result.get("reasoning", "Memories merged")
                })
                return {
                    "action": "replace",
                    "result": updated_memory,
                    "reasoning": result.get("reasoning", "Memories merged")
                }
            
            else:  # keep_both or any other action
                return {
                    "action": "create_new",
                    "reasoning": result.get("reasoning", "Memories are different enough to keep separate")
                }
                
        except Exception as e:
            print(f"Error resolving conflict: {e}")
            # Default fallback - favor the new memory for address updates
            address_keywords = ["address", "street", "avenue", "road", "apt", "apartment", "building"]
            existing_has_address = any(keyword in existing_memory["content"].lower() for keyword in address_keywords)
            new_has_address = any(keyword in new_memory["content"].lower() for keyword in address_keywords)
            
            if existing_has_address and new_has_address:
                # Handle source_messages safely
                existing_sources = existing_memory.get("source_messages", [])
                if isinstance(existing_sources, str):
                    try:
                        existing_sources = json.loads(existing_sources)
                    except:
                        existing_sources = []
                
                new_sources = new_memory.get("source_messages", [])
                if isinstance(new_sources, str):
                    try:
                        new_sources = json.loads(new_sources)
                    except:
                        new_sources = []
                
                updated_memory = existing_memory.copy()
                updated_memory.update({
                    "content": new_memory["content"],
                    "confidence": max(existing_memory["confidence"], new_memory["confidence"]),
                    "updated_at": datetime.now(),
                    "source_messages": list(set(existing_sources + new_sources)),
                    "reasoning": "Fallback: Address update detected"
                })
                
                return {
                    "action": "replace",
                    "result": updated_memory,
                    "reasoning": "Fallback: Address update detected"
                }
            
            # Default to keeping existing memory
            return {
                "action": "keep_existing",
                "reasoning": "Error in conflict resolution - keeping existing memory"
            }
        
    def _delete_similar_memories(self, similar_memories: List[Dict]):
        """Delete similar memories to avoid duplicates"""
        with get_db() as conn:
            with conn.cursor() as cursor:
                for sim_mem in similar_memories:
                    memory_id = sim_mem["memory"]["memory_id"]
                    cursor.execute("DELETE FROM memories WHERE memory_id = %s", (memory_id,))
            conn.commit()

    def _merge_memories(self, existing_memory: Dict, new_memory: Dict) -> Dict:
        """Merge two similar memories"""
        # Take the higher confidence
        confidence = max(existing_memory["confidence"], new_memory["confidence"])
        
        # Handle source_messages properly - ensure they're lists
        existing_sources = existing_memory.get("source_messages", [])
        if isinstance(existing_sources, str):
            try:
                existing_sources = json.loads(existing_sources)
            except:
                existing_sources = []
        
        new_sources = new_memory.get("source_messages", [])
        if isinstance(new_sources, str):
            try:
                new_sources = json.loads(new_sources)
            except:
                new_sources = []
        
        # Combine source messages
        all_sources = list(set(existing_sources + new_sources))
        
        # Use the more recent or more detailed content
        content = new_memory["content"] if len(new_memory["content"]) > len(existing_memory["content"]) else existing_memory["content"]
        
        # Update the existing memory
        updated_memory = existing_memory.copy()
        updated_memory.update({
            "content": content,
            "confidence": confidence,
            "source_messages": all_sources,
            "updated_at": datetime.now(),
            "extraction_method": new_memory.get("extraction_method", existing_memory.get("extraction_method", "rules")),
            "reasoning": f"Merged with similar memory. Original: {existing_memory['content'][:50]}..."
        })
        
        return updated_memory
        
    def _create_memory(self, memory: Dict) -> str:
        """Create a new memory"""
        # Generate content hash if not present
        if "content_hash" not in memory:
            memory["content_hash"] = hashlib.md5(memory["content"].encode()).hexdigest()
        
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO memories 
                    (memory_id, content, memory_type, confidence, source_messages, 
                     created_at, updated_at, chat_id, content_hash, extraction_method, reasoning)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    memory["memory_id"],
                    memory["content"],
                    memory["memory_type"],
                    memory["confidence"],
                    json.dumps(memory["source_messages"]),
                    memory["created_at"],
                    memory["updated_at"],
                    memory["chat_id"],
                    memory["content_hash"],
                    memory.get("extraction_method", "rules"),
                    memory.get("reasoning", "")
                ))
            conn.commit()
        
        return memory["memory_id"]
    
    def _update_memory(self, memory: Dict) -> str:
        """Update existing memory"""
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE memories 
                    SET content = %s, confidence = %s, source_messages = %s, updated_at = %s, 
                        extraction_method = %s, reasoning = %s, content_hash = %s
                    WHERE memory_id = %s
                """, (
                    memory["content"],
                    memory["confidence"],
                    json.dumps(memory["source_messages"]),
                    memory["updated_at"],
                    memory.get("extraction_method", "rules"),
                    memory.get("reasoning", ""),
                    memory.get("content_hash", hashlib.md5(memory["content"].encode()).hexdigest()),
                    memory["memory_id"]
                ))
            conn.commit()
        
        return memory["memory_id"]
    
class MemoryManager:
    def __init__(self):
        self.extractor = EnhancedMemoryExtractor()
        self.update_manager = MemoryUpdateManager(self.extractor)
    
    def process_new_memories(self, messages: List[ChatMessage]) -> Dict:
        """Extract and intelligently update memories"""
        # Extract new memories
        extracted_memories = self.extractor.extract_memories_from_messages(messages)
        
        if not extracted_memories:
            return {"status": "no_memories_extracted"}
        
        # Use update manager to handle creation/updates
        chat_id = messages[0].chat_id
        update_results = self.update_manager.update_or_create_memories(extracted_memories, chat_id)
        
        return {
            "status": "success",
            "extracted": len(extracted_memories),
            **update_results
        }
    
    def save_memory(self, memory: Dict) -> str:
        """Save or update memory in database"""
        with get_db() as conn:
            with conn.cursor() as cursor:
                # Check for existing similar memory
                cursor.execute(
                    "SELECT * FROM memories WHERE content_hash = %s AND chat_id = %s",
                    (memory.get("content_hash", ""), memory["chat_id"])
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing memory
                    cursor.execute("""
                        UPDATE memories 
                        SET content = %s, confidence = %s, source_messages = %s, updated_at = %s, 
                            extraction_method = %s, reasoning = %s
                        WHERE memory_id = %s
                    """, (
                        memory["content"],
                        memory["confidence"],
                        json.dumps(memory["source_messages"]),
                        memory["updated_at"],
                        memory.get("extraction_method", "rules"),
                        memory.get("reasoning", ""),
                        existing["memory_id"]
                    ))
                    conn.commit()
                    return existing["memory_id"]
                else:
                    # Insert new memory
                    cursor.execute("""
                        INSERT INTO memories 
                        (memory_id, content, memory_type, confidence, source_messages, 
                         created_at, updated_at, chat_id, content_hash, extraction_method, reasoning)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        memory["memory_id"],
                        memory["content"],
                        memory["memory_type"],
                        memory["confidence"],
                        json.dumps(memory["source_messages"]),
                        memory["created_at"],
                        memory["updated_at"],
                        memory["chat_id"],
                        memory.get("content_hash", ""),
                        memory.get("extraction_method", "rules"),
                        memory.get("reasoning", "")
                    ))
                    conn.commit()
                    return memory["memory_id"]
    
    def query_memories(self, query: str, chat_id: Optional[str] = None, 
                      memory_types: Optional[List[str]] = None) -> List[Dict]:
        """Query memories based on content and filters"""
        with get_db() as conn:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM memories WHERE 1=1"
                params = []
                
                if chat_id:
                    sql += " AND chat_id = %s"
                    params.append(chat_id)
                
                if memory_types:
                    placeholders = ",".join(["%s" for _ in memory_types])
                    sql += f" AND memory_type IN ({placeholders})"
                    params.extend(memory_types)
                
                if query:
                    sql += " AND content ILIKE %s"
                    params.append(f"%{query}%")
                
                sql += " ORDER BY confidence DESC, updated_at DESC"
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                memories = []
                for row in rows:
                    memory = dict(row)
                    # Parse JSON strings back to lists
                    try:
                        memory["source_messages"] = json.loads(memory["source_messages"]) if memory["source_messages"] else []
                    except (json.JSONDecodeError, TypeError):
                        memory["source_messages"] = []
                    memories.append(memory)
                return memories

# Initialize components
init_database()
memory_manager = MemoryManager()

# API Endpoints
@app.post("/api/chat/upload")
async def upload_chat_messages(messages: List[ChatMessage]):
    """Upload chat messages to the system"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            for message in messages:
                cursor.execute("""
                    INSERT INTO chat_messages 
                    (message_id, timestamp, sender, content, chat_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (message_id) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    sender = EXCLUDED.sender,
                    content = EXCLUDED.content,
                    chat_id = EXCLUDED.chat_id
                """, (
                    message.message_id,
                    message.timestamp,
                    message.sender,
                    message.content,
                    message.chat_id
                ))
        conn.commit()
    
    return {"status": "success", "uploaded": len(messages)}

# Add a cleanup endpoint to remove duplicate memories
@app.post("/api/memories/cleanup/{chat_id}")
async def cleanup_duplicate_memories(chat_id: str):
    """Clean up duplicate memories for a chat"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            # Find duplicate address memories
            cursor.execute("""
                SELECT * FROM memories 
                WHERE chat_id = %s AND memory_type = 'personal_info' 
                AND (content ILIKE '%address%' OR content ILIKE '%street%' OR content ILIKE '%avenue%')
                ORDER BY updated_at DESC
            """, (chat_id,))
            address_memories = cursor.fetchall()
            
            if len(address_memories) > 1:
                # Keep the most recent one, delete the rest
                keep_memory = address_memories[0]
                delete_ids = [mem["memory_id"] for mem in address_memories[1:]]
                
                for mem_id in delete_ids:
                    cursor.execute("DELETE FROM memories WHERE memory_id = %s", (mem_id,))
                
                conn.commit()
                
                return {
                    "status": "success",
                    "kept_memory": dict(keep_memory),
                    "deleted_count": len(delete_ids)
                }
            
            return {"status": "no_duplicates_found"}

@app.post("/api/memories/extract/{chat_id}")
async def extract_memories(chat_id: str):
    """Extract and intelligently update memories from chat messages"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM chat_messages WHERE chat_id = %s ORDER BY timestamp",
                (chat_id,)
            )
            rows = cursor.fetchall()
            
            if not rows:
                raise HTTPException(status_code=404, detail="Chat not found")
            
            messages = [
                ChatMessage(
                    message_id=row["message_id"],
                    timestamp=row["timestamp"],
                    sender=row["sender"],
                    content=row["content"],
                    chat_id=row["chat_id"]
                )
                for row in rows
            ]
    
    # Use smart processing
    results = memory_manager.process_new_memories(messages)
    
    return results

@app.get("/api/memories/{chat_id}")
async def get_memories(chat_id: str):
    """Get all memories for a chat"""
    memories = memory_manager.query_memories("", chat_id=chat_id)
    return {"memories": memories}

@app.post("/api/memories/query")
async def query_memories(query_request: MemoryQuery):
    """Query memories with filters"""
    memories = memory_manager.query_memories(
        query_request.query,
        chat_id=query_request.chat_id,
        memory_types=query_request.memory_types
    )
    return {"memories": memories}

@app.get("/api/chats")
async def get_chats():
    """Get list of all chats"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT chat_id, COUNT(*) as message_count, 
                       MIN(timestamp) as first_message, 
                       MAX(timestamp) as last_message
                FROM chat_messages 
                GROUP BY chat_id
            """)
            rows = cursor.fetchall()
            
            return {"chats": [dict(row) for row in rows]}

@app.get("/api/memory-types")
async def get_memory_types():
    """Get available memory types"""
    return {"memory_types": list(MEMORY_TYPES.keys())}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Memory Extraction API", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected" if result else "disconnected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)