# samantha.py
import os
import json
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import spacy
import numpy as np
from pydantic import BaseModel

# MCP imports
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("samantha")

# Initialize spaCy for NLP
try:
    nlp = spacy.load("en_core_web_md")
    logger.info("Loaded spaCy model en_core_web_md")
except OSError:
    logger.info("Downloading spaCy model en_core_web_md")
    try:
        # If model not found, download it
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"], check=True)
        nlp = spacy.load("en_core_web_md")
        logger.info("Successfully downloaded and loaded spaCy model")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        # Provide a minimal fallback model
        from spacy.lang.en import English
        nlp = English()
        logger.warning("Using minimal English model as fallback")

# -------------------------------------------------
# Data Models
# -------------------------------------------------

@dataclass
class LearningStyle:
    """VARK learning style model with descriptive representation."""
    visual: float = 0.0
    auditory: float = 0.0
    reading_writing: float = 0.0
    kinesthetic: float = 0.0
    
    def update(self, new_style: 'LearningStyle', weight: float = 0.3):
        """Update learning style with weighted approach."""
        self.visual = (1 - weight) * self.visual + weight * new_style.visual
        self.auditory = (1 - weight) * self.auditory + weight * new_style.auditory
        self.reading_writing = (1 - weight) * self.reading_writing + weight * new_style.reading_writing
        self.kinesthetic = (1 - weight) * self.kinesthetic + weight * new_style.kinesthetic
    
    def get_normalized(self) -> 'LearningStyle':
        """Return normalized learning style."""
        total = self.visual + self.auditory + self.reading_writing + self.kinesthetic
        if total == 0:
            return LearningStyle(0.25, 0.25, 0.25, 0.25)
        
        return LearningStyle(
            self.visual / total,
            self.auditory / total,
            self.reading_writing / total,
            self.kinesthetic / total
        )
    
    def get_primary_style(self) -> str:
        """Get the primary learning style."""
        styles = {
            "visual": self.visual,
            "auditory": self.auditory,
            "reading/writing": self.reading_writing,
            "kinesthetic": self.kinesthetic
        }
        return max(styles, key=styles.get)
    
    def get_secondary_style(self) -> str:
        """Get the secondary learning style."""
        normalized = self.get_normalized()
        styles = {
            "visual": normalized.visual,
            "auditory": normalized.auditory,
            "reading/writing": normalized.reading_writing,
            "kinesthetic": normalized.kinesthetic
        }
        primary = self.get_primary_style()
        del styles[primary]
        return max(styles, key=styles.get)
    
    def get_descriptive_text(self) -> str:
        """Convert learning style to descriptive text."""
        norm = self.get_normalized()
        primary = self.get_primary_style()
        secondary = self.get_secondary_style()
        
        # Determine strength of primary style
        primary_score = getattr(norm, primary.replace("/", "_"))
        if primary_score > 0.5:
            strength = "strong"
        elif primary_score > 0.35:
            strength = "moderate"
        else:
            strength = "balanced"
        
        description = f"A {strength} {primary} learner with {secondary} tendencies."
        
        # Add recommendations based on learning style
        recommendations = "\n\nRecommended learning approaches:\n"
        
        if norm.visual > 0.3:
            recommendations += "- Use diagrams, charts, and visual representations\n"
            recommendations += "- Color-code important information\n"
            recommendations += "- Create mind maps for complex topics\n"
        
        if norm.auditory > 0.3:
            recommendations += "- Learn through discussions and verbal explanations\n"
            recommendations += "- Record and listen to lectures or explanations\n"
            recommendations += "- Read content aloud when studying\n"
        
        if norm.reading_writing > 0.3:
            recommendations += "- Take detailed notes and rewrite key points\n"
            recommendations += "- Use written explanations and definitions\n"
            recommendations += "- Organize information in lists and outlines\n"
        
        if norm.kinesthetic > 0.3:
            recommendations += "- Use hands-on activities and practical exercises\n"
            recommendations += "- Take breaks and move around while learning\n"
            recommendations += "- Create physical models or demonstrations\n"
        
        return description + recommendations

@dataclass
class Topic:
    """Represents a topic of interest with mastery level."""
    name: str
    mastery: float = 0.0  # 0.0 to 1.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    related_topics: List[str] = field(default_factory=list)
    
    def update_mastery(self, new_mastery: float, weight: float = 0.3):
        """Update mastery level with weighted approach."""
        self.mastery = (1 - weight) * self.mastery + weight * new_mastery
        self.last_updated = datetime.now().isoformat()

@dataclass
class SystemPrompt:
    """Stores system prompts found to be effective."""
    content: str
    effectiveness: float = 0.0  # 0.0 to 1.0
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    topics: List[str] = field(default_factory=list)
    description: str = ""
    
    def update_effectiveness(self, new_effectiveness: float, weight: float = 0.3):
        """Update effectiveness rating with weighted approach."""
        self.effectiveness = (1 - weight) * self.effectiveness + weight * new_effectiveness
        self.last_used = datetime.now().isoformat()

@dataclass
class ConversationStyle:
    """Represents user's conversation style preferences."""
    formality: float = 0.5  # 0.0 (casual) to 1.0 (formal)
    verbosity: float = 0.5  # 0.0 (concise) to 1.0 (detailed)
    technical_level: float = 0.5  # 0.0 (simple) to 1.0 (technical)
    
    def update(self, new_style: 'ConversationStyle', weight: float = 0.3):
        """Update conversation style with weighted approach."""
        self.formality = (1 - weight) * self.formality + weight * new_style.formality
        self.verbosity = (1 - weight) * self.verbosity + weight * new_style.verbosity
        self.technical_level = (1 - weight) * self.technical_level + weight * new_style.technical_level
    
    def get_descriptive_text(self) -> str:
        """Convert conversation style to descriptive text."""
        formality = "formal" if self.formality > 0.7 else "casual" if self.formality < 0.3 else "neutral"
        verbosity = "detailed" if self.verbosity > 0.7 else "concise" if self.verbosity < 0.3 else "balanced"
        technical = "technical" if self.technical_level > 0.7 else "simple" if self.technical_level < 0.3 else "moderate"
        
        return f"Prefers {formality}, {verbosity}, and {technical} communication."

@dataclass
class Preference:
    """Represents a user preference."""
    category: str  # e.g., "movies", "books", "hobbies"
    items: Dict[str, float] = field(default_factory=dict)  # item name -> preference score
    
    def update_item(self, item: str, score: float, weight: float = 0.3):
        """Update preference item with weighted approach."""
        if item in self.items:
            self.items[item] = (1 - weight) * self.items[item] + weight * score
        else:
            self.items[item] = score
    
    def get_top_items(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get top preferred items."""
        sorted_items = sorted(self.items.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:limit]

@dataclass
class UserMemory:
    """Main user memory model."""
    user_id: str
    learning_style: LearningStyle = field(default_factory=LearningStyle)
    topics: Dict[str, Topic] = field(default_factory=dict)
    system_prompts: List[SystemPrompt] = field(default_factory=list)
    conversation_style: ConversationStyle = field(default_factory=ConversationStyle)
    preferences: Dict[str, Preference] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def update_last_updated(self):
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserMemory':
        """Create instance from dictionary."""
        user_id = data.pop('user_id')
        
        # Convert learning_style dict to object
        learning_style_data = data.pop('learning_style', {})
        learning_style = LearningStyle(**learning_style_data)
        
        # Convert topics dict to objects
        topics_data = data.pop('topics', {})
        topics = {}
        for topic_name, topic_data in topics_data.items():
            topics[topic_name] = Topic(**topic_data)
        
        # Convert system_prompts list to objects
        system_prompts_data = data.pop('system_prompts', [])
        system_prompts = [SystemPrompt(**prompt) for prompt in system_prompts_data]
        
        # Convert conversation_style dict to object
        conversation_style_data = data.pop('conversation_style', {})
        conversation_style = ConversationStyle(**conversation_style_data)
        
        # Convert preferences dict to objects
        preferences_data = data.pop('preferences', {})
        preferences = {}
        for pref_category, pref_data in preferences_data.items():
            preferences[pref_category] = Preference(**pref_data)
        
        # Get timestamps
        created_at = data.pop('created_at', datetime.now().isoformat())
        last_updated = data.pop('last_updated', datetime.now().isoformat())
        
        return cls(
            user_id=user_id,
            learning_style=learning_style,
            topics=topics,
            system_prompts=system_prompts,
            conversation_style=conversation_style,
            preferences=preferences,
            created_at=created_at,
            last_updated=last_updated
        )

    def get_llm_friendly_representation(self) -> str:
        """Generate an LLM-friendly representation of user memory."""
        output = f"# User Memory Profile\n\n"
        
        # Add learning style
        output += "## Learning Style\n\n"
        output += self.learning_style.get_descriptive_text() + "\n\n"
        
        # Add conversation style
        output += "## Communication Preferences\n\n"
        output += self.conversation_style.get_descriptive_text() + "\n\n"
        
        # Add top topics
        output += "## Knowledge Areas\n\n"
        sorted_topics = sorted(self.topics.values(), key=lambda t: t.mastery, reverse=True)
        for topic in sorted_topics[:5]:
            mastery_level = "Expert" if topic.mastery > 0.8 else "Advanced" if topic.mastery > 0.6 else "Intermediate" if topic.mastery > 0.3 else "Beginner"
            output += f"- {topic.name}: {mastery_level} level\n"
        
        if len(sorted_topics) > 5:
            output += f"- Plus {len(sorted_topics) - 5} more topics\n"
        output += "\n"
        
        # Add preferences
        output += "## Personal Preferences\n\n"
        for category, preference in self.preferences.items():
            top_items = preference.get_top_items(3)
            if top_items:
                output += f"### {category.title()}\n"
                for item, score in top_items:
                    output += f"- {item}\n"
                output += "\n"
        
        return output

# -------------------------------------------------
# Memory Storage & Retrieval
# -------------------------------------------------

class MemoryStore:
    """Handles persistence of user memory."""
    
    def __init__(self, data_dir: str = None):
        # Handle data directory path configuration
        if data_dir is None:
            # Check environment variable
            data_dir = os.environ.get("SAMANTHA_DATA_DIR", "data")
        
        self.data_dir = Path(data_dir)
        logger.info(f"Using data directory: {self.data_dir}")
        
        # Ensure directory exists
        try:
            self.data_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Data directory created/verified at {self.data_dir}")
        except Exception as e:
            logger.error(f"Error creating data directory: {e}")
            # Fallback to temporary directory if needed
            import tempfile
            self.data_dir = Path(tempfile.gettempdir()) / "samantha_data"
            self.data_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Using fallback data directory: {self.data_dir}")
        
        # Initialize memories dictionary
        self.memories = {}
        logger.info("MemoryStore initialized")
    
    async def load_all_memories(self):
        """Load all user memories from disk."""
        try:
            memory_files = list(self.data_dir.glob("*.json"))
            logger.info(f"Found {len(memory_files)} memory files")
            
            for memory_file in memory_files:
                try:
                    with open(memory_file, "r") as f:
                        memory_data = json.load(f)
                        user_id = memory_data.get("user_id")
                        if user_id:
                            self.memories[user_id] = UserMemory.from_dict(memory_data)
                            logger.info(f"Loaded memory for user {user_id}")
                except Exception as e:
                    logger.error(f"Error loading memory file {memory_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
    
    async def save_memory(self, memory: UserMemory):
        """Save user memory to disk."""
        try:
            memory.update_last_updated()
            self.memories[memory.user_id] = memory
            
            memory_path = self.data_dir / f"{memory.user_id}.json"
            with open(memory_path, "w") as f:
                json.dump(memory.to_dict(), f, indent=2)
            
            logger.info(f"Saved memory for user {memory.user_id}")
        except Exception as e:
            logger.error(f"Error saving memory for user {memory.user_id}: {e}")
    
    async def get_memory(self, user_id: str) -> UserMemory:
        """Get user memory, creating if not exists."""
        try:
            if user_id not in self.memories:
                logger.info(f"Creating new memory for user {user_id}")
                self.memories[user_id] = UserMemory(user_id=user_id)
                await self.save_memory(self.memories[user_id])
            
            return self.memories[user_id]
        except Exception as e:
            logger.error(f"Error getting memory for user {user_id}: {e}")
            # Return a new memory as fallback
            return UserMemory(user_id=user_id)
    
    async def delete_memory(self, user_id: str) -> bool:
        """Delete user memory."""
        try:
            if user_id in self.memories:
                del self.memories[user_id]
                
                memory_path = self.data_dir / f"{user_id}.json"
                if memory_path.exists():
                    memory_path.unlink()
                    logger.info(f"Deleted memory for user {user_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error deleting memory for user {user_id}: {e}")
            return False

# -------------------------------------------------
# Memory Extraction
# -------------------------------------------------

class MemoryExtractor:
    """Extracts memory patterns from conversations."""
    
    def __init__(self):
        self.learning_patterns = {
            "visual": [
                "see", "look", "view", "watch", "picture", "visualize", 
                "diagram", "chart", "graph", "image", "video", "visual", 
                "appear", "observe"
            ],
            "auditory": [
                "hear", "listen", "sound", "talk", "discuss", "tell", 
                "audio", "voice", "podcast", "conversation", "verbal", 
                "speak", "dialogue", "auditory"
            ],
            "reading_writing": [
                "read", "write", "note", "text", "book", "document", 
                "article", "essay", "report", "list", "outline", "word", 
                "reading", "writing"
            ],
            "kinesthetic": [
                "do", "act", "feel", "touch", "experience", "practice", 
                "hands-on", "exercise", "activity", "movement", "build", 
                "create", "physical", "kinesthetic"
            ]
        }
        self.topic_keywords = {}  # Will be updated dynamically
        self.learning_intent_patterns = [
            "learn about", "understand", "study", "interested in", 
            "want to know", "curious about", "explain", "tell me about"
        ]
        logger.info("MemoryExtractor initialized")
    
    def extract_learning_style(self, text: str) -> LearningStyle:
        """Extract learning style indicators from text."""
        try:
            doc = nlp(text.lower())
            
            # Initialize counters
            style_counts = {
                "visual": 0,
                "auditory": 0,
                "reading_writing": 0,
                "kinesthetic": 0
            }
            
            # Count occurrences of learning style indicators
            for token in doc:
                for style, indicators in self.learning_patterns.items():
                    if token.lemma_ in indicators:
                        style_counts[style] += 1
            
            # Create learning style object
            total = sum(style_counts.values())
            if total == 0:
                return LearningStyle(0.25, 0.25, 0.25, 0.25)
            
            return LearningStyle(
                visual=style_counts["visual"] / total,
                auditory=style_counts["auditory"] / total,
                reading_writing=style_counts["reading_writing"] / total,
                kinesthetic=style_counts["kinesthetic"] / total
            )
        except Exception as e:
            logger.error(f"Error extracting learning style: {e}")
            # Return balanced fallback
            return LearningStyle(0.25, 0.25, 0.25, 0.25)
    
    def extract_topics(self, text: str) -> Dict[str, float]:
        """Extract topics with confidence scores from text."""
        try:
            doc = nlp(text)
            topics = {}
            
            # Extract noun phrases as potential topics
            for chunk in doc.noun_chunks:
                # Filter out common pronouns and determiners
                if chunk.root.pos_ == "NOUN" and len(chunk.text) > 3:
                    topic_name = chunk.text.lower()
                    topics[topic_name] = 0.7  # Default confidence
            
            # Extract topics from learning intent patterns
            for pattern in self.learning_intent_patterns:
                if pattern in text.lower():
                    # Find what comes after the pattern
                    start_idx = text.lower().find(pattern) + len(pattern)
                    end_idx = text.find(".", start_idx)
                    if end_idx == -1:
                        end_idx = len(text)
                    
                    topic_text = text[start_idx:end_idx].strip()
                    if topic_text:
                        topic_doc = nlp(topic_text)
                        for chunk in topic_doc.noun_chunks:
                            if chunk.root.pos_ == "NOUN" and len(chunk.text) > 3:
                                topic_name = chunk.text.lower()
                                topics[topic_name] = 0.9  # Higher confidence
            
            return topics
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return {}
    
    def extract_conversation_style(self, text: str) -> ConversationStyle:
        """Extract conversation style preferences from text."""
        try:
            doc = nlp(text.lower())
            
            # Initialize metrics
            formality = 0.5
            verbosity = 0.5
            technical_level = 0.5
            
            # Formality indicators
            formal_indicators = ["formal", "professional", "academic", "proper", "official"]
            casual_indicators = ["casual", "informal", "relaxed", "conversational", "friendly"]
            
            # Verbosity indicators
            detailed_indicators = ["detailed", "thorough", "comprehensive", "elaborate", "in-depth"]
            concise_indicators = ["concise", "brief", "short", "summarize", "quick"]
            
            # Technical level indicators
            technical_indicators = ["technical", "advanced", "complex", "detailed", "specialized"]
            simple_indicators = ["simple", "basic", "easy", "beginner", "straightforward"]
            
            # Check for indicators
            text_lower = text.lower()
            
            # Check formality
            for indicator in formal_indicators:
                if indicator in text_lower:
                    formality += 0.1
            for indicator in casual_indicators:
                if indicator in text_lower:
                    formality -= 0.1
            
            # Check verbosity
            for indicator in detailed_indicators:
                if indicator in text_lower:
                    verbosity += 0.1
            for indicator in concise_indicators:
                if indicator in text_lower:
                    verbosity -= 0.1
            
            # Check technical level
            for indicator in technical_indicators:
                if indicator in text_lower:
                    technical_level += 0.1
            for indicator in simple_indicators:
                if indicator in text_lower:
                    technical_level -= 0.1
            
            # Ensure values remain in range 0.0-1.0
            formality = max(0.0, min(1.0, formality))
            verbosity = max(0.0, min(1.0, verbosity))
            technical_level = max(0.0, min(1.0, technical_level))
            
            return ConversationStyle(
                formality=formality,
                verbosity=verbosity,
                technical_level=technical_level
            )
        except Exception as e:
            logger.error(f"Error extracting conversation style: {e}")
            # Return balanced fallback
            return ConversationStyle()
    
    def extract_preferences(self, text: str) -> Dict[str, Dict[str, float]]:
        """Extract preferences from text."""
        try:
            preference_categories = {
                "movies": ["movie", "film", "watch", "cinema"],
                "books": ["book", "read", "novel", "author"],
                "music": ["music", "song", "listen", "artist", "band"],
                "food": ["food", "eat", "dish", "meal", "cuisine"],
                "hobbies": ["hobby", "activity", "enjoy", "pastime", "interest"]
            }
            
            # Initialize preferences
            preferences = {}
            
            # Analyze text for preferences in each category
            for category, keywords in preference_categories.items():
                category_matches = []
                
                # Check for category mentions
                for keyword in keywords:
                    if keyword in text.lower():
                        # Look for related entities
                        doc = nlp(text)
                        for ent in doc.ents:
                            if ent.label_ in ["PERSON", "ORG", "WORK_OF_ART", "PRODUCT"]:
                                # Check proximity to keyword
                                keyword_indices = [i for i, token in enumerate(doc) if token.text.lower() == keyword]
                                for idx in keyword_indices:
                                    # If entity is within 10 tokens of keyword
                                    entity_idx = ent.start
                                    if abs(idx - entity_idx) < 10:
                                        category_matches.append((ent.text, 0.8))
                        
                        # Also look for noun chunks near the keyword
                        keyword_indices = [i for i, token in enumerate(doc) if token.text.lower() == keyword]
                        for chunk in doc.noun_chunks:
                            chunk_start = chunk.start
                            for idx in keyword_indices:
                                if abs(idx - chunk_start) < 5 and chunk.text.lower() != keyword:
                                    category_matches.append((chunk.text, 0.7))
                
                if category_matches:
                    preferences[category] = {item: score for item, score in category_matches}
            
            return preferences
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}")
            return {}
    
    def extract_system_prompt(self, text: str) -> Optional[str]:
        """Extract potential system prompt from text."""
        try:
            # Look for explicit system prompt indicators
            system_prompt_indicators = [
                "system prompt:",
                "use this system prompt:",
                "here's a system prompt:",
                "this is a good system prompt:"
            ]
            
            for indicator in system_prompt_indicators:
                if indicator in text.lower():
                    start_idx = text.lower().find(indicator) + len(indicator)
                    # Find the end (period, newline, or end of text)
                    for end_marker in [".", "\n"]:
                        end_idx = text.find(end_marker, start_idx)
                        if end_idx != -1:
                            break
                    else:
                        end_idx = len(text)
                    
                    prompt_text = text[start_idx:end_idx].strip()
                    if prompt_text and len(prompt_text) > 20:  # Must be substantial
                        return prompt_text
            
            return None
        except Exception as e:
            logger.error(f"Error extracting system prompt: {e}")
            return None
    
    def process_conversation(self, messages: List[Dict]) -> Dict:
        """Process a full conversation to extract memory elements."""
        try:
            all_text = ""
            
            # Handle different message formats
            for message in messages:
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        all_text += content + " "
                elif isinstance(message, str):
                    all_text += message + " "
            
            if not all_text.strip():
                logger.warning("No text content found in conversation")
                return {
                    "learning_style": LearningStyle(0.25, 0.25, 0.25, 0.25),
                    "topics": {},
                    "conversation_style": ConversationStyle(),
                    "preferences": {},
                    "system_prompt": None
                }
            
            # Extract all memory elements
            learning_style = self.extract_learning_style(all_text)
            topics = self.extract_topics(all_text)
            conversation_style = self.extract_conversation_style(all_text)
            preferences = self.extract_preferences(all_text)
            system_prompt = self.extract_system_prompt(all_text)
            
            return {
                "learning_style": learning_style,
                "topics": topics,
                "conversation_style": conversation_style,
                "preferences": preferences,
                "system_prompt": system_prompt
            }
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            # Return empty extraction as fallback
            return {
                "learning_style": LearningStyle(0.25, 0.25, 0.25, 0.25),
                "topics": {},
                "conversation_style": ConversationStyle(),
                "preferences": {},
                "system_prompt": None
            }

# -------------------------------------------------
# MCP Server Implementation
# -------------------------------------------------

@dataclass
class SamanthaContext:
    """Context for the Samantha MCP server."""
    memory_store: MemoryStore
    memory_extractor: MemoryExtractor

@asynccontextmanager
async def samantha_lifespan(server: FastMCP) -> AsyncIterator[SamanthaContext]:
    """Manage the Samantha server lifecycle."""
    # Initialize on startup
    logger.info("Initializing Samantha MCP server...")
    memory_store = MemoryStore()
    await memory_store.load_all_memories()
    memory_extractor = MemoryExtractor()
    
    try:
        yield SamanthaContext(
            memory_store=memory_store,
            memory_extractor=memory_extractor
        )
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down Samantha MCP server...")

# Create MCP server
mcp = FastMCP(
    "Samantha Memory Manager",
    lifespan=samantha_lifespan,
    dependencies=[
        "spacy", "numpy", "pandas", "nltk", "fastapi",
        "streamlit", "plotly", "networkx"
    ]
)

# Helper function to get the Samantha context
def get_samantha_context() -> SamanthaContext:
    """Get the current Samantha context."""
    try:
        return mcp.current_context.request_context.lifespan_context
    except Exception as e:
        logger.error(f"Error getting Samantha context: {e}")
        # Create emergency fallback context
        return SamanthaContext(
            memory_store=MemoryStore(),
            memory_extractor=MemoryExtractor()
        )

# -------------------------------------------------
# Resources
# -------------------------------------------------

@mcp.resource("memory://{user_id}/profile")
async def get_user_profile(user_id: str) -> str:
    """Get the user's memory profile."""
    try:
        logger.info(f"Accessing profile for user {user_id}")
        context = get_samantha_context()
        memory = await context.memory_store.get_memory(user_id)
        return memory.get_llm_friendly_representation()
    except Exception as e:
        logger.error(f"Error getting user profile for {user_id}: {e}")
        return f"Error retrieving profile for user {user_id}: {str(e)}"

@mcp.resource("memory://{user_id}/learning_style")
async def get_learning_style(user_id: str) -> str:
    """Get the user's learning style."""
    try:
        logger.info(f"Accessing learning style for user {user_id}")
        context = get_samantha_context()
        memory = await context.memory_store.get_memory(user_id)
        return memory.learning_style.get_descriptive_text()
    except Exception as e:
        logger.error(f"Error getting learning style for {user_id}: {e}")
        return f"Error retrieving learning style for user {user_id}: {str(e)}"

@mcp.resource("memory://{user_id}/topics")
async def get_topics(user_id: str) -> str:
    """Get the user's topics of interest."""
    try:
        logger.info(f"Accessing topics for user {user_id}")
        context = get_samantha_context()
        memory = await context.memory_store.get_memory(user_id)
        
        output = "# Knowledge Areas\n\n"
        sorted_topics = sorted(memory.topics.values(), key=lambda t: t.mastery, reverse=True)
        
        if not sorted_topics:
            return "No topics have been identified yet."
        
        for topic in sorted_topics:
            mastery_level = "Expert" if topic.mastery > 0.8 else "Advanced" if topic.mastery > 0.6 else "Intermediate" if topic.mastery > 0.3 else "Beginner"
            last_updated = datetime.fromisoformat(topic.last_updated).strftime("%Y-%m-%d")
            
            output += f"## {topic.name}\n"
            output += f"- Mastery: {mastery_level} ({topic.mastery:.2f})\n"
            output += f"- Last updated: {last_updated}\n"
            
            if topic.related_topics:
                output += "- Related topics:\n"
                for related in topic.related_topics:
                    output += f"  - {related}\n"
            
            output += "\n"
        
        return output
    except Exception as e:
        logger.error(f"Error getting topics for {user_id}: {e}")
        return f"Error retrieving topics for user {user_id}: {str(e)}"

@mcp.resource("memory://{user_id}/system_prompts")
async def get_system_prompts(user_id: str) -> str:
    """Get the user's effective system prompts."""
    try:
        logger.info(f"Accessing system prompts for user {user_id}")
        context = get_samantha_context()
        memory = await context.memory_store.get_memory(user_id)
        
        output = "# Effective System Prompts\n\n"
        sorted_prompts = sorted(memory.system_prompts, key=lambda p: p.effectiveness, reverse=True)
        
        if not sorted_prompts:
            return "No system prompts have been saved yet."
        
        for i, prompt in enumerate(sorted_prompts):
            effectiveness = "High" if prompt.effectiveness > 0.7 else "Medium" if prompt.effectiveness > 0.4 else "Low"
            last_used = datetime.fromisoformat(prompt.last_used).strftime("%Y-%m-%d")
            
            output += f"## Prompt {i+1}\n"
            if prompt.description:
                output += f"### {prompt.description}\n\n"
            output += f"- Effectiveness: {effectiveness}\n"
            output += f"- Last used: {last_used}\n"
            
            if prompt.topics:
                output += "- Related topics: " + ", ".join(prompt.topics) + "\n"
            
            output += "\n```\n" + prompt.content + "\n```\n\n"
        
        return output
    except Exception as e:
        logger.error(f"Error getting system prompts for {user_id}: {e}")
        return f"Error retrieving system prompts for user {user_id}: {str(e)}"

@mcp.resource("memory://{user_id}/preferences/{category}")
async def get_preferences(user_id: str, category: str) -> str:
    """Get the user's preferences for a specific category."""
    try:
        logger.info(f"Accessing {category} preferences for user {user_id}")
        context = get_samantha_context()
        memory = await context.memory_store.get_memory(user_id)
        
        if category not in memory.preferences:
            return f"No preferences found for category '{category}'."
        
        preference = memory.preferences[category]
        top_items = preference.get_top_items(10)
        
        if not top_items:
            return f"No preference items found for category '{category}'."
        
        output = f"# {category.title()} Preferences\n\n"
        
        for item, score in top_items:
            strength = "Strongly likes" if score > 0.8 else "Likes" if score > 0.5 else "Somewhat likes"
            output += f"- {item}: {strength}\n"
        
        return output
    except Exception as e:
        logger.error(f"Error getting preferences for {user_id}/{category}: {e}")
        return f"Error retrieving preferences for user {user_id}, category {category}: {str(e)}"

@mcp.resource("memory://{user_id}/conversation_style")
async def get_conversation_style(user_id: str) -> str:
    """Get the user's conversation style preferences."""
    try:
        logger.info(f"Accessing conversation style for user {user_id}")
        context = get_samantha_context()
        memory = await context.memory_store.get_memory(user_id)
        
        style = memory.conversation_style
        output = "# Communication Preferences\n\n"
        output += memory.conversation_style.get_descriptive_text() + "\n\n"
        
        output += "## Detailed Analysis\n\n"
        
        # Formality
        formality = "Very formal" if style.formality > 0.8 else "Formal" if style.formality > 0.6 else "Neutral" if style.formality > 0.4 else "Casual" if style.formality > 0.2 else "Very casual"
        output += f"- Formality: {formality} ({style.formality:.2f})\n"
        
        # Verbosity
        verbosity = "Very detailed" if style.verbosity > 0.8 else "Detailed" if style.verbosity > 0.6 else "Balanced" if style.verbosity > 0.4 else "Concise" if style.verbosity > 0.2 else "Very concise"
        output += f"- Verbosity: {verbosity} ({style.verbosity:.2f})\n"
        
        # Technical level
        technical = "Very technical" if style.technical_level > 0.8 else "Technical" if style.technical_level > 0.6 else "Moderate" if style.technical_level > 0.4 else "Simple" if style.technical_level > 0.2 else "Very simple"
        output += f"- Technical level: {technical} ({style.technical_level:.2f})\n"
        
        return output
    except Exception as e:
        logger.error(f"Error getting conversation style for {user_id}: {e}")
        return f"Error retrieving conversation style for user {user_id}: {str(e)}"

# -------------------------------------------------
# Tools
# -------------------------------------------------

@mcp.tool()
async def extract_memory(conversation: Any, user_id: str, ctx: Context) -> str:
    """
    Extract memory from a conversation.
    
    Args:
        conversation: List of conversation messages with 'role' and 'content'
        user_id: The user ID to store memory for
    
    Returns:
        A summary of the extracted memory
    """
    try:
        logger.info(f"Extracting memory for user {user_id}")
        memory_store = ctx.request_context.lifespan_context.memory_store
        memory_extractor = ctx.request_context.lifespan_context.memory_extractor
        
        # Get existing memory
        memory = await memory_store.get_memory(user_id)
        
        # Handle different conversation formats
        if isinstance(conversation, list):
            conv_to_process = conversation
        elif isinstance(conversation, str):
            conv_to_process = [{"role": "user", "content": conversation}]
        elif conversation is None:
            # Use minimal conversation to avoid errors
            conv_to_process = [{"role": "user", "content": ""}]
        else:
            # Fallback for any other format
            try:
                conv_to_process = list(conversation)
            except:
                conv_to_process = [{"role": "user", "content": str(conversation)}]
        
        # Extract new memory elements
        extracted = memory_extractor.process_conversation(conv_to_process)
        
        # Update learning style
        if extracted["learning_style"]:
            memory.learning_style.update(extracted["learning_style"])
        
        # Update topics
        for topic_name, confidence in extracted["topics"].items():
            if topic_name in memory.topics:
                memory.topics[topic_name].update_mastery(confidence)
            else:
                memory.topics[topic_name] = Topic(name=topic_name, mastery=confidence)
        
        # Update conversation style
        if extracted["conversation_style"]:
            memory.conversation_style.update(extracted["conversation_style"])
        
        # Update preferences
        for category, items in extracted["preferences"].items():
            if category not in memory.preferences:
                memory.preferences[category] = Preference(category=category)
            
            for item, score in items.items():
                memory.preferences[category].update_item(item, score)
        
        # Add system prompt if found
        if extracted["system_prompt"]:
            topics = list(extracted["topics"].keys())
            system_prompt = SystemPrompt(
                content=extracted["system_prompt"],
                effectiveness=0.5,  # Initial score
                topics=topics[:5],  # Associate with top 5 topics
                description="Extracted from conversation"
            )
            memory.system_prompts.append(system_prompt)
        
        # Save updated memory
        await memory_store.save_memory(memory)
        
        # Return summary
        topics_count = len(extracted["topics"])
        preferences_list = ", ".join(extracted["preferences"].keys()) if extracted["preferences"] else "None identified"
        
        return f"""Memory extracted and updated for user {user_id}:
- Learning style: {memory.learning_style.get_primary_style()} (primary), {memory.learning_style.get_secondary_style()} (secondary)
- Topics: {topics_count} topics identified or updated
- Conversation style: Updated
- Preferences: {preferences_list}
- System prompt: {"Extracted" if extracted["system_prompt"] else "None identified"}"""
    except Exception as e:
        logger.error(f"Error extracting memory: {e}")
        return f"Error extracting memory: {str(e)}"

@mcp.tool()
async def add_system_prompt(prompt: str, description: str, user_id: str, topics: List[str], ctx: Context) -> str:
    """
    Add a new system prompt.
    
    Args:
        prompt: The system prompt content
        description: A description of the prompt
        user_id: The user ID to add the prompt for
        topics: Related topics for this prompt
    
    Returns:
        Confirmation message
    """
    try:
        logger.info(f"Adding system prompt for user {user_id}")
        memory_store = ctx.request_context.lifespan_context.memory_store
        
        # Get existing memory
        memory = await memory_store.get_memory(user_id)
        
        # Create new system prompt
        system_prompt = SystemPrompt(
            content=prompt,
            effectiveness=0.5,  # Initial score
            topics=topics,
            description=description
        )
        
        # Add to memory
        memory.system_prompts.append(system_prompt)
        
        # Save updated memory
        await memory_store.save_memory(memory)
        
        return f"System prompt added for user {user_id} with description: {description}"
    except Exception as e:
        logger.error(f"Error adding system prompt: {e}")
        return f"Error adding system prompt: {str(e)}"

@mcp.tool()
async def update_system_prompt_effectiveness(prompt_index: int, effectiveness: float, user_id: str, ctx: Context) -> str:
    """
    Update the effectiveness score of a system prompt.
    
    Args:
        prompt_index: The index of the prompt to update
        effectiveness: New effectiveness score (0.0 to 1.0)
        user_id: The user ID
    
    Returns:
        Confirmation message
    """
    try:
        logger.info(f"Updating system prompt effectiveness for user {user_id}")
        memory_store = ctx.request_context.lifespan_context.memory_store
        
        # Get existing memory
        memory = await memory_store.get_memory(user_id)
        
        # Validate parameters
        if not memory.system_prompts:
            return f"No system prompts found for user {user_id}."
        
        if not (0 <= prompt_index < len(memory.system_prompts)):
            return f"Error: Invalid prompt index. Valid range is 0-{len(memory.system_prompts)-1}."
        
        effectiveness = max(0.0, min(1.0, effectiveness))
        
        # Update effectiveness
        memory.system_prompts[prompt_index].update_effectiveness(effectiveness)
        
        # Save updated memory
        await memory_store.save_memory(memory)
        
        return f"Updated effectiveness of system prompt #{prompt_index} to {effectiveness:.2f}"
    except Exception as e:
        logger.error(f"Error updating system prompt effectiveness: {e}")
        return f"Error updating system prompt effectiveness: {str(e)}"

@mcp.tool()
async def update_topic_mastery(topic_name: str, mastery: float, user_id: str, ctx: Context) -> str:
    """
    Update the mastery level of a topic.
    
    Args:
        topic_name: The name of the topic to update
        mastery: New mastery level (0.0 to 1.0)
        user_id: The user ID
    
    Returns:
        Confirmation message
    """
    try:
        logger.info(f"Updating topic mastery for user {user_id}")
        memory_store = ctx.request_context.lifespan_context.memory_store
        
        # Get existing memory
        memory = await memory_store.get_memory(user_id)
        
        # Validate parameters
        mastery = max(0.0, min(1.0, mastery))
        
        # Update or create topic
        if topic_name in memory.topics:
            memory.topics[topic_name].update_mastery(mastery)
            action = "Updated"
        else:
            memory.topics[topic_name] = Topic(name=topic_name, mastery=mastery)
            action = "Created"
        
        # Save updated memory
        await memory_store.save_memory(memory)
        
        return f"{action} topic '{topic_name}' with mastery level {mastery:.2f}"
    except Exception as e:
        logger.error(f"Error updating topic mastery: {e}")
        return f"Error updating topic mastery: {str(e)}"

@mcp.tool()
async def relate_topics(topic_name: str, related_topics: List[str], user_id: str, ctx: Context) -> str:
    """
    Establish relationships between topics.
    
    Args:
        topic_name: The primary topic
        related_topics: List of related topics
        user_id: The user ID
    
    Returns:
        Confirmation message
    """
    try:
        logger.info(f"Relating topics for user {user_id}")
        memory_store = ctx.request_context.lifespan_context.memory_store
        
        # Get existing memory
        memory = await memory_store.get_memory(user_id)
        
        # Ensure primary topic exists
        if topic_name not in memory.topics:
            memory.topics[topic_name] = Topic(name=topic_name)
        
        # Ensure related topics exist
        for related in related_topics:
            if related not in memory.topics:
                memory.topics[related] = Topic(name=related)
        
        # Update relationships
        memory.topics[topic_name].related_topics = related_topics
        
        # Save updated memory
        await memory_store.save_memory(memory)
        
        return f"Related topic '{topic_name}' to {len(related_topics)} other topics: {', '.join(related_topics)}"
    except Exception as e:
        logger.error(f"Error relating topics: {e}")
        return f"Error relating topics: {str(e)}"

@mcp.tool()
async def update_learning_style(visual: float, auditory: float, reading_writing: float, kinesthetic: float, user_id: str, ctx: Context) -> str:
    """
    Manually update learning style.
    
    Args:
        visual: Visual learning score (0.0 to 1.0)
        auditory: Auditory learning score (0.0 to 1.0)
        reading_writing: Reading/writing learning score (0.0 to 1.0)
        kinesthetic: Kinesthetic learning score (0.0 to 1.0)
        user_id: The user ID
    
    Returns:
        Confirmation message with new learning style description
    """
    try:
        logger.info(f"Updating learning style for user {user_id}")
        memory_store = ctx.request_context.lifespan_context.memory_store
        
        # Get existing memory
        memory = await memory_store.get_memory(user_id)
        
        # Validate parameters
        visual = max(0.0, min(1.0, visual))
        auditory = max(0.0, min(1.0, auditory))
        reading_writing = max(0.0, min(1.0, reading_writing))
        kinesthetic = max(0.0, min(1.0, kinesthetic))
        
        # Create new learning style
        new_style = LearningStyle(
            visual=visual,
            auditory=auditory,
            reading_writing=reading_writing,
            kinesthetic=kinesthetic
        )
        
        # Update learning style (overwrite completely)
        memory.learning_style = new_style
        
        # Save updated memory
        await memory_store.save_memory(memory)
        
        # Get descriptive text
        description = memory.learning_style.get_descriptive_text()
        
        return f"Updated learning style for user {user_id}. New learning style:\n\n{description}"
    except Exception as e:
        logger.error(f"Error updating learning style: {e}")
        return f"Error updating learning style: {str(e)}"

@mcp.tool()
async def export_memory(user_id: str, format: str, ctx: Context) -> str:
    """
    Export user memory in the specified format.
    
    Args:
        user_id: The user ID
        format: Export format ("json", "mentorsync", "llama")
    
    Returns:
        The exported memory data
    """
    try:
        logger.info(f"Exporting memory for user {user_id} in {format} format")
        memory_store = ctx.request_context.lifespan_context.memory_store
        
        # Get existing memory
        memory = await memory_store.get_memory(user_id)
        
        # Export in specified format
        if format.lower() == "json":
            return json.dumps(memory.to_dict(), indent=2)
        
        elif format.lower() == "mentorsync":
            # Create MentorSync compatible format
            mentorsync_data = {
                "learning_style": {
                    "visual": memory.learning_style.visual,
                    "auditory": memory.learning_style.auditory,
                    "reading_writing": memory.learning_style.reading_writing,
                    "kinesthetic": memory.learning_style.kinesthetic,
                    "description": memory.learning_style.get_descriptive_text()
                },
                "knowledge_graph": {
                    "topics": [
                        {
                            "name": topic.name,
                            "mastery": topic.mastery,
                            "related_topics": topic.related_topics
                        }
                        for topic in memory.topics.values()
                    ]
                },
                "preferences": {
                    category: [
                        {"item": item, "score": score}
                        for item, score in pref.items.items()
                    ]
                    for category, pref in memory.preferences.items()
                }
            }
            return json.dumps(mentorsync_data, indent=2)
        
        elif format.lower() == "llama":
            # Create Llama-compatible format (simplified for LLM context)
            return memory.get_llm_friendly_representation()
        
        else:
            return f"Error: Unsupported export format. Please use 'json', 'mentorsync', or 'llama'."
    except Exception as e:
        logger.error(f"Error exporting memory: {e}")
        return f"Error exporting memory: {str(e)}"

@mcp.tool()
async def import_memory(user_id: str, data: str, format: str, ctx: Context) -> str:
    """
    Import user memory from the specified format.
    
    Args:
        user_id: The user ID
        data: The memory data to import
        format: Import format ("json", "mentorsync")
    
    Returns:
        Confirmation message
    """
    try:
        logger.info(f"Importing memory for user {user_id} from {format} format")
        memory_store = ctx.request_context.lifespan_context.memory_store
        
        try:
            if format.lower() == "json":
                # Parse JSON data
                memory_data = json.loads(data)
                
                # Ensure user_id is correct
                memory_data["user_id"] = user_id
                
                # Create memory object
                memory = UserMemory.from_dict(memory_data)
                
                # Save memory
                await memory_store.save_memory(memory)
                
                return f"Successfully imported memory for user {user_id} from JSON format."
            
            elif format.lower() == "mentorsync":
                # Parse MentorSync data
                mentorsync_data = json.loads(data)
                
                # Create new memory
                memory = UserMemory(user_id=user_id)
                
                # Import learning style
                if "learning_style" in mentorsync_data:
                    ls_data = mentorsync_data["learning_style"]
                    memory.learning_style = LearningStyle(
                        visual=ls_data.get("visual", 0.25),
                        auditory=ls_data.get("auditory", 0.25),
                        reading_writing=ls_data.get("reading_writing", 0.25),
                        kinesthetic=ls_data.get("kinesthetic", 0.25)
                    )
                
                # Import topics
                if "knowledge_graph" in mentorsync_data and "topics" in mentorsync_data["knowledge_graph"]:
                    for topic_data in mentorsync_data["knowledge_graph"]["topics"]:
                        topic = Topic(
                            name=topic_data["name"],
                            mastery=topic_data.get("mastery", 0.0),
                            related_topics=topic_data.get("related_topics", [])
                        )
                        memory.topics[topic.name] = topic
                
                # Import preferences
                if "preferences" in mentorsync_data:
                    for category, items in mentorsync_data["preferences"].items():
                        pref = Preference(category=category)
                        for item_data in items:
                            pref.items[item_data["item"]] = item_data["score"]
                        memory.preferences[category] = pref
                
                # Save memory
                await memory_store.save_memory(memory)
                
                return f"Successfully imported memory for user {user_id} from MentorSync format."
            
            else:
                return f"Error: Unsupported import format. Please use 'json' or 'mentorsync'."
        
        except Exception as e:
            logger.error(f"Error in data import: {e}")
            return f"Error importing memory data: {str(e)}"
    except Exception as e:
        logger.error(f"Error importing memory: {e}")
        return f"Error importing memory: {str(e)}"

@mcp.tool()
async def delete_user_memory(user_id: str, ctx: Context) -> str:
    """
    Delete a user's memory.
    
    Args:
        user_id: The user ID to delete
    
    Returns:
        Confirmation message
    """
    try:
        logger.info(f"Deleting memory for user {user_id}")
        memory_store = ctx.request_context.lifespan_context.memory_store
        
        # Delete memory
        success = await memory_store.delete_memory(user_id)
        
        if success:
            return f"Successfully deleted memory for user {user_id}."
        else:
            return f"No memory found for user {user_id}."
    except Exception as e:
        logger.error(f"Error deleting user memory: {e}")
        return f"Error deleting user memory: {str(e)}"

# -------------------------------------------------
# Prompts
# -------------------------------------------------

@mcp.prompt()
def analyze_learning_style(user_id: str) -> str:
    """Create a prompt to analyze a user's learning style."""
    return f"""Please analyze the memory profile for user {user_id}, focusing on their learning style preferences.

Review their VARK profile (Visual, Auditory, Reading/Writing, Kinesthetic) and provide tailored recommendations for how they can optimize their learning process.

Include specific examples of study techniques, content formats, and practical approaches that align with their learning style strengths.

Based on their existing knowledge areas and preferences, suggest how they might apply their learning style preferences to those specific domains.
"""

@mcp.prompt()
def generate_system_prompt(user_id: str, purpose: str) -> str:
    """Create a prompt to generate an effective system prompt for a user."""
    return f"""Based on the memory profile for user {user_id}, please create an effective system prompt for Claude that would provide a personalized experience aligned with their learning style, knowledge areas, and preferences.

The system prompt should be optimized for: {purpose}

Consider:
1. Their learning style (VARK profile)
2. Their conversation style preferences (formality, verbosity, technical level)
3. Their known areas of interest and knowledge levels
4. Any specific preferences that might be relevant

The goal is to create a system prompt that Claude could use to provide an experience tailored specifically to this user's needs and preferences without requiring explicit instruction each time.

Format your response with the system prompt inside triple backticks, followed by an explanation of how it addresses the user's specific preferences and learning style.
"""

@mcp.prompt()
def recommend_resources(user_id: str, topic: str) -> list[base.Message]:
    """Create a prompt sequence to recommend learning resources for a topic."""
    return [
        base.UserMessage(f"I'd like recommendations for learning resources on '{topic}' that match my learning style and preferences."),
        base.AssistantMessage(f"I'll help you find resources on '{topic}' tailored to your learning style. Let me check your memory profile..."),
        base.FunctionMessage(f"The user's memory profile shows they are primarily a {{learning_style}} learner with {{secondary_style}} tendencies. Their knowledge of {topic} is {{mastery_level}}. They prefer {{conversation_style}} communication."),
        base.AssistantMessage("Based on your learning style and preferences, here are some recommended resources:")
    ]

# Add signal handlers for graceful shutdown
def graceful_shutdown(signum, frame):
    """Handle shutdown signals properly to release resources."""
    logger.info(f"Received signal {signum}. Cleaning up resources...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# Run the server
if __name__ == "__main__":
    try:
        logger.info("Starting Samantha MCP server...")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running Samantha MCP server: {e}")
        sys.exit(1)