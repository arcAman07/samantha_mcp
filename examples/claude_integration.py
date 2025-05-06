"""
Example of using Samantha with Claude for personalized learning
"""

# This example demonstrates how to use Samantha with Claude to create 
# a personalized learning experience based on user memory

# Example Claude Prompt:

"""
You are a personalized learning assistant that adapts to the user's learning style.
Use the memory://{user_id}/profile resource to access the user's memory profile.

Based on their memory profile, please:
1. Tailor explanations to match their learning style (VARK)
2. Reference their interests and knowledge areas when providing examples
3. Match their preferred conversation style
4. Use system prompts that have worked well for them in the past

After the conversation, use the extract_memory tool to update their memory profile.
"""

# Example conversation flow:

USER_ID = "student123"

# Step 1: Fetch user profile from Samantha
# Claude would use: await ctx.read_resource(f"memory://{USER_ID}/profile")

# Example profile that might be returned:
USER_PROFILE = """
# User Memory Profile

## Learning Style

A strong visual learner with kinesthetic tendencies.

Recommended learning approaches:
- Use diagrams, charts, and visual representations
- Color-code important information
- Create mind maps for complex topics
- Use hands-on activities and practical exercises
- Take breaks and move around while learning
- Create physical models or demonstrations

## Communication Preferences

Prefers casual, detailed, and moderate communication.

## Knowledge Areas

- Python: Intermediate level
- Machine learning: Beginner level
- Data visualization: Advanced level
- Mathematics: Intermediate level
- Plus 3 more topics

## Personal Preferences

### Movies
- Interstellar
- The Matrix
- Inception

### Books
- Thinking, Fast and Slow
- Deep Learning
- The Visual Display of Quantitative Information
"""

# Step 2: User asks a question about a new topic
USER_QUESTION = "Can you explain neural networks to me? I'm trying to understand how they work."

# Step 3: Claude generates a personalized response based on the memory profile
CLAUDE_RESPONSE = """
I'd be happy to explain neural networks in a way that matches your visual learning style!

📊 **Visual Explanation of Neural Networks**

Let's imagine a neural network as a series of connected layers:

```
Input Layer       Hidden Layers       Output Layer
   O                O    O              O
   O       →       O    O       →      O
   O                O    O              O
  Data          Processing          Prediction
```

Since you're a Python programmer with experience in data visualization but are new to machine learning, let's relate this to concepts you're familiar with:

Think of each "neuron" (circle) as a function that takes multiple inputs, applies weights (like parameters in a visualization), and produces an output. It's similar to how you might use matplotlib to transform data into visual representations!

Would you like me to create an interactive visualization to help you explore this concept further? Given your kinesthetic learning tendencies, a hands-on approach might be helpful.
"""

# Step 4: After the conversation, extract memory from the interaction
CONVERSATION = [
    {"role": "user", "content": USER_QUESTION},
    {"role": "assistant", "content": CLAUDE_RESPONSE},
    {"role": "user", "content": "Yes, an interactive visualization would be great! I'd also like to understand how neural networks are trained."}
]

# Claude would use:
# await ctx.call_tool("extract_memory", {"conversation": CONVERSATION, "user_id": USER_ID})

# Example of what the extract_memory tool might extract:
EXTRACTED_MEMORY = {
    "learning_style": {
        "visual": 0.8,   # Reinforced by user's positive response to visual explanation
        "kinesthetic": 0.6,  # Reinforced by interest in interactive visualization
    },
    "topics": {
        "neural networks": 0.2,  # New topic with beginner mastery
        "machine learning": 0.25  # Slight increase in existing topic
    },
    "conversation_style": {
        "formality": 0.3,  # Casual
        "verbosity": 0.7,  # Detailed
        "technical_level": 0.5  # Moderate
    }
}

# Example of calling the update_topic_mastery tool to add a relationship
# await ctx.call_tool("relate_topics", {
#     "topic_name": "neural networks",
#     "related_topics": ["machine learning", "mathematics", "python"],
#     "user_id": USER_ID
# })

# Later, Claude could recommend a system prompt based on successful interactions
SYSTEM_PROMPT_RECOMMENDATION = """
You are a learning assistant for {USER_ID}. 
Always explain concepts with strong visual elements (diagrams, charts, illustrations).
Include interactive elements when possible to engage kinesthetic learning.
Connect new material to their knowledge of Python and data visualization.
Use a casual, approachable tone but provide detailed technical explanations.
Give practical examples that reinforce the visual understanding of concepts.
"""

# Claude could then save this effective system prompt:
# await ctx.call_tool("add_system_prompt", {
#     "prompt": SYSTEM_PROMPT_RECOMMENDATION,
#     "description": "Visual-focused learning assistant prompt",
#     "user_id": USER_ID,
#     "topics": ["machine learning", "neural networks", "python"]
# })

# This cycle continues with each interaction, gradually building
# a more accurate user memory profile for personalized experiences