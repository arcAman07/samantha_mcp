# dashboard.py
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Import from samantha.py
from samantha import (
    UserMemory, LearningStyle, Topic, SystemPrompt, 
    ConversationStyle, Preference, MemoryStore
)

# Set page config
st.set_page_config(
    page_title="Samantha Memory Manager",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize memory store
@st.cache_resource
def get_memory_store():
    """Get or create a MemoryStore instance."""
    try:
        memory_store = MemoryStore()
        asyncio.run(memory_store.load_all_memories())
        return memory_store
    except Exception as e:
        st.error(f"Error initializing memory store: {e}")
        # Fallback to empty store
        return MemoryStore()

memory_store = get_memory_store()

# Get all users
def get_all_users():
    """Get list of all users in the memory store."""
    try:
        user_ids = list(memory_store.memories.keys())
        return ["Select a user"] + sorted(user_ids)
    except Exception as e:
        st.error(f"Error getting users: {e}")
        return ["Select a user"]

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1976D2;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #1565C0;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .info-text {
        color: #555;
        font-size: 0.9rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h1 class='main-header'>Samantha</h1>", unsafe_allow_html=True)
st.sidebar.markdown("### Memory Management Dashboard")

# User selection
users = get_all_users()
selected_user = st.sidebar.selectbox("Select User", users)

# Create new user
with st.sidebar.expander("Create New User"):
    new_user_id = st.text_input("New User ID")
    create_user = st.button("Create User")
    
    if create_user and new_user_id:
        try:
            if new_user_id in memory_store.memories:
                st.error(f"User {new_user_id} already exists!")
            else:
                new_memory = UserMemory(user_id=new_user_id)
                asyncio.run(memory_store.save_memory(new_memory))
                st.success(f"Created user {new_user_id}")
                # Refresh users
                st.rerun()
        except Exception as e:
            st.error(f"Error creating user: {e}")

# Navigation
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Learning Style", "Topics", "System Prompts", "Preferences", "Conversation Style", "Import/Export"]
)

# Delete user
if selected_user != "Select a user":
    with st.sidebar.expander("⚠️ Danger Zone"):
        delete_user = st.button("Delete User", type="primary")
        
        if delete_user:
            confirm_delete = st.button("Confirm Deletion", type="primary")
            
            if confirm_delete:
                try:
                    asyncio.run(memory_store.delete_memory(selected_user))
                    st.success(f"Deleted user {selected_user}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting user: {e}")

# Main content
if selected_user == "Select a user":
    st.markdown("<h1 class='main-header'>Welcome to Samantha Memory Manager</h1>", unsafe_allow_html=True)
    st.markdown("""
    Samantha is an MCP (Model Context Protocol) server that extracts and manages user memory from LLM conversations.
    This memory can be shared across different LLMs, allowing for consistent personalization.
    
    ## Features
    
    - **Dynamic Memory Extraction** - Learning styles, topics, preferences, and more
    - **Learning Style Analysis** - VARK model with descriptive recommendations
    - **System Prompt Memory** - Store and rank effective system prompts
    - **Memory Pattern Recognition** - Identify user interests and preferences
    - **MCP Integration** - Works with Claude Desktop and other MCP clients
    - **Memory Export/Import** - Support for multiple formats
    
    ## Getting Started
    
    1. Create a new user using the sidebar
    2. Explore and manage the user's memory profile
    3. Connect to the MCP server with Claude Desktop
    
    Made with ❤️ by Samantha
    """)

else:
    try:
        # Load user memory
        memory = asyncio.run(memory_store.get_memory(selected_user))
        
        # Overview page
        if page == "Overview":
            st.markdown(f"<h1 class='main-header'>Memory Profile: {selected_user}</h1>", unsafe_allow_html=True)
            
            # Display summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='subsection-header'>Learning Style</h3>", unsafe_allow_html=True)
                st.markdown(f"**Primary**: {memory.learning_style.get_primary_style()}")
                st.markdown(f"**Secondary**: {memory.learning_style.get_secondary_style()}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='subsection-header'>Topics</h3>", unsafe_allow_html=True)
                st.markdown(f"**Total**: {len(memory.topics)}")
                if memory.topics:
                    top_topic = max(memory.topics.values(), key=lambda t: t.mastery)
                    st.markdown(f"**Top**: {top_topic.name} ({top_topic.mastery:.2f})")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='subsection-header'>System Prompts</h3>", unsafe_allow_html=True)
                st.markdown(f"**Total**: {len(memory.system_prompts)}")
                if memory.system_prompts:
                    top_prompt = max(memory.system_prompts, key=lambda p: p.effectiveness)
                    st.markdown(f"**Top**: {top_prompt.description[:20]}... ({top_prompt.effectiveness:.2f})")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='subsection-header'>Preferences</h3>", unsafe_allow_html=True)
                st.markdown(f"**Categories**: {len(memory.preferences)}")
                preference_cats = ", ".join(list(memory.preferences.keys())[:2])
                if len(memory.preferences) > 2:
                    preference_cats += f" + {len(memory.preferences) - 2} more"
                st.markdown(f"**Types**: {preference_cats}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Memory profile
            st.markdown("<h2 class='section-header'>Memory Profile</h2>", unsafe_allow_html=True)
            st.markdown(memory.get_llm_friendly_representation())
            
            # Visualizations
            st.markdown("<h2 class='section-header'>Visualizations</h2>", unsafe_allow_html=True)
            
            # Learning style radar chart
            st.markdown("<h3 class='subsection-header'>Learning Style Distribution</h3>", unsafe_allow_html=True)
            
            # Prepare data for radar chart
            ls = memory.learning_style.get_normalized()
            ls_data = {
                'Category': ['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic'],
                'Value': [ls.visual, ls.auditory, ls.reading_writing, ls.kinesthetic]
            }
            ls_df = pd.DataFrame(ls_data)
            
            # Create radar chart
            fig = px.line_polar(
                ls_df, r='Value', theta='Category', line_close=True,
                range_r=[0, 1], markers=True, color_discrete_sequence=['#1E88E5']
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeline
            st.markdown("<h3 class='subsection-header'>Memory Timeline</h3>", unsafe_allow_html=True)
            
            # Get all timestamps from different memory elements
            timeline_data = []
            
            # Add user creation
            timeline_data.append({
                'Event': 'User Created',
                'Timestamp': datetime.fromisoformat(memory.created_at),
                'Type': 'user'
            })
            
            # Add topic updates
            for topic_name, topic in memory.topics.items():
                timeline_data.append({
                    'Event': f'Topic: {topic_name}',
                    'Timestamp': datetime.fromisoformat(topic.last_updated),
                    'Type': 'topic'
                })
            
            # Add system prompt updates
            for i, prompt in enumerate(memory.system_prompts):
                timeline_data.append({
                    'Event': f'System Prompt: {prompt.description[:20]}...',
                    'Timestamp': datetime.fromisoformat(prompt.last_used),
                    'Type': 'prompt'
                })
            
            # Create timeline dataframe
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                timeline_df = timeline_df.sort_values('Timestamp')
                
                # Create timeline chart
                fig = px.scatter(
                    timeline_df, x='Timestamp', y='Event', color='Type',
                    color_discrete_map={'user': '#1E88E5', 'topic': '#4CAF50', 'prompt': '#FF9800'},
                    title='Memory Event Timeline'
                )
                fig.update_traces(marker=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)
        
        # Learning Style page
        elif page == "Learning Style":
            st.markdown(f"<h1 class='main-header'>Learning Style: {selected_user}</h1>", unsafe_allow_html=True)
            
            # Display current learning style
            st.markdown("<h2 class='section-header'>Current Learning Style</h2>", unsafe_allow_html=True)
            st.markdown(memory.learning_style.get_descriptive_text())
            
            # Learning style visualization
            st.markdown("<h2 class='section-header'>Learning Style Distribution</h2>", unsafe_allow_html=True)
            
            # Display current values
            ls = memory.learning_style
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Visual", f"{ls.visual:.2f}")
                st.progress(ls.visual)
            
            with col2:
                st.metric("Auditory", f"{ls.auditory:.2f}")
                st.progress(ls.auditory)
            
            with col3:
                st.metric("Reading/Writing", f"{ls.reading_writing:.2f}")
                st.progress(ls.reading_writing)
            
            with col4:
                st.metric("Kinesthetic", f"{ls.kinesthetic:.2f}")
                st.progress(ls.kinesthetic)
            
            # Radar chart
            ls_norm = memory.learning_style.get_normalized()
            ls_data = {
                'Category': ['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic'],
                'Value': [ls_norm.visual, ls_norm.auditory, ls_norm.reading_writing, ls_norm.kinesthetic]
            }
            ls_df = pd.DataFrame(ls_data)
            
            fig = px.line_polar(
                ls_df, r='Value', theta='Category', line_close=True,
                range_r=[0, 1], markers=True, color_discrete_sequence=['#1E88E5']
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Update learning style
            st.markdown("<h2 class='section-header'>Update Learning Style</h2>", unsafe_allow_html=True)
            
            with st.form("update_learning_style"):
                visual = st.slider("Visual", 0.0, 1.0, ls.visual, 0.01)
                auditory = st.slider("Auditory", 0.0, 1.0, ls.auditory, 0.01)
                reading_writing = st.slider("Reading/Writing", 0.0, 1.0, ls.reading_writing, 0.01)
                kinesthetic = st.slider("Kinesthetic", 0.0, 1.0, ls.kinesthetic, 0.01)
                
                update_ls_button = st.form_submit_button("Update Learning Style")
                
                if update_ls_button:
                    # Normalize values
                    total = visual + auditory + reading_writing + kinesthetic
                    if total > 0:
                        memory.learning_style = LearningStyle(
                            visual=visual,
                            auditory=auditory,
                            reading_writing=reading_writing,
                            kinesthetic=kinesthetic
                        )
                        
                        asyncio.run(memory_store.save_memory(memory))
                        st.success("Learning style updated!")
                        st.rerun()
        
        # Topics page
        elif page == "Topics":
            st.markdown(f"<h1 class='main-header'>Knowledge Topics: {selected_user}</h1>", unsafe_allow_html=True)
            
            # Add new topic
            with st.expander("Add New Topic"):
                with st.form("add_topic"):
                    new_topic_name = st.text_input("Topic Name")
                    new_topic_mastery = st.slider("Mastery Level", 0.0, 1.0, 0.5, 0.01)
                    related_topics = st.multiselect(
                        "Related Topics",
                        options=list(memory.topics.keys()),
                        default=[]
                    )
                    
                    add_topic_button = st.form_submit_button("Add Topic")
                    
                    if add_topic_button and new_topic_name:
                        if new_topic_name in memory.topics:
                            st.error(f"Topic {new_topic_name} already exists!")
                        else:
                            memory.topics[new_topic_name] = Topic(
                                name=new_topic_name,
                                mastery=new_topic_mastery,
                                related_topics=related_topics
                            )
                            
                            asyncio.run(memory_store.save_memory(memory))
                            st.success(f"Added topic {new_topic_name}")
                            st.rerun()
            
            # Topics table
            st.markdown("<h2 class='section-header'>Topics</h2>", unsafe_allow_html=True)
            
            if not memory.topics:
                st.info("No topics have been added yet.")
            else:
                # Create dataframe
                topics_data = []
                for topic_name, topic in memory.topics.items():
                    topics_data.append({
                        'Topic': topic_name,
                        'Mastery': topic.mastery,
                        'Last Updated': datetime.fromisoformat(topic.last_updated).strftime("%Y-%m-%d"),
                        'Related Topics': ", ".join(topic.related_topics),
                    })
                
                topics_df = pd.DataFrame(topics_data)
                topics_df = topics_df.sort_values('Mastery', ascending=False)
                
                # Display table
                st.dataframe(topics_df, use_container_width=True)
                
                # Topic network graph
                st.markdown("<h2 class='section-header'>Topic Network</h2>", unsafe_allow_html=True)
                
                # Create graph
                G = nx.Graph()
                
                # Add nodes
                for topic_name, topic in memory.topics.items():
                    mastery = topic.mastery
                    G.add_node(topic_name, mastery=mastery)
                
                # Add edges
                for topic_name, topic in memory.topics.items():
                    for related in topic.related_topics:
                        if related in memory.topics:
                            G.add_edge(topic_name, related)
                
                # Create node positions
                pos = nx.spring_layout(G, seed=42)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Node colors based on mastery
                node_colors = [G.nodes[node]['mastery'] for node in G.nodes()]
                cmap = LinearSegmentedColormap.from_list('mastery', ['#FF9800', '#4CAF50'])
                
                # Draw nodes and edges
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, cmap=cmap, alpha=0.8, ax=ax)
                nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
                cbar.set_label('Mastery Level')
                
                # Set background color
                ax.set_facecolor('#f5f5f5')
                
                # Remove axis
                ax.axis('off')
                
                # Add title
                plt.title("Topic Relationship Network", fontsize=16)
                
                # Display graph
                st.pyplot(fig)
                
                # Edit topic
                st.markdown("<h2 class='section-header'>Edit Topic</h2>", unsafe_allow_html=True)
                
                selected_topic = st.selectbox("Select Topic to Edit", list(memory.topics.keys()))
                
                if selected_topic:
                    topic = memory.topics[selected_topic]
                    
                    with st.form("edit_topic"):
                        topic_mastery = st.slider("Mastery Level", 0.0, 1.0, topic.mastery, 0.01)
                        topic_related = st.multiselect(
                            "Related Topics",
                            options=[t for t in memory.topics.keys() if t != selected_topic],
                            default=topic.related_topics
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            update_topic_button = st.form_submit_button("Update Topic")
                        
                        with col2:
                            delete_topic_button = st.form_submit_button("Delete Topic", type="primary")
                        
                        if update_topic_button:
                            memory.topics[selected_topic].mastery = topic_mastery
                            memory.topics[selected_topic].related_topics = topic_related
                            memory.topics[selected_topic].last_updated = datetime.now().isoformat()
                            
                            asyncio.run(memory_store.save_memory(memory))
                            st.success(f"Updated topic {selected_topic}")
                            st.rerun()
                        
                        if delete_topic_button:
                            # Remove from related topics first
                            for t_name, t in memory.topics.items():
                                if selected_topic in t.related_topics:
                                    t.related_topics.remove(selected_topic)
                            
                            # Delete topic
                            del memory.topics[selected_topic]
                            
                            asyncio.run(memory_store.save_memory(memory))
                            st.success(f"Deleted topic {selected_topic}")
                            st.rerun()
        
        # System Prompts page
        elif page == "System Prompts":
            st.markdown(f"<h1 class='main-header'>System Prompts: {selected_user}</h1>", unsafe_allow_html=True)
            
            # Add new system prompt
            with st.expander("Add New System Prompt"):
                with st.form("add_prompt"):
                    new_prompt_description = st.text_input("Description")
                    new_prompt_content = st.text_area("Prompt Content", height=200)
                    new_prompt_topics = st.multiselect(
                        "Related Topics",
                        options=list(memory.topics.keys()),
                        default=[]
                    )
                    new_prompt_effectiveness = st.slider("Effectiveness", 0.0, 1.0, 0.5, 0.01)
                    
                    add_prompt_button = st.form_submit_button("Add System Prompt")
                    
                    if add_prompt_button and new_prompt_content and new_prompt_description:
                        memory.system_prompts.append(SystemPrompt(
                            content=new_prompt_content,
                            description=new_prompt_description,
                            topics=new_prompt_topics,
                            effectiveness=new_prompt_effectiveness,
                            last_used=datetime.now().isoformat()
                        ))
                        
                        asyncio.run(memory_store.save_memory(memory))
                        st.success("Added new system prompt")
                        st.rerun()
            
            # Display system prompts
            st.markdown("<h2 class='section-header'>System Prompts</h2>", unsafe_allow_html=True)
            
            if not memory.system_prompts:
                st.info("No system prompts have been added yet.")
            else:
                # Sort by effectiveness
                sorted_prompts = sorted(memory.system_prompts, key=lambda p: p.effectiveness, reverse=True)
                
                for i, prompt in enumerate(sorted_prompts):
                    with st.expander(f"{prompt.description} (Effectiveness: {prompt.effectiveness:.2f})"):
                        # Display prompt info
                        st.markdown(f"**Last Used**: {datetime.fromisoformat(prompt.last_used).strftime('%Y-%m-%d')}")
                        
                        if prompt.topics:
                            st.markdown(f"**Related Topics**: {', '.join(prompt.topics)}")
                        
                        st.markdown("**Prompt Content**:")
                        st.code(prompt.content)
                        
                        # Edit prompt
                        with st.form(f"edit_prompt_{i}"):
                            prompt_description = st.text_input("Description", prompt.description)
                            prompt_content = st.text_area("Content", prompt.content, height=150)
                            prompt_topics = st.multiselect(
                                "Related Topics",
                                options=list(memory.topics.keys()),
                                default=prompt.topics
                            )
                            prompt_effectiveness = st.slider("Effectiveness", 0.0, 1.0, prompt.effectiveness, 0.01)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                update_prompt_button = st.form_submit_button("Update Prompt")
                            
                            with col2:
                                delete_prompt_button = st.form_submit_button("Delete Prompt", type="primary")
                            
                            if update_prompt_button:
                                memory.system_prompts[i].description = prompt_description
                                memory.system_prompts[i].content = prompt_content
                                memory.system_prompts[i].topics = prompt_topics
                                memory.system_prompts[i].effectiveness = prompt_effectiveness
                                memory.system_prompts[i].last_used = datetime.now().isoformat()
                                
                                asyncio.run(memory_store.save_memory(memory))
                                st.success("Updated system prompt")
                                st.rerun()
                            
                            if delete_prompt_button:
                                memory.system_prompts.pop(i)
                                
                                asyncio.run(memory_store.save_memory(memory))
                                st.success("Deleted system prompt")
                                st.rerun()
        
        # Preferences page
        elif page == "Preferences":
            st.markdown(f"<h1 class='main-header'>Preferences: {selected_user}</h1>", unsafe_allow_html=True)
            
            # Add new preference category
            with st.expander("Add New Preference Category"):
                with st.form("add_preference_category"):
                    new_category = st.text_input("Category Name")
                    
                    add_category_button = st.form_submit_button("Add Category")
                    
                    if add_category_button and new_category:
                        if new_category in memory.preferences:
                            st.error(f"Category {new_category} already exists!")
                        else:
                            memory.preferences[new_category] = Preference(category=new_category)
                            
                            asyncio.run(memory_store.save_memory(memory))
                            st.success(f"Added category {new_category}")
                            st.rerun()
            
            # Display preferences
            if not memory.preferences:
                st.info("No preference categories have been added yet.")
            else:
                # Tabs for each category
                categories = list(memory.preferences.keys())
                tabs = st.tabs(categories)
                
                for i, tab in enumerate(tabs):
                    category = categories[i]
                    preference = memory.preferences[category]
                    
                    with tab:
                        # Add new preference item
                        with st.form(f"add_preference_item_{i}"):
                            new_item = st.text_input("Item Name")
                            new_score = st.slider("Preference Score", 0.0, 1.0, 0.8, 0.01)
                            
                            add_item_button = st.form_submit_button(f"Add {category} Item")
                            
                            if add_item_button and new_item:
                                preference.items[new_item] = new_score
                                
                                asyncio.run(memory_store.save_memory(memory))
                                st.success(f"Added {new_item} to {category}")
                                st.rerun()
                        
                        # Display items
                        st.markdown(f"<h2 class='section-header'>{category} Items</h2>", unsafe_allow_html=True)
                        
                        if not preference.items:
                            st.info(f"No {category} preferences have been added yet.")
                        else:
                            # Create dataframe
                            items_data = []
                            for item, score in preference.items.items():
                                items_data.append({
                                    'Item': item,
                                    'Score': score
                                })
                            
                            items_df = pd.DataFrame(items_data)
                            items_df = items_df.sort_values('Score', ascending=False)
                            
                            # Display table
                            st.dataframe(items_df, use_container_width=True)
                            
                            # Bar chart
                            fig = px.bar(
                                items_df, x='Item', y='Score',
                                color='Score', color_continuous_scale=['#FF9800', '#4CAF50'],
                                title=f'{category} Preferences'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Edit items
                            st.markdown(f"<h2 class='section-header'>Edit {category} Item</h2>", unsafe_allow_html=True)
                            
                            selected_item = st.selectbox(f"Select {category} Item to Edit", list(preference.items.keys()), key=f"select_{category}")
                            
                            if selected_item:
                                with st.form(f"edit_preference_{category}_{selected_item}"):
                                    item_score = st.slider("Preference Score", 0.0, 1.0, preference.items[selected_item], 0.01)
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        update_item_button = st.form_submit_button("Update Item")
                                    
                                    with col2:
                                        delete_item_button = st.form_submit_button("Delete Item", type="primary")
                                    
                                    if update_item_button:
                                        preference.items[selected_item] = item_score
                                        
                                        asyncio.run(memory_store.save_memory(memory))
                                        st.success(f"Updated {selected_item}")
                                        st.rerun()
                                    
                                    if delete_item_button:
                                        del preference.items[selected_item]
                                        
                                        asyncio.run(memory_store.save_memory(memory))
                                        st.success(f"Deleted {selected_item}")
                                        st.rerun()
                        
                        # Delete category
                        with st.expander("⚠️ Delete Category"):
                            delete_category = st.button(f"Delete {category} Category", type="primary", key=f"delete_{category}")
                            
                            if delete_category:
                                del memory.preferences[category]
                                
                                asyncio.run(memory_store.save_memory(memory))
                                st.success(f"Deleted category {category}")
                                st.rerun()
        
        # Conversation Style page
        elif page == "Conversation Style":
            st.markdown(f"<h1 class='main-header'>Conversation Style: {selected_user}</h1>", unsafe_allow_html=True)
            
            # Display current conversation style
            st.markdown("<h2 class='section-header'>Current Conversation Style</h2>", unsafe_allow_html=True)
            st.markdown(memory.conversation_style.get_descriptive_text())
            
            # Style visualization
            cs = memory.conversation_style
            
            # Display current values
            col1, col2, col3 = st.columns(3)
            
            with col1:
                formality = "Very formal" if cs.formality > 0.8 else "Formal" if cs.formality > 0.6 else "Neutral" if cs.formality > 0.4 else "Casual" if cs.formality > 0.2 else "Very casual"
                st.metric("Formality", f"{formality} ({cs.formality:.2f})")
                st.progress(cs.formality)
            
            with col2:
                verbosity = "Very detailed" if cs.verbosity > 0.8 else "Detailed" if cs.verbosity > 0.6 else "Balanced" if cs.verbosity > 0.4 else "Concise" if cs.verbosity > 0.2 else "Very concise"
                st.metric("Verbosity", f"{verbosity} ({cs.verbosity:.2f})")
                st.progress(cs.verbosity)
            
            with col3:
                technical = "Very technical" if cs.technical_level > 0.8 else "Technical" if cs.technical_level > 0.6 else "Moderate" if cs.technical_level > 0.4 else "Simple" if cs.technical_level > 0.2 else "Very simple"
                st.metric("Technical Level", f"{technical} ({cs.technical_level:.2f})")
                st.progress(cs.technical_level)
            
            # Update conversation style
            st.markdown("<h2 class='section-header'>Update Conversation Style</h2>", unsafe_allow_html=True)
            
            with st.form("update_conversation_style"):
                formality = st.slider("Formality (Casual to Formal)", 0.0, 1.0, cs.formality, 0.01)
                verbosity = st.slider("Verbosity (Concise to Detailed)", 0.0, 1.0, cs.verbosity, 0.01)
                technical_level = st.slider("Technical Level (Simple to Technical)", 0.0, 1.0, cs.technical_level, 0.01)
                
                update_cs_button = st.form_submit_button("Update Conversation Style")
                
                if update_cs_button:
                    memory.conversation_style = ConversationStyle(
                        formality=formality,
                        verbosity=verbosity,
                        technical_level=technical_level
                    )
                    
                    asyncio.run(memory_store.save_memory(memory))
                    st.success("Conversation style updated!")
                    st.rerun()
            
            # Examples
            st.markdown("<h2 class='section-header'>Conversation Style Examples</h2>", unsafe_allow_html=True)
            
            examples = {
                "casual_concise_simple": "Hey there! Here's a quick tip for you. AI models use patterns from data to make predictions. Simple enough, right?",
                "casual_concise_technical": "Hey there! Here's the gist: Large language models implement transformer architectures with attention mechanisms to predict token probabilities based on context. Pretty cool!",
                "casual_detailed_simple": "Hey there! I wanted to tell you about how AI works. It's like having a really smart pattern recognizer. The AI sees lots of examples and learns what usually comes next. It doesn't actually understand things like humans do, but it can make really good guesses based on patterns it's seen before. When you ask it a question, it tries to figure out what response would make the most sense, kind of like predicting what words would naturally follow your question!",
                "casual_detailed_technical": "Hey there! Thought I'd explain how large language models work. They're built on transformer architectures that utilize self-attention mechanisms to weigh the importance of different tokens in the input sequence. During pre-training, they optimize for next-token prediction across enormous datasets, effectively learning distributional statistics of language. The model doesn't have explicit symbolic reasoning but encodes implicit statistical relationships that approximate semantic understanding. When generating text, the model samples from probability distributions over the vocabulary conditioned on the preceding tokens!",
                "formal_concise_simple": "The artificial intelligence system functions by recognizing patterns in data and making predictions based on prior examples. This process enables it to generate appropriate responses to queries.",
                "formal_concise_technical": "Large language models implement transformer architectures with multi-head attention mechanisms to compute contextualized token embeddings. The models utilize these representations to derive conditional probability distributions over vocabulary tokens during inference.",
                "formal_detailed_simple": "Artificial intelligence systems operate by analyzing patterns within extensive datasets. These systems identify correlations and relationships that enable them to make predictions based on new inputs. When presented with a query, the system evaluates the input against its trained parameters and generates an appropriate response. It is important to note that these systems do not possess understanding in the human sense, but rather function through statistical pattern matching and prediction algorithms that approximate human-like responses.",
                "formal_detailed_technical": "Large language models are computational systems based on transformer architectures that utilize multi-head self-attention mechanisms to process token sequences. During pre-training, these models are optimized to predict tokens in a sequence given preceding context, thereby learning distributional statistics across extensive corpora. The models encode contextual information through attention-weighted vector representations that capture semantic and syntactic relationships between tokens. When generating text, the model samples from conditional probability distributions over the vocabulary, producing coherent sequences that adhere to learned linguistic patterns and factual associations present in the training data.",
            }
            
            # Determine closest example
            def get_closest_example():
                formality_label = "formal" if cs.formality > 0.5 else "casual"
                verbosity_label = "detailed" if cs.verbosity > 0.5 else "concise"
                technical_label = "technical" if cs.technical_level > 0.5 else "simple"
                
                return f"{formality_label}_{verbosity_label}_{technical_label}"
            
            closest = get_closest_example()
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**Predicted Response Style Based on Current Settings:**")
            st.markdown(examples[closest])
            st.markdown("</div>", unsafe_allow_html=True)
            
            with st.expander("View All Style Examples"):
                for style, example in examples.items():
                    parts = style.split('_')
                    style_name = f"{parts[0].title()}, {parts[1].title()}, {parts[2].title()}"
                    
                    st.markdown(f"**{style_name}**")
                    st.markdown(f"{example}")
                    st.markdown("---")
        
        # Import/Export page
        elif page == "Import/Export":
            st.markdown(f"<h1 class='main-header'>Import/Export: {selected_user}</h1>", unsafe_allow_html=True)
            
            # Export memory
            st.markdown("<h2 class='section-header'>Export Memory</h2>", unsafe_allow_html=True)
            
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "MentorSync", "LLM-friendly"],
                help="JSON = Full memory export, MentorSync = Compatible with MentorSync AI, LLM-friendly = Text format for LLMs"
            )
            
            export_button = st.button("Export Memory")
            
            if export_button:
                if export_format == "JSON":
                    export_data = json.dumps(memory.to_dict(), indent=2)
                    mime_type = "application/json"
                    filename = f"{selected_user}_memory.json"
                
                elif export_format == "MentorSync":
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
                    export_data = json.dumps(mentorsync_data, indent=2)
                    mime_type = "application/json"
                    filename = f"{selected_user}_mentorsync.json"
                
                else:  # LLM-friendly
                    export_data = memory.get_llm_friendly_representation()
                    mime_type = "text/plain"
                    filename = f"{selected_user}_memory.txt"
                
                # Create download button
                st.download_button(
                    label="Download Export",
                    data=export_data,
                    file_name=filename,
                    mime=mime_type
                )
                
                # Also display for copy-paste
                st.code(export_data)
            
            # Import memory
            st.markdown("<h2 class='section-header'>Import Memory</h2>", unsafe_allow_html=True)
            
            import_format = st.selectbox(
                "Import Format",
                ["JSON", "MentorSync"],
                help="JSON = Full memory import, MentorSync = Import from MentorSync AI"
            )
            
            import_data = st.text_area("Paste Import Data", height=300)
            
            import_button = st.button("Import Memory")
            
            if import_button and import_data:
                try:
                    if import_format == "JSON":
                        # Parse JSON data
                        memory_data = json.loads(import_data)
                        
                        # Ensure user_id is correct
                        memory_data["user_id"] = selected_user
                        
                        # Create memory object
                        imported_memory = UserMemory.from_dict(memory_data)
                        
                        # Save memory
                        asyncio.run(memory_store.save_memory(imported_memory))
                        
                        st.success(f"Successfully imported memory for user {selected_user} from JSON format.")
                        st.rerun()
                    
                    elif import_format == "MentorSync":
                        # Parse MentorSync data
                        mentorsync_data = json.loads(import_data)
                        
                        # Create new memory
                        imported_memory = UserMemory(user_id=selected_user)
                        
                        # Import learning style
                        if "learning_style" in mentorsync_data:
                            ls_data = mentorsync_data["learning_style"]
                            imported_memory.learning_style = LearningStyle(
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
                                imported_memory.topics[topic.name] = topic
                        
                        # Import preferences
                        if "preferences" in mentorsync_data:
                            for category, items in mentorsync_data["preferences"].items():
                                pref = Preference(category=category)
                                for item_data in items:
                                    pref.items[item_data["item"]] = item_data["score"]
                                imported_memory.preferences[category] = pref
                        
                        # Save memory
                        asyncio.run(memory_store.save_memory(imported_memory))
                        
                        st.success(f"Successfully imported memory for user {selected_user} from MentorSync format.")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error importing memory: {str(e)}")
            
            # Claude Desktop Integration
            st.markdown("<h2 class='section-header'>Claude Desktop Integration</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            To use Samantha with Claude Desktop:
            
            1. Make sure the MCP server is running with:
               ```
               python samantha.py
               ```
            
            2. Install the MCP server in Claude Desktop by editing the config file and adding:
               ```json
               "Samantha Memory Manager": {
                 "command": "python",
                 "args": [
                   "/path/to/samantha.py"
                 ]
               }
               ```
               
               Or for Docker:
               ```json
               "Samantha Memory Manager": {
                 "command": "docker",
                 "args": [
                   "exec",
                   "-i",
                   "samantha",
                   "python",
                   "-c",
                   "from mcp.server.stdio import stdio_server; import asyncio; from samantha import mcp; asyncio.run(mcp.run_stdio_async())"
                 ]
               }
               ```
            
            3. Use these resources in Claude:
               ```
               memory://{username}/profile
               memory://{username}/learning_style
               memory://{username}/topics
               memory://{username}/system_prompts
               memory://{username}/preferences/{category}
               memory://{username}/conversation_style
               ```
            
            4. Use these tools in Claude:
               ```
               extract_memory
               add_system_prompt
               update_system_prompt_effectiveness
               update_topic_mastery
               relate_topics
               update_learning_style
               ```
            """)
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")

if __name__ == "__main__":
    # This allows running dashboard.py directly with streamlit
    pass