# Samantha: MCP Memory Manager
  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](https://www.docker.com/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-orange.svg)](https://modelcontextprotocol.io/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

</div>

## 📚 Overview

Samantha is a sophisticated MCP (Model Context Protocol) server that dynamically extracts and manages user memory from LLM conversations. This memory can be shared across different LLMs, allowing for consistent personalization even when switching between models.

Samantha analyzes conversations, extracting information about learning styles, topics of interest, knowledge levels, and personal preferences, creating a comprehensive user profile that LLMs can use to provide tailored responses.

## 🌟 Key Features

- **Dynamic Memory Extraction** - Automatically identifies learning styles, topics, and preferences from conversations
- **Learning Style Analysis** - Uses the VARK model (Visual, Auditory, Reading/Writing, Kinesthetic) with detailed recommendations
- **Topic Relationship Mapping** - Creates connection graphs between knowledge areas
- **Personal Preference Tracking** - Records and ranks preferences in categories like movies, books, and music
- **System Prompt Memory** - Stores effective system prompts with effectiveness ratings
- **Interactive Dashboard** - Visualizes and manages memory profiles
- **Multi-format Export/Import** - Supports JSON, MentorSync, and LLM-friendly formats
- **Docker Integration** - Runs in isolated containers with persistent storage

## 🔧 System Requirements

- Docker and Docker Compose
- 512MB RAM minimum (recommend 1GB+)
- 1GB disk space
- For direct installation: Python 3.11+

## 📋 Memory Layers

Samantha uses a multi-layered memory architecture:

1. **Identity Layer** - Core user identity
2. **Learning Style Layer** - VARK profile with weighted representations
3. **Knowledge Layer** - Topics with mastery levels and relationships
4. **Preference Layer** - Category-based preference tracking with scoring
5. **Interaction Layer** - Conversation style preferences
6. **System Layer** - Effective system prompts with effectiveness scores

## 💾 Docker Memory Requirements

- **Base Image**: ~500MB
- **Runtime Memory**: 300-500MB (depends on number of active users)
- **Storage**: Minimal (~1KB per user profile)
- **Total**: ~1GB recommended allocation

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/samantha-mcp.git
cd samantha-mcp

# Start the services
docker-compose up -d

# Access the dashboard
open http://localhost:8501
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/samantha-mcp.git
cd samantha-mcp

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_md

# Run the server
python samantha.py

# In a separate terminal, run the dashboard
streamlit run dashboard.py
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Desktop                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                 ┌──────────▼─────────────┐
                 │    MCP Protocol        │
                 └──────────┬─────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      Samantha Server                        │
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │   Memory    │    │   Memory    │    │    MCP      │    │
│   │  Extractor  │◄───┤    Store    │◄───┤  Interface  │    │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    │
│          │                  │                  │           │
│   ┌──────▼──────────────────▼──────────────────▼──────┐    │
│   │                  Data Models                       │    │
│   │   (Learning Style, Topics, Preferences, etc.)      │    │
│   └───────────────────────┬───────────────────────────┘    │
│                           │                                 │
│                     ┌─────▼────┐                            │
│                     │   JSON   │                            │
│                     │ Storage  │                            │
│                     └──────────┘                            │
└─────────────────────────────────────────────────────────────┘
                            │
                  ┌─────────▼────────┐
                  │ Streamlit        │
                  │   Dashboard      │
                  └──────────────────┘
```

## 📦 File Structure

```
samantha-mcp/
├── samantha.py         # Core MCP server implementation
├── dashboard.py        # Streamlit dashboard
├── Dockerfile          # Container definition
├── docker-compose.yml  # Container orchestration
├── requirements.txt    # Python dependencies
├── setup.sh            # Setup script
├── data/               # Persistent storage directory
└── docs/               # Documentation
```

## 🔌 MCP Integration

### Resources

| Resource Path | Description |
|--------------|-------------|
| `memory://{user_id}/profile` | Complete user memory profile |
| `memory://{user_id}/learning_style` | VARK learning style with recommendations |
| `memory://{user_id}/topics` | Knowledge topics and mastery levels |
| `memory://{user_id}/system_prompts` | Effective system prompts |
| `memory://{user_id}/preferences/{category}` | Specific preference category |
| `memory://{user_id}/conversation_style` | Communication style preferences |

### Tools

| Tool Name | Description |
|-----------|-------------|
| `extract_memory` | Extract memory from a conversation |
| `add_system_prompt` | Add a new system prompt |
| `update_system_prompt_effectiveness` | Update system prompt effectiveness rating |
| `update_topic_mastery` | Update topic mastery level |
| `relate_topics` | Establish relationships between topics |
| `update_learning_style` | Manually update learning style |
| `export_memory` | Export user memory in specified format |
| `import_memory` | Import user memory from specified format |
| `delete_user_memory` | Delete a user's memory |

## 🔄 Memory Extraction Process

The memory extraction process involves:

1. **Conversation Analysis**: Processing text for learning style indicators, topic mentions, and preferences
2. **Pattern Matching**: Identifying learning patterns using NLP techniques
3. **Weighted Updates**: Incorporating new insights with a 30/70 weighted model (new/existing)
4. **Profile Construction**: Building a comprehensive user profile
5. **Persistent Storage**: Saving memory in JSON format for retrieval

## 🎮 Dashboard Features

The interactive Streamlit dashboard provides:

- **User Profile Overview**: Summary of all memory components
- **Learning Style Management**: Visual editors for VARK profiles
- **Topic Network Visualization**: Interactive graph of knowledge areas
- **System Prompt Library**: Create and rate effective prompts
- **Preference Management**: Track and organize user likes
- **Import/Export Tools**: Data backup and transfer functionality

## 🔧 Production Deployment

For production deployment, consider:

### Docker Swarm/Kubernetes

```bash
# Deploy to Docker Swarm
docker stack deploy -c docker-compose.yml samantha

# Or with Kubernetes
kubectl apply -f k8s-deployment.yml
```

### Multiple Instances with Load Balancing

```yaml
# Example docker-compose.override.yml
services:
  samantha:
    deploy:
      replicas: 3
  
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - samantha
```

### Data Volume Management

The `data` directory contains all user profiles and should be backed up regularly:

```bash
# Backup data
tar -czf samantha-data-backup.tar.gz data/

# Restore data
tar -xzf samantha-data-backup.tar.gz
```

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Port conflicts | Modify port mappings in docker-compose.yml |
| Memory errors | Increase container memory limits |
| Missing dependencies | Ensure all requirements are installed |
| Data persistence issues | Check volume mounts and permissions |
| spaCy model errors | Manually download with `python -m spacy download en_core_web_md` |

### Logs

```bash
# View container logs
docker-compose logs -f

# Check specific container
docker logs samantha
```

## 📊 Performance Monitoring

For production deployments, add monitoring:

```bash
# With Prometheus/Grafana
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📜 License

Samantha is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- The Model Context Protocol team for the MCP specification
- Streamlit for the dashboard framework
- spaCy for NLP capabilities