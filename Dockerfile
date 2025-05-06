# Dockerfile for Samantha MCP Server
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model explicitly
RUN python -m spacy download en_core_web_md

# Create data directory with proper permissions
RUN mkdir -p /app/data && chmod 777 /app/data

# Copy application code
COPY samantha.py .
COPY dashboard.py .

# Expose ports for MCP and Streamlit
EXPOSE 5000 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MCP_PORT=5000
ENV STREAMLIT_PORT=8501
ENV SAMANTHA_DATA_DIR=/app/data

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "mcp" ]; then\n\
    python samantha.py\n\
elif [ "$1" = "dashboard" ]; then\n\
    streamlit run dashboard.py --server.port=$STREAMLIT_PORT --server.address=0.0.0.0\n\
elif [ "$1" = "all" ]; then\n\
    python samantha.py & streamlit run dashboard.py --server.port=$STREAMLIT_PORT --server.address=0.0.0.0\n\
    wait -n\n\
    exit $?\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["all"]