#!/bin/bash

# Script to set up Samantha MCP Server

# Create necessary directories
mkdir -p data

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build the Docker image
echo "Building Samantha Docker image..."
docker-compose build

# Start the containers
echo "Starting Samantha services..."
docker-compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Check if the services are running
if docker ps | grep -q samantha; then
    echo "Samantha services are running!"
    echo "- MCP Server: http://localhost:5001"
    echo "- Dashboard: http://localhost:8501"
    
    # Add instructions for Claude Desktop
    echo ""
    echo "To connect with Claude Desktop, add this to your Claude Desktop config:"
    echo ""
    echo '"Samantha Memory Manager": {'
    echo '  "command": "docker",'
    echo '  "args": ['
    echo '    "exec",'
    echo '    "-i",'
    echo '    "samantha",'
    echo '    "python",'
    echo '    "-c",'
    echo '    "from mcp.server.stdio import stdio_server; import asyncio; from samantha import mcp; asyncio.run(mcp.run_stdio_async())"'
    echo '  ]'
    echo '}'
else
    echo "Failed to start Samantha services. Check the logs with 'docker-compose logs'."
    exit 1
fi