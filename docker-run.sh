#!/bin/bash

# Run Docker containers for Film Clustering Analysis

set -e

echo "🚀 Starting Film Clustering Analysis Application..."
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build and start containers
echo "🐳 Building and starting containers..."
docker-compose up --build

echo ""
echo "✅ Application started!"
echo ""
echo "🌐 Access the application at:"
echo "   Frontend (Streamlit): http://localhost:8501"
echo "   Backend API: http://localhost:5000"
echo ""
echo "💡 Tips:"
echo "   - Press Ctrl+C to stop the application"
echo "   - Use 'docker-compose logs -f' to view logs"
echo "   - Use 'docker-compose down' to stop and remove containers"
