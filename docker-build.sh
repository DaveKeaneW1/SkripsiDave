#!/bin/bash

# Build Docker containers for Film Clustering Analysis

set -e

echo "🐳 Building Docker images..."
docker-compose build

echo "✅ Build complete!"
echo ""
echo "📋 To start the application, run:"
echo "   docker-compose up"
echo ""
echo "🌐 Access the application at:"
echo "   Frontend (Streamlit): http://localhost:8501"
echo "   Backend API: http://localhost:5000"
echo ""
echo "🛑 To stop the application, run:"
echo "   docker-compose down"
