#!/bin/bash
# Quick setup script for English Speaking Assessment MVP

set -e

echo "🚀 English Speaking Assessment - MVP Setup"
echo "=========================================="
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env created. Please edit and add your OpenAI API key:"
    echo "  - Open .env in your editor"
    echo "  - Replace OPENAI_API_KEY value with your actual key"
    echo ""
fi

# Check if Docker is available
if command -v docker-compose &> /dev/null; then
    echo "🐳 Docker Compose detected"
    echo ""
    echo "Starting with Docker Compose..."
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend: http://localhost:8000"
    echo ""
    docker-compose up --build
else
    echo "⚙️  Docker not found. Setting up local development environment..."
    echo ""
    
    # Backend setup
    echo "📦 Setting up backend..."
    cd backend
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✓ Virtual environment created"
    fi
    
    source venv/bin/activate
    pip install -r requirements.txt
    echo "✓ Backend dependencies installed"
    
    cd ..
    
    # Frontend setup
    echo "📦 Setting up frontend..."
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        npm install
        echo "✓ Frontend dependencies installed"
    fi
    
    cd ..
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "📋 To run locally (2 terminals):"
    echo "  Terminal 1 (Backend):"
    echo "    cd backend"
    echo "    source venv/bin/activate"
    echo "    uvicorn main:app --reload"
    echo ""
    echo "  Terminal 2 (Frontend):"
    echo "    cd frontend"
    echo "    REACT_APP_API_URL=http://localhost:8000 npm start"
    echo ""
    echo "🌐 Then open: http://localhost:3000"
fi
