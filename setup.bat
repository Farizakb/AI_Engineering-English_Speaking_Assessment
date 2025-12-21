@echo off
REM Quick setup script for Windows
REM English Speaking Assessment MVP

echo.
echo 🚀 English Speaking Assessment - MVP Setup
echo ==========================================
echo.

REM Check for .env file
if not exist ".env" (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ✓ .env created. Please edit and add your OpenAI API key
    echo.
)

REM Check if Docker is available
where docker-compose >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo 🐳 Docker Compose detected
    echo.
    echo Starting with Docker Compose...
    echo   - Frontend: http://localhost:3000
    echo   - Backend: http://localhost:8000
    echo.
    docker-compose up --build
) else (
    echo ⚙️  Docker not found. Setting up local development environment...
    echo.
    
    echo 📦 Setting up backend...
    cd backend
    
    if not exist "venv" (
        python -m venv venv
        echo ✓ Virtual environment created
    )
    
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    echo ✓ Backend dependencies installed
    
    cd ..
    
    echo 📦 Setting up frontend...
    cd frontend
    
    if not exist "node_modules" (
        call npm install
        echo ✓ Frontend dependencies installed
    )
    
    cd ..
    
    echo.
    echo ✅ Setup complete!
    echo.
    echo 📋 To run locally ^(2 terminals^):
    echo   Terminal 1 ^(Backend^):
    echo     cd backend
    echo     venv\Scripts\activate.bat
    echo     uvicorn main:app --reload
    echo.
    echo   Terminal 2 ^(Frontend^):
    echo     cd frontend
    echo     set REACT_APP_API_URL=http://localhost:8000
    echo     npm start
    echo.
    echo 🌐 Then open: http://localhost:3000
)
