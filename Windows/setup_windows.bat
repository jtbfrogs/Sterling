@echo off
REM ============================================================
REM  Sterling — Windows Setup Script
REM  RTX 3060 / 16 GB RAM
REM
REM  Run this from the Sterling project folder:
REM    cd C:\path\to\sterling\Windows
REM    setup_windows.bat
REM
REM  Prerequisites (must be installed BEFORE running this):
REM    - Python 3.11  (python.org)
REM    - CUDA Toolkit 12.x  (developer.nvidia.com/cuda-downloads)
REM    - Ollama for Windows  (ollama.ai)
REM    - Microsoft C++ Build Tools (for dlib/face_recognition)
REM ============================================================

echo.
echo  ============================================
echo   Sterling Windows Setup
echo  ============================================
echo.

REM Check Python
python --version 2>NUL
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11 from python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo [1/6] Creating virtual environment...
python -m venv ster
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo       Done.

REM Activate
echo [2/6] Activating virtual environment...
call ster\Scripts\activate.bat

REM Upgrade pip
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 12.1
echo [4/6] Installing PyTorch with CUDA 12.1 support...
echo       This downloads ~2.5 GB — may take several minutes.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [WARNING] PyTorch CUDA install failed. Trying CPU version...
    pip install torch torchvision torchaudio
)

REM Install requirements
echo [5/6] Installing Sterling requirements...
pip install edge-tts pyttsx3 pygame pyaudio numpy pyyaml requests
pip install faster-whisper>=1.0.0
pip install ultralytics opencv-python
pip install ollama spotipy
if errorlevel 1 (
    echo [WARNING] Some packages may have failed. Check output above.
    echo           PyAudio often needs manual install — see WINDOWS_SETUP.md
)

REM Pull Ollama models
echo [6/6] Pulling Ollama models...
echo       Pulling qwen2.5:7b (LLM) — ~4.4 GB download...
ollama pull qwen2.5:7b
echo       Pulling whisper is handled by faster-whisper automatically.

echo.
echo  ============================================
echo   Setup complete!
echo  ============================================
echo.
echo  Next steps:
echo    1. Edit config.yaml (weather location, workspace path, etc.)
echo    2. Run Sterling: python main.py
echo    3. Say "Sterling" or "Hey Sterling" to wake it up
echo.
echo  See WINDOWS_SETUP.md for full instructions and troubleshooting.
echo.
pause
