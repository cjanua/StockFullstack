@echo off
REM Windows Native Setup for AMD RX 7800 XT GPU Training
echo ========================================
echo StockFullstack GPU Training Setup
echo ========================================

echo.
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.12 from python.org
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version

echo.
echo Step 2: Creating virtual environment...
if not exist "venv_windows" (
    python -m venv venv_windows
)

echo.
echo Step 3: Activating virtual environment...
call venv_windows\Scripts\activate.bat

echo.
echo Step 4: Installing PyTorch with ROCm support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

echo.
echo Step 5: Installing project dependencies...
pip install -e .
pip install scikit-learn pandas numpy matplotlib seaborn joblib

echo.
echo Step 6: Testing GPU availability...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"

echo.
echo Step 7: Ready to train!
echo To run training:
echo   cd ai
echo   python main.py

pause