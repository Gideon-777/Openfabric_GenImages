#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PROJECT_DIR=$(pwd)
HUNYUAN_REPO="https://github.com/Tencent/Hunyuan3D-2.git"
HUNYUAN_DIR="Hunyuan3D-2"
PYTHON_CMD="python3" # Or just "python" if that's your command
PIP_CMD="pip3"       # Or just "pip"

# --- Helper Functions ---
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo_step() {
    echo "----------------------------------------"
    echo "STEP: $1"
    echo "----------------------------------------"
}

# --- Prerequisite Checks ---
echo_step "Checking prerequisites..."
if ! command_exists git; then
    echo "Error: git is not installed. Please install git and try again."
    exit 1
fi
if ! command_exists $PYTHON_CMD; then
    echo "Error: $PYTHON_CMD is not installed or not in PATH. Please install Python 3 and try again."
    exit 1
fi
if ! command_exists $PIP_CMD; then
    echo "Error: $PIP_CMD is not installed or not in PATH. Please install pip for Python 3 and try again."
    exit 1
fi
echo "Prerequisites met."

# --- Setup Virtual Environment (Recommended) ---
echo_step "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "Virtual environment 'venv' created."
else
    echo "Virtual environment 'venv' already exists."
fi
# Activate virtual environment - Note: Activation needs to be done manually in the shell
# or sourced before running subsequent commands outside this script.
# source venv/bin/activate # This line won't affect the parent shell
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "Using pip from virtual environment for subsequent steps..."
PIP_CMD="venv/bin/pip" # Point pip to the venv

# --- Clone and Setup Hunyuan3D-2 ---
echo_step "Cloning and setting up Hunyuan3D-2..."
if [ ! -d "$HUNYUAN_DIR" ]; then
    echo "Cloning Hunyuan3D-2 repository..."
    git clone "$HUNYUAN_REPO" "$HUNYUAN_DIR"
else
    echo "Directory '$HUNYUAN_DIR' already exists. Skipping clone."
    # Optional: Add logic here to pull latest changes if needed
    # cd "$HUNYUAN_DIR"
    # git pull
    # cd "$PROJECT_DIR"
fi

echo "Navigating into $HUNYUAN_DIR..."
cd "$HUNYUAN_DIR"

echo "Installing Hunyuan3D-2 dependencies..."
# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    "$PIP_CMD" install -r requirements.txt
else
    echo "Warning: requirements.txt not found in $HUNYUAN_DIR. Skipping dependency installation."
fi

echo "Installing Hunyuan3D-2 package..."
# Check if setup.py exists for editable install
if [ -f "setup.py" ]; then
    "$PIP_CMD" install -e .
else
    echo "Warning: setup.py not found in $HUNYUAN_DIR. Skipping editable install."
fi

echo "Navigating back to project root ($PROJECT_DIR)..."
cd "$PROJECT_DIR"

# --- Install Main Application Dependencies ---
echo_step "Installing main application dependencies..."
# IMPORTANT: Ensure you have a requirements.txt in your project root (/home/usama/ai-test/)
if [ -f "requirements.txt" ]; then
    "$PIP_CMD" install -r requirements.txt
else
    echo "Warning: Main requirements.txt not found in $PROJECT_DIR."
    echo "Please create a requirements.txt file with your project's dependencies (e.g., gradio, fastapi, uvicorn, Pillow, torch, transformers, accelerate, bitsandbytes, openfabric-pysdk, etc.)"
fi

# --- Final Instructions ---
echo_step "Setup Complete!"
echo ""
echo "Next Steps:"
echo "1. Activate the virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "2. Ensure your local LLM (e.g., google/gemma-2-2b-it) is downloaded and accessible."
echo "   (You might need to configure paths in the application code or environment variables)."
echo ""
echo "3. Ensure the Openfabric Core is running and you are logged in."
echo ""
echo "4. Run the Gradio application:"
echo "   $PYTHON_CMD app/run_gradio.py"
echo ""
echo "5. Access the application in your browser (usually http://localhost:7860 or as specified by Gradio)."
echo ""
echo "----------------------------------------"

exit 0
