#!/bin/bash

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Create data directories if they don't exist
mkdir -p datastore/images
mkdir -p datastore/models

# Install dependencies
echo "Installing dependencies..."
poetry install

# Create a default tokens.json if it doesn't exist
if [ ! -f datastore/tokens.json ]; then
    echo "Creating default tokens.json..."
    echo '{
        "api_key": "YOUR_API_KEY_HERE"
    }' > datastore/tokens.json
    echo "Please edit datastore/tokens.json and add your Openfabric API key"
fi

# Run the application
echo "Starting Creative AI Engine..."
poetry run python3 run_gradio.py