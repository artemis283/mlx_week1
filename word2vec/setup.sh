#!/bin/bash

# Create necessary directories
mkdir -p static models

# Copy frontend HTML to static directory
if [ -f "index.html" ]; then
    cp index.html static/
    echo "✓ Copied index.html to static directory"
else
    echo "⚠ index.html not found in current directory"
fi

# Check for model files
if [ -f "skipgram_model.pth" ]; then
    cp skipgram_model.pth models/
    echo "✓ Copied skipgram_model.pth to models directory"
else
    echo "⚠ skipgram_model.pth not found in current directory"
fi

if [ -f "hn_predictor_complete.pth" ]; then
    cp hn_predictor_complete.pth models/
    echo "✓ Copied hn_predictor_complete.pth to models directory"
else
    echo "⚠ hn_predictor_complete.pth not found in current directory"
fi

# Build the Docker container
echo "Building Docker container..."
docker-compose build

echo ""
echo "Setup complete! You can now run the application with:"
echo "docker-compose up -d"
echo ""
echo "Then access the web interface at: http://localhost:8000"