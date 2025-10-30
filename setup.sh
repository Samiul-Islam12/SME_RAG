#!/bin/bash
# Setup script for PDF RAG System

echo "=========================================="
echo "PDF RAG System - Setup Script"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "⚠️  Ollama is not installed!"
    echo "Please install Ollama from: https://ollama.com/download"
    echo ""
    echo "Installation commands:"
    echo "  Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Windows: Download from https://ollama.com/download"
    exit 1
fi

echo "✓ Ollama found: $(ollama --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p pdfs
mkdir -p plots

# Check if Ollama models are installed
echo ""
echo "Checking Ollama models..."
echo ""

models=("mistral" "qwen2.5" "llama3")
missing_models=()

for model in "${models[@]}"; do
    if ollama list | grep -q "$model"; then
        echo "✓ $model is installed"
    else
        echo "⚠️  $model is NOT installed"
        missing_models+=("$model")
    fi
done

if [ ${#missing_models[@]} -ne 0 ]; then
    echo ""
    echo "Installing missing models..."
    for model in "${missing_models[@]}"; do
        echo "Installing $model..."
        ollama pull "$model"
    done
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Add PDF files to the 'pdfs/' directory"
echo "  2. Activate virtual environment: source venv/bin/activate"
echo "  3. Run evaluation: python run_evaluation.py"
echo ""
echo "For examples: python example_usage.py"
echo "For documentation: cat README.md"
echo ""
echo "=========================================="
