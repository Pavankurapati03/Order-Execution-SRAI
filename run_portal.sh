#!/usr/bin/env bash
# ==============================================================================
# Quantellix Setup & Run Script for macOS & Linux
# ==============================================================================

echo "================================================================="
echo "  Quantellix Autonomous Order Fulfillment & AI Supply Chain Platform"
echo "================================================================="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "[INFO] Installing / checking dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Ensure .env exists
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "[INFO] Creating .env from .env.example..."
    cp .env.example .env
fi

echo ""
echo "================================================================="
echo "  Portal ready! Starting server at http://127.0.0.1:5000"
echo "================================================================="
echo ""

python3 manage.py runserver 127.0.0.1:5000
