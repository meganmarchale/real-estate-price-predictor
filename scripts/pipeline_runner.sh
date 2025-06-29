#!/bin/bash

# Make this file executable: chmod +x pipeline_runner.sh
# Run it with: ./scripts/pipeline_runner.sh && ./scripts/pipeline_runner.sh
# From root directory:

clear

# === Define color codes ===
BLUE_BG="\033[44m"
GREEN_BG="\033[42m"
RED_BG="\033[41m"
WHITE_TEXT="\033[97m"
RESET="\033[0m"

# === Define print helpers ===
print_blue() {
    echo ""
    echo -e "${BLUE_BG}${WHITE_TEXT}>>> $1${RESET}"
    echo ""
}

print_green() {
    echo ""
    echo -e "${GREEN_BG}${WHITE_TEXT}>>> $1${RESET}"
    echo ""
}

print_error() {
    echo ""
    echo -e "${RED_BG}${WHITE_TEXT}>>> ERROR: $1${RESET}"
    echo ""
    exit 1
}

# === Activate virtual environment ===
print_blue "Activating virtual environment..."
source .venv/Scripts/activate || print_error "Failed to activate virtual environment. Is it created?"

# === Run the pipeline runner script ===
print_blue "Running pipeline (executing notebooks via Papermill)..."
PYTHONPATH=. python scripts/pipeline_runner.py || print_error "Pipeline execution failed."

# === Done ===
print_green "Pipeline execution completed successfully."
