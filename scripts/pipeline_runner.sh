#!/bin/bash

# Make this file executable: chmod +x ./scripts/pipeline_runner.sh
# Run it from project root: ./scripts/pipeline_runner.sh

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

# === Prepare logging ===
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="logs"
log_file="$log_dir/pipeline_$timestamp.log"
mkdir -p "$log_dir"

# === Activate virtual environment ===
print_blue "Activating virtual environment..."
source .venv/Scripts/activate || print_error "Failed to activate virtual environment. Is it created?"

# === Run the pipeline runner script ===
print_blue "Running pipeline notebooks with nbclient..."
PYTHONPATH=. python scripts/pipeline_runner.py | tee "$log_file"

# === Done ===
print_green "Pipeline execution completed successfully. Log saved to $log_file"
