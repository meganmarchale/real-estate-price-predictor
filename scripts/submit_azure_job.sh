#!/bin/bash

# Make this file executable: chmod +x scripts/submit_azure_job.sh
# Run it with: ./scripts/submit_azure_job.sh
# From root directory: chmod +x scripts/submit_azure_job.sh && ./scripts/submit_azure_job.sh

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
source .venv/Scripts/activate || print_error "Failed to activate virtual environment. Make sure it exists."

# === Submit Azure ML Job ===
print_blue "Submitting Azure ML job..."
PYTHONPATH=. python scripts/submit_azure_job.py || print_error "Failed to submit job to Azure ML."

# === Done ===
print_green "Azure ML job submitted successfully."
