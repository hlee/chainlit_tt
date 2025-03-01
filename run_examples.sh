#!/bin/bash

# Run Examples Script for Chainlit RAG Tutorial
# This script helps users run the different examples in this tutorial

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3 to continue.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed. Please install pip to continue.${NC}"
    exit 1
fi

# Function to check if OpenAI API key is set
check_api_key() {
    if grep -q "your_openai_api_key_here" .env; then
        echo -e "${YELLOW}Warning: OpenAI API key is not set in the .env file.${NC}"
        echo -e "${YELLOW}Please update the .env file with your OpenAI API key before running the examples.${NC}"
        echo ""
        read -p "Do you want to enter your OpenAI API key now? (y/n): " choice
        if [[ $choice == "y" || $choice == "Y" ]]; then
            read -p "Enter your OpenAI API key: " api_key
            sed -i "s/your_openai_api_key_here/$api_key/" .env
            echo -e "${GREEN}API key updated in .env file.${NC}"
        else
            echo -e "${YELLOW}Please update the .env file manually before running the examples.${NC}"
        fi
    fi
}

# Function to install dependencies
install_dependencies() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Dependencies installed successfully.${NC}"
    else
        echo -e "${RED}Error installing dependencies. Please check the error message above.${NC}"
        exit 1
    fi
}

# Function to run the basic example
run_basic_example() {
    echo -e "${BLUE}Running basic Chainlit RAG example...${NC}"
    echo -e "${YELLOW}This will start a web server at http://localhost:8000${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the server when you're done.${NC}"
    echo ""
    chainlit run app.py
}

# Function to run the advanced example
run_advanced_example() {
    echo -e "${BLUE}Running advanced Chainlit RAG example...${NC}"
    echo -e "${YELLOW}This will start a web server at http://localhost:8000${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the server when you're done.${NC}"
    echo ""
    chainlit run advanced_rag.py
}

# Function to run the evaluation script
run_evaluation() {
    echo -e "${BLUE}Running RAG evaluation script...${NC}"
    python3 evaluate_rag.py
}

# Function to add a new document
add_document() {
    echo -e "${BLUE}Add a new document to the RAG system${NC}"
    read -p "Enter the path to the document: " doc_path
    if [ -f "$doc_path" ]; then
        python3 add_document.py "$doc_path"
    else
        echo -e "${RED}Error: File not found.${NC}"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo -e "${GREEN}=== Chainlit RAG Tutorial Examples ===${NC}"
    echo "1. Install dependencies"
    echo "2. Run basic Chainlit RAG example"
    echo "3. Run advanced Chainlit RAG example"
    echo "4. Run RAG evaluation script"
    echo "5. Add a new document to the RAG system"
    echo "6. Exit"
    echo ""
    read -p "Enter your choice (1-6): " choice
    
    case $choice in
        1) install_dependencies; show_menu ;;
        2) check_api_key; run_basic_example ;;
        3) check_api_key; run_advanced_example ;;
        4) check_api_key; run_evaluation; show_menu ;;
        5) add_document; show_menu ;;
        6) echo -e "${GREEN}Goodbye!${NC}"; exit 0 ;;
        *) echo -e "${RED}Invalid choice. Please try again.${NC}"; show_menu ;;
    esac
}

# Make the script executable
chmod +x run_examples.sh

# Show welcome message
echo -e "${GREEN}Welcome to the Chainlit RAG Tutorial!${NC}"
echo -e "${BLUE}This script will help you run the different examples in this tutorial.${NC}"
echo ""

# Check if OpenAI API key is set
check_api_key

# Show the main menu
show_menu