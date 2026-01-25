#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "Starting local reproducibility check..."

# Check if required tools are installed
if ! command -v black &> /dev/null; then
    echo -e "${RED}Error: black is not installed.${NC}"
    exit 1
fi
if ! command -v isort &> /dev/null; then
    echo -e "${RED}Error: isort is not installed.${NC}"
    exit 1
fi
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed.${NC}"
    exit 1
fi

# Run Black
echo -n "Running black..."
if black --check . &> /dev/null; then
    echo -e "${GREEN} PASSED${NC}"
else
    echo -e "${RED} FAILED${NC}"
    echo "Run 'black .' to fix."
    exit 1
fi

# Run Isort
echo -n "Running isort..."
if isort --check . &> /dev/null; then
    echo -e "${GREEN} PASSED${NC}"
else
    echo -e "${RED} FAILED${NC}"
    echo "Run 'isort .' to fix."
    exit 1
fi

# Run Lightweight Tests
echo "Running lightweight tests..."
if pytest -m lightweight; then
    echo -e "${GREEN}Tests PASSED${NC}"
else
    echo -e "${RED}Tests FAILED${NC}"
    exit 1
fi

echo -e "${GREEN}READY TO COMMIT${NC}"
