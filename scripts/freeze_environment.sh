#!/bin/bash
# scripts/freeze_environment.sh

# Run this to generate a bit-for-bit reproducible lock file of your current environment.
# This ensures that other users or CI/CD pipelines can recreate the exact same environment.

echo "Freezing environment to environment.lock.yml..."
conda env export --no-builds > environment.lock.yml
echo "Done! The 'environment.lock.yml' file has been created."
