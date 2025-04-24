#!/bin/bash

# Project base directory
PROJECT_ROOT="project-root"

# Create the updated folder structure
mkdir -p "$PROJECT_ROOT"/src/main/python
mkdir -p "$PROJECT_ROOT"/notebooks

# Create placeholder files
touch "$PROJECT_ROOT"/src/main/python/app.py
touch "$PROJECT_ROOT"/requirements.txt
touch "$PROJECT_ROOT"/README.md
touch "$PROJECT_ROOT"/.gitignore

echo "✅ Folder structure set up under '$PROJECT_ROOT'"