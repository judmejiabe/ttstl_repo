#!/usr/bin/env python3
# scripts/add_license_headers.py

import os

LICENSE_HEADER = """# Copyright (c) 2026 Juan Diego Mejía Becerra. Licensed under the MIT License.
"""

TARGET_EXTENSIONS = {'.py', '.R'}
TARGET_DIRS = ['src/ttstl', 'scripts']

def has_license_header(file_path):
    """Checks if the file already starts with the license header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = f.readline()
            # Check if the first line roughly matches our expected header or if it's a shebang followed by header
            if LICENSE_HEADER.strip() in first_lines:
                return True
            
            # Handle files with shebangs
            if first_lines.startswith("#!"):
                second_line = f.readline()
                if LICENSE_HEADER.strip() in second_line:
                    return True
            
        return False
    except Exception:
        return False

def add_header(file_path):
    """Prepends the license header to the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = ""
        # Preserve shebang if present
        if content.startswith("#!"):
            lines = content.splitlines(keepends=True)
            new_content = lines[0] + LICENSE_HEADER + "".join(lines[1:])
        else:
            new_content = LICENSE_HEADER + content

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Stamped {file_path}")
    except Exception as e:
        print(f"❌ Failed to stamp {file_path}: {e}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for target_dir in TARGET_DIRS:
        full_path = os.path.join(root_dir, target_dir)
        if not os.path.exists(full_path):
            print(f"Warning: Directory not found: {full_path}")
            continue

        for root, _, files in os.walk(full_path):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in TARGET_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    if has_license_header(file_path):
                        print(f"⏭️  Skipped {file_path} (Header present)")
                    else:
                        add_header(file_path)

if __name__ == "__main__":
    main()
