# tests/conftest.py
import sys
import os

# Get the absolute path of the directory containing this file (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root, then point to src
# Structure: project_root/tests/conftest.py -> project_root/src
src_path = os.path.join(os.path.dirname(current_dir), "src")

# Insert it into sys.path if it's not already there
if src_path not in sys.path:
    sys.path.insert(0, src_path)