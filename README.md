# Deltect

## Setup

### Prerequisites
- Git
- libuv (system library)
    - Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y libuv1-dev`
    - macOS: `brew install libuv`
    - Windows: use vcpkg or Chocolatey (`vcpkg install libuv` or `choco install libuv`)

### Install dependencies
- Python (recommended)
    - Create and activate a virtual env:
        - `python -m venv .venv`
        - `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)
    - Install project deps:
        - `pip install -r requirements.txt`
    - Install the uv wrapper used by this project:
        - `pip install uv`

### Run
- Python:
    - `uv run main.py`