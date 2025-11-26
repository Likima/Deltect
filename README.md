# Deltect

## Setup

# Download GTF annotation file (GRCh37/hg19)
wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz
gunzip gencode.v19.annotation.gtf.gz

# Run with gene annotations
uv run main.py --mode both \
    --bam [file].bam \
    --chr [chr] --start [start] --end [end] \
    --reference hs37d5.fa \
    --gtf gencode.v19.annotation.gtf

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