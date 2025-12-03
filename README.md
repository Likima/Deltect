# Deltect: Genomic Deletion Pathogenicity Classifier

A machine learning system that predicts whether genomic deletions are pathogenic or benign using ensemble classification trained on ClinVar annotations and reference genome data.

## Features

- **Ensemble ML Model**: Random Forest + Gradient Boosting + XGBoost with soft voting
- **High Performance**: 97.5% AUC-ROC, 96% recall, 89% precision on test set
- **BAM File Analysis**: Extract deletions directly from sequencing data via CIGAR string parsing
- **Gene Annotation**: Integrate GENCODE gene context for improved predictions
- **Reference Genome Balancing**: Sample benign regions to address ClinVar's pathogenic bias

---

## Quick Start

**System Requirements:**
- Python 3.9 or higher
- 8GB+ RAM
- 5GB disk space (for reference files)

**Required Tools:**
- Git
- `samtools` (for BAM indexing)
- `wget` (for downloading reference files)

**System Libraries:**
- **libuv** (required by `uv` package manager)
  - Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y libuv1-dev`
  - macOS: `brew install libuv`
  - Windows: `vcpkg install libuv` or `choco install libuv`

# Run the Quick Install Bash Script
```bash
./download_reference.sh
```

# Run with gene annotations
```bash
uv run main.py --mode both \
    --bam [file].bam \
    --chr [chr] --start [start] --end [end] \
    --reference hs37d5.fa \
    --gtf gencode.v19.annotation.gtf
```

# Example Usage

See notebooks/example_usage.ipynb to view the example usage of Deltect
