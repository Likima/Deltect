#!/bin/bash

# Deltect Reference Files Downloader
# Downloads required genetics reference files for the Deltect pipeline

set -e

NCBI_GIAB_BASE="ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab"
GENCODE_BASE="ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19"


# Download human reference genome (hs37d5 - GRCh37)
echo "Downloading human reference genome (hs37d5)..."
if [ ! -f "hs37d5.fa" ]; then
    wget -c "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz"
    echo "Decompressing reference genome..."
    gunzip hs37d5.fa.gz
else
    echo "Reference genome already exists, skipping..."
fi

# Download GIAB HG002 truth VCF
echo "Downloading GIAB HG002 truth VCF..."
if [ ! -f "HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz" ]; then
    wget -c "${NCBI_GIAB_BASE}/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh37/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz"
    wget -c "${NCBI_GIAB_BASE}/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh37/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz.tbi"
else
    echo "Truth VCF already exists, skipping..."
fi

# Download gene annotation (GTF) - GENCODE v19
echo "Downloading gene annotation file (GENCODE v19)..."
if [ ! -f "gencode.v19.annotation.gtf.gz" ]; then
    wget -c "${GENCODE_BASE}/gencode.v19.annotation.gtf.gz"
    gunzip gencode.v19.annotation.gtf.gz
else
    echo "Gene annotation already exists, skipping..."
fi

# Create index for reference genome
echo "Creating reference genome index..."
if [ ! -f "hs37d5.fa.fai" ]; then
    if command -v samtools &> /dev/null; then
        samtools faidx hs37d5.fa
    else
        echo "Warning: samtools not found. Index will need to be created separately."
    fi
fi

echo ""
echo "Download complete!"
echo ""
echo "Downloaded files:"
ls -lh

cd ..