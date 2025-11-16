"""
Sample normal genomic regions from reference genome for baseline training data.
Uses pysam for fast indexed access to reference genome.
"""
import random
from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    import pysam
    _HAS_PYSAM = True
except ImportError:
    _HAS_PYSAM = False

logger = logging.getLogger(__name__)


class ReferenceGenomeSampler:
    """Sample normal genomic regions from reference genome FASTA using indexed access."""
    
    # Standard chromosomes (exclude sex chromosomes for consistency if needed)
    AUTOSOMES = [str(i) for i in range(1, 23)]
    SEX_CHROMOSOMES = ["X", "Y"]
    ALL_CHROMOSOMES = AUTOSOMES + SEX_CHROMOSOMES
    
    def __init__(self, reference_fasta: str, chromosomes: Optional[List[str]] = None):
        """
        Initialize reference genome sampler.
        
        Args:
            reference_fasta: Path to reference FASTA file (e.g., hs37d5.fa)
            chromosomes: List of chromosomes to sample from (default: all autosomes)
        """
        if not _HAS_PYSAM:
            raise ImportError("pysam required for fast FASTA access: pip install pysam")
        
        self.reference_path = Path(reference_fasta)
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference genome not found: {reference_fasta}")
        
        # Check for FASTA index
        fai_path = Path(str(self.reference_path) + ".fai")
        if not fai_path.exists():
            raise FileNotFoundError(
                f"FASTA index not found: {fai_path}\n"
                f"Create index with: samtools faidx {reference_fasta}"
            )
        
        self.chromosomes = chromosomes or self.AUTOSOMES
        self.chromosome_lengths = {}
        
        logger.info(f"Loading reference genome: {reference_fasta}")
        
        # Open indexed FASTA file
        self.fasta = pysam.FastaFile(str(self.reference_path))
        
        self._load_chromosome_lengths()
    
    def _load_chromosome_lengths(self):
        """Load chromosome lengths from FASTA index."""
        fai_path = Path(str(self.reference_path) + ".fai")
        
        logger.info("Using FASTA index file")
        with open(fai_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                chrom = parts[0]
                length = int(parts[1])
                
                # Normalize chromosome names (handle both "chr1" and "1" formats)
                chrom_clean = chrom.replace("chr", "")
                
                if chrom_clean in self.chromosomes:
                    # Store both the display name and actual name in FASTA
                    self.chromosome_lengths[chrom_clean] = {
                        'length': length,
                        'fasta_name': chrom  # Actual name in FASTA file
                    }
        
        logger.info(f"Loaded {len(self.chromosome_lengths)} chromosomes")
        for chrom in sorted(self.chromosome_lengths.keys(), key=lambda x: int(x) if x.isdigit() else ord(x[0])):
            length = self.chromosome_lengths[chrom]['length']
            logger.info(f"  chr{chrom}: {length:,} bp")
    
    def sample_random_regions(
        self,
        num_samples: int,
        min_length: int = 50,
        max_length: int = 10000,
        avoid_n_regions: bool = True
    ) -> List[Dict]:
        """
        Sample random genomic regions from reference genome.
        
        Args:
            num_samples: Number of regions to sample
            min_length: Minimum region length
            max_length: Maximum region length
            avoid_n_regions: Skip regions with >50% N bases
            
        Returns:
            List of sampled region dictionaries
        """
        samples = []
        attempts = 0
        max_attempts = num_samples * 10  # Prevent infinite loop
        
        logger.info(f"Sampling {num_samples} random genomic regions")
        
        chrom_list = list(self.chromosome_lengths.keys())
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Random chromosome
            chrom = random.choice(chrom_list)
            chrom_info = self.chromosome_lengths[chrom]
            chrom_length = chrom_info['length']
            fasta_name = chrom_info['fasta_name']
            
            # Random length
            length = random.randint(min_length, max_length)
            
            # Random start position
            max_start = chrom_length - length
            if max_start <= 0:
                continue
            
            start = random.randint(0, max_start)
            end = start + length
            
            # Extract sequence using indexed access (FAST!)
            sequence = self._extract_sequence(fasta_name, start, end)
            
            if sequence is None:
                continue
            
            # Skip if too many N bases
            if avoid_n_regions:
                n_count = sequence.upper().count('N')
                if n_count / len(sequence) > 0.5:
                    continue
            
            samples.append({
                "uid": f"REF_{chrom}_{start}_{end}",
                "gene": "N/A",
                "title": f"Reference genome region chr{chrom}:{start}-{end}",
                "chr": chrom,
                "start": str(start),
                "end": str(end),
                "assembly": "GRCh37",  # Adjust if using GRCh38
                "variant_type": "reference",
                "clinical_significance": "benign",
                "review_status": "reference",
                "condition": "normal",
                "consequence": "N/A",
                "sequence": sequence,
                "length": length,
                "is_str": False,
                "label": 0  # Negative class
            })
            
            if len(samples) % 500 == 0:
                logger.info(f"Sampled {len(samples)}/{num_samples} regions")
        
        logger.info(f"Sampled {len(samples)} regions in {attempts} attempts")
        
        return samples
    
    def _extract_sequence(self, chrom: str, start: int, end: int) -> Optional[str]:
        """
        Extract DNA sequence from reference genome using indexed access.
        
        Args:
            chrom: Chromosome name (as it appears in FASTA file)
            start: Start position (0-based)
            end: End position (0-based, exclusive)
            
        Returns:
            DNA sequence string or None if extraction fails
        """
        try:
            # Fast indexed access!
            sequence = self.fasta.fetch(chrom, start, end)
            return sequence
        except Exception as e:
            logger.debug(f"Failed to extract {chrom}:{start}-{end}: {e}")
            return None
    
    def match_str_distribution(
        self,
        str_variants: List[Dict],
        ratio: float = 1.0
    ) -> List[Dict]:
        """
        Sample reference regions matching STR variant length distribution.
        
        Args:
            str_variants: List of STR variant dictionaries
            ratio: Ratio of reference samples to STR samples (1.0 = equal)
            
        Returns:
            List of matched reference regions
        """
        # Extract STR lengths
        str_lengths = []
        for variant in str_variants:
            try:
                start = int(variant.get("start", 0))
                end = int(variant.get("end", 0))
                length = end - start
                if length > 0:
                    str_lengths.append(length)
            except (ValueError, TypeError):
                continue
        
        if not str_lengths:
            logger.warning("No valid STR lengths found")
            return []
        
        num_samples = int(len(str_variants) * ratio)
        logger.info(f"Matching {num_samples} reference samples to STR distribution")
        logger.info(f"STR length range: {min(str_lengths)}-{max(str_lengths)} bp")
        
        samples = []
        attempts = 0
        max_attempts = num_samples * 10
        
        chrom_list = list(self.chromosome_lengths.keys())
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Sample a length from STR distribution
            target_length = random.choice(str_lengths)
            
            # Random chromosome
            chrom = random.choice(chrom_list)
            chrom_info = self.chromosome_lengths[chrom]
            chrom_length = chrom_info['length']
            fasta_name = chrom_info['fasta_name']
            
            max_start = chrom_length - target_length
            if max_start <= 0:
                continue
            
            start = random.randint(0, max_start)
            end = start + target_length
            
            # Fast indexed extraction
            sequence = self._extract_sequence(fasta_name, start, end)
            
            if sequence is None:
                continue
            
            # Skip if too many N bases
            n_count = sequence.upper().count('N')
            if n_count / len(sequence) > 0.5:
                continue
            
            samples.append({
                "uid": f"REF_{chrom}_{start}_{end}",
                "gene": "N/A",
                "title": f"Reference genome region chr{chrom}:{start}-{end}",
                "chr": chrom,
                "start": str(start),
                "end": str(end),
                "assembly": "GRCh37",
                "variant_type": "reference",
                "clinical_significance": "benign",
                "review_status": "reference",
                "condition": "normal",
                "consequence": "N/A",
                "sequence": sequence,
                "length": target_length,
                "is_str": False,
                "label": 0
            })
            
            if len(samples) % 500 == 0:
                logger.info(f"Matched {len(samples)}/{num_samples} samples")
        
        logger.info(f"Sampled {len(samples)} regions in {attempts} attempts")
        
        return samples
    
    def __del__(self):
        """Close FASTA file on cleanup."""
        if hasattr(self, 'fasta'):
            self.fasta.close()