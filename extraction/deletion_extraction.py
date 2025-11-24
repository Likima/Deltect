"""
Extract deletion variants from BAM files by analyzing CIGAR strings and alignment patterns.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
import argparse

try:
    import pysam
    _HAS_PYSAM = True
except ImportError:
    _HAS_PYSAM = False

logger = logging.getLogger(__name__)


@dataclass
class DeletionVariant:
    """Represents a deletion found in a sequencing read."""
    
    # Core deletion information
    chromosome: str
    start: int
    end: int
    length: int
    
    # Read information
    read_name: str
    mapping_quality: int
    cigar_string: str
    cigar_tuples: List[Tuple[int, int]]
    
    # Read flags
    is_paired: bool = False
    is_proper_pair: bool = False
    is_reverse: bool = False
    is_secondary: bool = False
    is_supplementary: bool = False
    is_duplicate: bool = False
    
    # Reference sequence context (optional)
    reference_before: Optional[str] = None
    reference_after: Optional[str] = None
    
    # Position in read where deletion occurs
    read_position: int = 0
    
    # Optional tags from BAM file
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # Gene annotation (optional - populated by annotate_with_genes)
    gene: Optional[str] = None
    consequence: Optional[str] = None
    
    def to_variant_dict(self) -> Dict:
        """Convert to variant dictionary format compatible with pathogenicity predictor.
        
        This matches the ClinVar format expected by the model with gene annotations.
        
        Returns:
            Dictionary compatible with DeletionPathogenicityPredictor
        """
        # Clean chromosome name (remove 'chr' prefix if present)
        chrom = self.chromosome.replace('chr', '')
        
        # Use annotated gene if available, otherwise 'N/A'
        gene = self.gene if self.gene else 'N/A'
        
        # Use annotated consequence if available
        consequence = self.consequence if self.consequence else 'N/A'
        
        # Determine condition based on gene (if annotated)
        condition = 'N/A'
        if self.gene and self.gene != 'N/A':
            condition = f'{self.gene}-related disorder'
        
        return {
            'uid': f"BAM_{self.read_name}_{self.start}_{self.end}",
            'gene': gene,
            'title': f'Deletion in {gene} from read {self.read_name}' if gene != 'N/A' else f'Deletion from read {self.read_name}',
            'chr': chrom,
            'start': str(self.start),
            'end': str(self.end),
            'assembly': 'GRCh37',
            'variant_type': 'deletion',
            'clinical_significance': 'unknown',
            'review_status': 'from_bam',
            'condition': condition,
            'consequence': consequence,
            
            # Additional BAM-specific metadata
            '_bam_metadata': {
                'read_name': self.read_name,
                'mapping_quality': self.mapping_quality,
                'cigar_string': self.cigar_string,
                'is_paired': self.is_paired,
                'is_proper_pair': self.is_proper_pair,
                'is_reverse': self.is_reverse,
                'length': self.length
            }
        }
    
    def __str__(self) -> str:
        gene_str = f", gene={self.gene}" if self.gene else ""
        return f"Deletion({self.chromosome}:{self.start}-{self.end}, len={self.length}{gene_str}, read={self.read_name})"


@dataclass
class DeletionCluster:
    """Group of deletions at similar positions (potential SV)."""
    
    chromosome: str
    start: int
    end: int
    deletions: List[DeletionVariant] = field(default_factory=list)
    
    @property
    def support_count(self) -> int:
        """Number of reads supporting this deletion."""
        return len(self.deletions)
    
    @property
    def mean_length(self) -> float:
        """Average deletion length in cluster."""
        if not self.deletions:
            return 0.0
        return sum(d.length for d in self.deletions) / len(self.deletions)
    
    @property
    def mean_mapping_quality(self) -> float:
        """Average mapping quality of supporting reads."""
        if not self.deletions:
            return 0.0
        return sum(d.mapping_quality for d in self.deletions) / len(self.deletions)


@dataclass
class DeletionExtractionMetadata:
    """Metadata about deletion extraction from a BAM region."""
    
    region: str
    total_reads_processed: int = 0
    reads_with_deletions: int = 0
    total_deletions: int = 0
    deletions_by_size: Dict[str, int] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Region: {self.region}",
            f"Total reads processed: {self.total_reads_processed}",
            f"Reads with deletions: {self.reads_with_deletions}",
            f"Total deletions found: {self.total_deletions}",
        ]
        
        if self.deletions_by_size:
            lines.append("Deletions by size:")
            for size_range, count in sorted(self.deletions_by_size.items()):
                lines.append(f"  {size_range}: {count}")
        
        return "\n".join(lines)


class DeletionExtractor:
    """Extract deletion variants from BAM files."""
    
    def __init__(self, reference_fasta: Optional[str] = None, gene_annotation_gtf: Optional[str] = None):
        """Initialize deletion extractor.
        
        Args:
            reference_fasta: Path to reference genome FASTA file (optional, for sequence context)
            gene_annotation_gtf: Path to GTF/GFF gene annotation file (optional, for gene enrichment)
        """
        self.reference_fasta = reference_fasta
        self.gene_annotation_gtf = gene_annotation_gtf
        self.fasta = None
        self.gene_intervals = {}  # chromosome -> list of (start, end, gene_name, feature_type)
        
        if reference_fasta and _HAS_PYSAM:
            try:
                logger.debug(f"DEBUG: Initializing DeletionExtractor with reference_fasta {reference_fasta}")
                if Path(reference_fasta).exists():
                    self.fasta = pysam.FastaFile(reference_fasta)
                    logger.debug("DEBUG: Reference Genome Found!")
                else:
                    logger.warning(f"Reference genome not found: {reference_fasta}")
            except Exception as e:
                logger.warning(f"Could not open reference genome: {e}")
        
        # Load gene annotations if provided
        if gene_annotation_gtf:
            self._load_gene_annotations(gene_annotation_gtf)
    
    def _load_gene_annotations(self, gtf_file: str):
        """Load gene annotations from GTF/GFF file.
        
        Args:
            gtf_file: Path to GTF or GFF file
        """
        if not Path(gtf_file).exists():
            logger.warning(f"Gene annotation file not found: {gtf_file}")
            return
        
        logger.info(f"Loading gene annotations from {gtf_file}")
        
        try:
            gene_count = 0
            with open(gtf_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) < 9:
                        continue
                    
                    chrom = parts[0].replace('chr', '')
                    feature_type = parts[2]
                    start = int(parts[3])
                    end = int(parts[4])
                    attributes = parts[8]
                    
                    # Only process gene features
                    if feature_type not in ['gene', 'exon', 'CDS']:
                        continue
                    
                    # Extract gene name from attributes
                    gene_name = None
                    for attr in attributes.split(';'):
                        attr = attr.strip()
                        if attr.startswith('gene_name'):
                            gene_name = attr.split('"')[1]
                            break
                        elif attr.startswith('gene_id'):
                            gene_name = attr.split('"')[1]
                            break
                    
                    if gene_name:
                        if chrom not in self.gene_intervals:
                            self.gene_intervals[chrom] = []
                        
                        self.gene_intervals[chrom].append((start, end, gene_name, feature_type))
                        gene_count += 1
            
            # Sort intervals by start position for efficient lookup
            for chrom in self.gene_intervals:
                self.gene_intervals[chrom].sort(key=lambda x: x[0])
            
            logger.info(f"Loaded {gene_count} gene annotations across {len(self.gene_intervals)} chromosomes")
            
        except Exception as e:
            logger.error(f"Error loading gene annotations: {e}")
    
    def annotate_with_genes(self, deletions: List[DeletionVariant]) -> List[DeletionVariant]:
        """Annotate deletions with overlapping gene information.
        
        Args:
            deletions: List of DeletionVariant objects
            
        Returns:
            Same list with gene and consequence fields populated
        """
        if not self.gene_intervals:
            logger.warning("No gene annotations loaded - variants will have gene='N/A'")
            return deletions
        
        annotated_count = 0
        
        for deletion in deletions:
            chrom = deletion.chromosome.replace('chr', '')
            
            if chrom not in self.gene_intervals:
                continue
            
            # Find overlapping genes
            overlapping_genes = set()
            overlapping_features = set()
            
            for gene_start, gene_end, gene_name, feature_type in self.gene_intervals[chrom]:
                # Check if deletion overlaps with gene
                if deletion.start < gene_end and deletion.end > gene_start:
                    overlapping_genes.add(gene_name)
                    overlapping_features.add(feature_type)
            
            if overlapping_genes:
                # Use the first gene if multiple (or could concatenate)
                deletion.gene = list(overlapping_genes)[0] if len(overlapping_genes) == 1 else ';'.join(sorted(overlapping_genes))
                
                # Determine consequence based on feature type
                if 'CDS' in overlapping_features:
                    deletion.consequence = 'coding_sequence_variant'
                elif 'exon' in overlapping_features:
                    deletion.consequence = 'exon_variant'
                elif 'gene' in overlapping_features:
                    deletion.consequence = 'intragenic_variant'
                else:
                    deletion.consequence = 'gene_variant'
                
                annotated_count += 1
        
        logger.info(f"Annotated {annotated_count}/{len(deletions)} deletions with gene information")
        
        return deletions
    
    def extract_deletions_from_read(
        self,
        read: pysam.AlignedSegment,
        min_deletion_length: int = 1,
        max_deletion_length: int = 1_000_000,
        context_length: int = 10
    ) -> List[DeletionVariant]:
        """Extract deletions from a single read.
        
        Args:
            read: pysam AlignedSegment
            min_deletion_length: Minimum deletion size to extract
            max_deletion_length: Maximum deletion size to extract
            context_length: Length of reference context to extract on each side
            
        Returns:
            List of DeletionVariant objects
        """
        deletions = []
        
        if not read.cigartuples:
            return deletions
        
        # Track position in reference
        ref_pos = read.reference_start
        read_pos = 0
        
        for op, length in read.cigartuples:
            if op == 0:  # Match (M)
                ref_pos += length
                read_pos += length
            elif op == 1:  # Insertion (I)
                read_pos += length
            elif op == 2:  # Deletion (D)
                # Check deletion size filters
                if min_deletion_length <= length <= max_deletion_length:
                    del_start = ref_pos
                    del_end = ref_pos + length
                    
                    # Extract reference context if FASTA available
                    ref_before = None
                    ref_after = None
                    
                    if self.fasta and read.reference_name:
                        try:
                            # Try different chromosome name formats
                            chrom_variants = [
                                read.reference_name,
                                read.reference_name.replace('chr', ''),
                                f"chr{read.reference_name.replace('chr', '')}"
                            ]
                            
                            for chrom_name in chrom_variants:
                                if chrom_name in self.fasta.references:
                                    # Extract context before deletion
                                    if del_start >= context_length:
                                        ref_before = self.fasta.fetch(
                                            chrom_name,
                                            del_start - context_length,
                                            del_start
                                        )
                                    
                                    # Extract context after deletion
                                    ref_after = self.fasta.fetch(
                                        chrom_name,
                                        del_end,
                                        del_end + context_length
                                    )
                                    break
                        except Exception as e:
                            logger.debug(f"Could not extract reference context: {e}")
                    
                    # Extract tags
                    tags = {}
                    try:
                        for tag_name, tag_value in read.get_tags():
                            tags[tag_name] = tag_value
                    except:
                        pass
                    
                    # Create deletion variant
                    deletion = DeletionVariant(
                        chromosome=read.reference_name or "unknown",
                        start=del_start,
                        end=del_end,
                        length=length,
                        read_name=read.query_name,
                        mapping_quality=read.mapping_quality,
                        cigar_string=read.cigarstring or "",
                        cigar_tuples=list(read.cigartuples),
                        is_paired=read.is_paired,
                        is_proper_pair=read.is_proper_pair,
                        is_reverse=read.is_reverse,
                        is_secondary=read.is_secondary,
                        is_supplementary=read.is_supplementary,
                        is_duplicate=read.is_duplicate,
                        reference_before=ref_before,
                        reference_after=ref_after,
                        read_position=read_pos,
                        tags=tags
                    )
                    
                    deletions.append(deletion)
                
                ref_pos += length
            elif op == 3:  # Reference skip (N)
                ref_pos += length
            elif op == 4:  # Soft clip (S)
                read_pos += length
            elif op == 5:  # Hard clip (H)
                pass
        
        return deletions
    
    def extract_deletions_from_region(
        self,
        bam_file: str,
        chromosome: str,
        start: int,
        end: int,
        min_deletion_length: int = 1,
        max_deletion_length: int = 1_000_000,
        min_mapping_quality: int = 0,
        skip_duplicates: bool = True,
        skip_secondary: bool = True,
        skip_supplementary: bool = True,
        context_length: int = 10,
        annotate_genes: bool = True
    ) -> List[DeletionVariant]:
        """Extract all deletions from a genomic region in a BAM file.
        
        Args:
            bam_file: Path to BAM file (can be local or URL)
            chromosome: Chromosome name
            start: Start position (0-based)
            end: End position (0-based, exclusive)
            min_deletion_length: Minimum deletion size
            max_deletion_length: Maximum deletion size
            min_mapping_quality: Minimum mapping quality threshold
            skip_duplicates: Skip duplicate reads
            skip_secondary: Skip secondary alignments
            skip_supplementary: Skip supplementary alignments
            context_length: Reference context length
            annotate_genes: Whether to annotate with gene information
            
        Returns:
            List of DeletionVariant objects
        """
        if not _HAS_PYSAM:
            raise ImportError("pysam is required for BAM file processing")
        
        logger.debug(f"DEBUG(b): BF sent sent - {bam_file}")
        
        all_deletions = []
        
        try:
            # Open BAM file
            bamfile = pysam.AlignmentFile(bam_file, "rb")
            
            # Fetch reads in region
            reads_iter = bamfile.fetch(chromosome, start, end)
            
            logger.debug(f"DEBUG(b) reads_iter - {reads_iter}")
            
            total_reads = 0
            reads_with_dels = 0
            
            for read in reads_iter:
                total_reads += 1
                
                logger.debug(f"DEBUG(b): read - {read}")
                
                # Apply filters
                if read.mapping_quality < min_mapping_quality:
                    continue
                if skip_duplicates and read.is_duplicate:
                    continue
                if skip_secondary and read.is_secondary:
                    continue
                if skip_supplementary and read.is_supplementary:
                    continue
                
                # Extract deletions from this read
                deletions = self.extract_deletions_from_read(
                    read,
                    min_deletion_length=min_deletion_length,
                    max_deletion_length=max_deletion_length,
                    context_length=context_length
                )
                
                if deletions:
                    reads_with_dels += 1
                    all_deletions.extend(deletions)
            
            bamfile.close()
            
            logger.info(f"Processed {total_reads} reads, found {len(all_deletions)} deletions in {reads_with_dels} reads")
            
            # Annotate with genes if requested and annotations available
            if annotate_genes and self.gene_intervals:
                all_deletions = self.annotate_with_genes(all_deletions)
            
        except Exception as e:
            logger.error(f"Error processing BAM file: {e}")
            raise
        
        return all_deletions
    
    def cluster_deletions(
        self,
        deletions: List[DeletionVariant],
        position_tolerance: int = 10
    ) -> List[DeletionCluster]:
        """Cluster deletions by position to identify potential SVs.
        
        Args:
            deletions: List of DeletionVariant objects
            position_tolerance: Maximum distance between deletions to cluster
            
        Returns:
            List of DeletionCluster objects
        """
        if not deletions:
            return []
        
        # Sort deletions by position
        sorted_dels = sorted(deletions, key=lambda d: (d.chromosome, d.start))
        
        clusters = []
        current_cluster = None
        
        for deletion in sorted_dels:
            if current_cluster is None:
                # Start new cluster
                current_cluster = DeletionCluster(
                    chromosome=deletion.chromosome,
                    start=deletion.start,
                    end=deletion.end,
                    deletions=[deletion]
                )
            else:
                # Check if deletion belongs to current cluster
                if (deletion.chromosome == current_cluster.chromosome and
                    deletion.start <= current_cluster.end + position_tolerance):
                    # Add to current cluster
                    current_cluster.deletions.append(deletion)
                    current_cluster.end = max(current_cluster.end, deletion.end)
                else:
                    # Save current cluster and start new one
                    clusters.append(current_cluster)
                    current_cluster = DeletionCluster(
                        chromosome=deletion.chromosome,
                        start=deletion.start,
                        end=deletion.end,
                        deletions=[deletion]
                    )
        
        # Don't forget last cluster
        if current_cluster is not None:
            clusters.append(current_cluster)
        
        return clusters
    
    def __del__(self):
        """Clean up FASTA file handle."""
        if self.fasta:
            self.fasta.close()


def main():
    """Example usage of DeletionExtractor."""
    
    parser = argparse.ArgumentParser(description="Extract deletions from BAM file")
    parser.add_argument('--bam', required=True, help='BAM file path or URL')
    parser.add_argument('--chr', default='chr22', help='Chromosome')
    parser.add_argument('--start', type=int, default=1947000, help='Start position')
    parser.add_argument('--end', type=int, default=1947019, help='End position')
    parser.add_argument('--reference', help='Reference genome FASTA')
    parser.add_argument('--gtf', help='Gene annotation GTF file')
    parser.add_argument('--min-del-size', type=int, default=1, help='Minimum deletion size')
    parser.add_argument('--min-mapq', type=int, default=0, help='Minimum mapping quality')
    parser.add_argument('--output', default='deletions.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    logger.debug(f"DEBUG(b): Extracting Deletions w. reference: {args.reference}")
    
    # Initialize extractor
    extractor = DeletionExtractor(
        reference_fasta=args.reference,
        gene_annotation_gtf=args.gtf
    )
    
    # Extract deletions
    deletions = extractor.extract_deletions_from_region(
        bam_file=args.bam,
        chromosome=args.chr,
        start=args.start,
        end=args.end,
        min_deletion_length=args.min_del_size,
        min_mapping_quality=args.min_mapq,
        annotate_genes=True
    )
    
    print(f"\nFound {len(deletions)} deletions")
    
    # Show sample with gene annotations
    if deletions:
        print("\nSample deletion with annotations:")
        sample = deletions[0]
        print(f"  Position: {sample.chromosome}:{sample.start}-{sample.end}")
        print(f"  Gene: {sample.gene or 'N/A'}")
        print(f"  Consequence: {sample.consequence or 'N/A'}")
    
    # Convert to dictionaries for JSON export
    deletions_dict = [d.to_variant_dict() for d in deletions]
    
    # Save to JSON
    output_data = {
        'region': f"{args.chr}:{args.start}-{args.end}",
        'total_deletions': len(deletions),
        'deletions': deletions_dict
    }
    
    with open(args.output, 'w') as f:
        import json
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(deletions)} deletions to {args.output}")


if __name__ == "__main__":
    main()