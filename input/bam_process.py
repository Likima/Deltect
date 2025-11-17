#!/usr/bin/env python3
import sys
import argparse
import pysam
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json

@dataclass
class ReadMetadata:
    """Structured metadata for a single BAM read."""
    query_name: str
    flag: int
    reference_name: Optional[str]
    reference_start: int  # 0-based
    reference_end: int    # 0-based
    mapping_quality: int
    cigar_string: Optional[str]
    cigar_tuples: List[tuple]
    next_reference_name: Optional[str]
    next_reference_start: Optional[int]
    template_length: int
    query_sequence: Optional[str]
    query_qualities: Optional[List[int]]
    query_length: int
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # Flag-based boolean properties
    is_paired: bool = False
    is_proper_pair: bool = False
    is_unmapped: bool = False
    mate_is_unmapped: bool = False
    is_reverse: bool = False
    mate_is_reverse: bool = False
    is_read1: bool = False
    is_read2: bool = False
    is_secondary: bool = False
    is_qcfail: bool = False
    is_duplicate: bool = False
    is_supplementary: bool = False
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return {
            'query_name': self.query_name,
            'flag': self.flag,
            'reference_name': self.reference_name,
            'reference_start': self.reference_start,
            'reference_end': self.reference_end,
            'mapping_quality': self.mapping_quality,
            'cigar_string': self.cigar_string,
            'cigar_tuples': self.cigar_tuples,
            'next_reference_name': self.next_reference_name,
            'next_reference_start': self.next_reference_start,
            'template_length': self.template_length,
            'query_length': self.query_length,
            'tags': self.tags,
            'flags': {
                'is_paired': self.is_paired,
                'is_proper_pair': self.is_proper_pair,
                'is_unmapped': self.is_unmapped,
                'mate_is_unmapped': self.mate_is_unmapped,
                'is_reverse': self.is_reverse,
                'mate_is_reverse': self.mate_is_reverse,
                'is_read1': self.is_read1,
                'is_read2': self.is_read2,
                'is_secondary': self.is_secondary,
                'is_qcfail': self.is_qcfail,
                'is_duplicate': self.is_duplicate,
                'is_supplementary': self.is_supplementary,
            }
        }
    
    def to_sam_format(self) -> str:
        """Convert metadata back to SAM format string."""
        # Convert qualities back to ASCII
        qual_string = ''.join(chr(q + 33) for q in self.query_qualities) if self.query_qualities else '*'
        
        output = (f"{self.query_name}\t"
                 f"{self.flag}\t"
                 f"{self.reference_name or '*'}\t"
                 f"{self.reference_start + 1}\t"  # SAM is 1-based
                 f"{self.mapping_quality}\t"
                 f"{self.cigar_string or '*'}\t"
                 f"{self.next_reference_name or '*'}\t"
                 f"{(self.next_reference_start + 1) if self.next_reference_start is not None else 0}\t"
                 f"{self.template_length}\t"
                 f"{self.query_sequence or '*'}\t"
                 f"{qual_string}")
        
        # Add tags
        for tag_name, tag_value in self.tags.items():
            if isinstance(tag_value, int):
                output += f"\t{tag_name}:i:{tag_value}"
            elif isinstance(tag_value, float):
                output += f"\t{tag_name}:f:{tag_value}"
            elif isinstance(tag_value, str):
                output += f"\t{tag_name}:Z:{tag_value}"
            elif isinstance(tag_value, bytes):
                output += f"\t{tag_name}:H:{tag_value.hex()}"
        
        return output


@dataclass
class BamRegionMetadata:
    """Metadata for a region of a BAM file."""
    bam_file: str
    chromosome: str
    start: int
    end: int
    has_index: bool
    total_reads: int
    reads: List[ReadMetadata] = field(default_factory=list)
    header: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        """Convert region metadata to dictionary."""
        return {
            'bam_file': self.bam_file,
            'region': {
                'chromosome': self.chromosome,
                'start': self.start,
                'end': self.end
            },
            'has_index': self.has_index,
            'total_reads': self.total_reads,
            'reads': [read.to_dict() for read in self.reads],
            'header': self.header
        }
    
    def to_json(self, indent=2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def get_read_statistics(self) -> dict:
        """Calculate statistics about the reads."""
        if not self.reads:
            return {}
        
        return {
            'total_reads': len(self.reads),
            'mapped_reads': sum(1 for r in self.reads if not r.is_unmapped),
            'unmapped_reads': sum(1 for r in self.reads if r.is_unmapped),
            'paired_reads': sum(1 for r in self.reads if r.is_paired),
            'proper_pairs': sum(1 for r in self.reads if r.is_proper_pair),
            'duplicates': sum(1 for r in self.reads if r.is_duplicate),
            'supplementary': sum(1 for r in self.reads if r.is_supplementary),
            'secondary': sum(1 for r in self.reads if r.is_secondary),
            'avg_mapping_quality': sum(r.mapping_quality for r in self.reads) / len(self.reads),
            'avg_read_length': sum(r.query_length for r in self.reads) / len(self.reads),
        }


def parse_read_to_metadata(read: pysam.AlignedSegment) -> ReadMetadata:
    """Parse a pysam AlignedSegment into ReadMetadata object."""
    
    # Extract all tags
    tags = {}
    for tag_name, tag_value in read.get_tags():
        tags[tag_name] = tag_value
    
    return ReadMetadata(
        query_name=read.query_name,
        flag=read.flag,
        reference_name=read.reference_name,
        reference_start=read.reference_start if read.reference_start is not None else -1,
        reference_end=read.reference_end if read.reference_end is not None else -1,
        mapping_quality=read.mapping_quality,
        cigar_string=read.cigarstring,
        cigar_tuples=read.cigartuples if read.cigartuples else [],
        next_reference_name=read.next_reference_name,
        next_reference_start=read.next_reference_start,
        template_length=read.template_length,
        query_sequence=read.query_sequence,
        query_qualities=read.query_qualities.tolist() if read.query_qualities is not None else None,
        query_length=read.query_length,
        tags=tags,
        is_paired=read.is_paired,
        is_proper_pair=read.is_proper_pair,
        is_unmapped=read.is_unmapped,
        mate_is_unmapped=read.mate_is_unmapped,
        is_reverse=read.is_reverse,
        mate_is_reverse=read.mate_is_reverse,
        is_read1=read.is_read1,
        is_read2=read.is_read2,
        is_secondary=read.is_secondary,
        is_qcfail=read.is_qcfail,
        is_duplicate=read.is_duplicate,
        is_supplementary=read.is_supplementary,
    )


def extract_bam_region(bam_file="ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/10XGenomics/NA24385_phased_possorted_bam.bam", 
                      chromosome="chr22", start=1000000, end=1000100, num_lines=10, 
                      return_metadata=True, print_reads=True, print_stats=True):
    """
    Extract reads from a specific region of a BAM file and parse into metadata objects.
    Streams directly from URL without downloading.
    
    Args:
        bam_file: Path to the BAM file or URL
        chromosome: Chromosome name (default: chr20)
        start: Start position (default: 1000000)
        end: End position (default: 1000100)
        num_lines: Number of lines to display (default: 10)
        return_metadata: If True, return BamRegionMetadata object
        print_reads: If True, print reads in SAM format
        print_stats: If True, print read statistics
    
    Returns:
        BamRegionMetadata object if return_metadata=True, else True/False
    """
    try:
        print(f"Opening BAM file: {bam_file}")
        print(f"Extracting region {chromosome}:{start}-{end}")
        print("=" * 80)
        
        bamfile = pysam.AlignmentFile(bam_file, "rb", require_index=False)
        
        # Extract header information
        header_dict = dict(bamfile.header.to_dict())
        
        has_index = bamfile.has_index()
        
        if has_index:
            print("BAM file has index - using efficient region query")
            reads_iter = bamfile.fetch(chromosome, start, end)
        else:
            print("Warning: BAM file has no index - streaming entire file (slow)")
            reads_iter = (read for read in bamfile.fetch(until_eof=True) 
                         if read.reference_name == chromosome and 
                         start <= read.reference_start <= end)
        
        # Create region metadata object
        region_metadata = BamRegionMetadata(
            bam_file=bam_file,
            chromosome=chromosome,
            start=start,
            end=end,
            has_index=has_index,
            total_reads=0,
            header=header_dict
        )
        
        # Process reads
        line_count = 0
        for read in reads_iter:
            # Parse read into metadata
            read_metadata = parse_read_to_metadata(read)
            region_metadata.reads.append(read_metadata)
            region_metadata.total_reads += 1
            
            # Print in SAM format if requested
            if print_reads and line_count < num_lines:
                print(read_metadata.to_sam_format())
                line_count += 1
            
            if not return_metadata and line_count >= num_lines:
                break
        
        bamfile.close()
        
        if region_metadata.total_reads == 0:
            print("\nNo reads found in the specified region.")
        else:
            print(f"\nProcessed {region_metadata.total_reads} reads from region {chromosome}:{start}-{end}")
            
            if print_stats:
                print("\nRead Statistics:")
                stats = region_metadata.get_read_statistics()
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
        
        if return_metadata:
            return region_metadata
        return True
        
    except ValueError as e:
        print(f"Error: Invalid BAM file or region - {e}", file=sys.stderr)
        return None if return_metadata else False
    except IOError as e:
        print(f"Error: Cannot open BAM file - {e}", file=sys.stderr)
        print("Note: For FTP URLs, the BAM file must be accessible and may require an index file.", file=sys.stderr)
        return None if return_metadata else False
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None if return_metadata else False


def main():
    parser = argparse.ArgumentParser(
        description="Extract reads from a BAM file region and parse into metadata objects"
    )
    parser.add_argument(
        "bam_file", 
        nargs='?', 
        default="ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/10XGenomics/NA24385_phased_possorted_bam.bam",
        help="Path to the BAM file or URL"
    )
    parser.add_argument(
        "-c", "--chromosome", 
        default="chr20", 
        help="Chromosome (default: chr20)"
    )
    parser.add_argument(
        "-s", "--start", 
        type=int, 
        default=1000000, 
        help="Start position (default: 1000000)"
    )
    parser.add_argument(
        "-e", "--end", 
        type=int, 
        default=1000100, 
        help="End position (default: 1000100)"
    )
    parser.add_argument(
        "-n", "--num-lines", 
        type=int, 
        default=10, 
        help="Number of lines to display (default: 10)"
    )
    parser.add_argument(
        "--json-output",
        help="Save metadata to JSON file"
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Don't print reads to console"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't print statistics"
    )
    
    args = parser.parse_args()
    
    metadata = extract_bam_region(
        args.bam_file, 
        args.chromosome, 
        args.start, 
        args.end, 
        args.num_lines,
        return_metadata=True,
        print_reads=not args.no_print,
        print_stats=not args.no_stats
    )
    
    if metadata is None:
        sys.exit(1)
    
    # Save to JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            f.write(metadata.to_json())
        print(f"\nMetadata saved to: {args.json_output}")


if __name__ == "__main__":
    main()