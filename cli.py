"""
Command-line interface for the genomic analysis pipeline.
"""
import argparse
import logging
from pathlib import Path

from data.api import ClinVarClient
from data.variant_processor import VariantProcessor
from data.filters import VariantFilter, VariantSummarizer, FilterStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Genomic variant analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--max-variants",
        type=int,
        default=3000,
        help="Maximum number of variants to fetch (default: 3000)"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum variant length in bp (default: 50)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache"),
        help="Directory for caching API results"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of API results"
    )
    
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train pathogenicity prediction model"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for processed variants (JSON)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main entry point for CLI."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    
    logger.info("Starting genomic variant analysis pipeline")
    
    # Initialize components
    dbvar_client = ClinVarClient(
        cache_dir=None if args.no_cache else args.cache_dir
    )
    
    variant_processor = VariantProcessor()
    variant_filter = VariantFilter()
    
    # Fetch variants
    logger.info(f"Fetching up to {args.max_variants} variants from dbVar")
    raw_variants = dbvar_client.fetch_str_variants(
        max_results=args.max_variants,
        use_cache=not args.no_cache
    )
    logger.info(f"Fetched {len(raw_variants)} raw variants")
    
    # Process variants
    logger.info("Processing variants")
    processed_variants = variant_processor.process_variants(raw_variants)
    processed_dicts = variant_processor.to_dict_list(processed_variants)
    logger.info(f"Processed {len(processed_dicts)} variants")
    
    # Filter by length
    logger.info(f"Filtering variants (min_length={args.min_length}bp)")
    filter_result = variant_filter.filter_by_length(
        processed_dicts,
        min_bp=args.min_length,
        unknown_strategy=FilterStrategy.EXCLUDE_UNKNOWN
    )
    logger.info(filter_result.summary())
    
    filtered_variants = filter_result.kept
    
    # Summarize
    logger.info("Generating summary statistics")
    VariantSummarizer.print_summary(filtered_variants)
    
    # Save output if requested
    if args.output:
        import json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(filtered_variants, f, indent=2)
        logger.info(f"Saved processed variants to {args.output}")
    
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()