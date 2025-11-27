"""
Main pipeline for fetching and processing genomic variants from ClinVar,
and training a deletion pathogenicity prediction model.
"""
from extraction.deletion_extraction import DeletionExtractor
from training.model import DeletionPathogenicityPredictor
import json
import logging
from pathlib import Path
import numpy as np

from utils.pipelines import inference_pipeline
from utils.pipelines.training_pipeline import train_pipeline
from validation.truvari_validator import TruvariValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point - can run training, inference, or validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deletion pathogenicity prediction pipeline")
    parser.add_argument('--mode', choices=['train', 'inference', 'both', 'validate'], 
                       default='train', help='Pipeline mode')
    parser.add_argument('--bam', type=str, help='BAM file for inference')
    parser.add_argument('--chr', type=str, default='chr22', help='Chromosome')
    parser.add_argument('--start', type=int, default=1000000, help='Start position')
    parser.add_argument('--end', type=int, default=2000000, help='End position')
    parser.add_argument('--reference', type=str, default='hs37d5.fa', help='Reference genome')
    parser.add_argument('--gtf', type=str, help='Gene annotation GTF file (e.g., gencode.v19.annotation.gtf)')
    parser.add_argument('--min-del-size', type=int, default=1, help='Minimum deletion size (bp)')
    parser.add_argument('--min-mapq', type=int, default=20, help='Minimum mapping quality')
    
    parser.add_argument('--truth-vcf', type=str, 
                       help='Truth set VCF for Truvari validation (e.g., GIAB, ClinVar)')
    parser.add_argument('--validate-predictions', type=str,
                       help='JSON file with predictions to validate')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        # Run training pipeline
        pathogenicity_predictor = train_pipeline()
        
        if pathogenicity_predictor is None:
            print("\nERROR: Training failed")
            return
        
        if args.mode == 'both':
            # Run inference pipeline
            if not args.bam:
                print("\nWARNING: No BAM file specified for inference mode")
                print("Use --bam <file> to analyze a BAM file")
                return
            
            inference_pipeline(
                bam_file=args.bam,
                chromosome=args.chr,
                start=args.start,
                end=args.end,
                pathogenicity_predictor=pathogenicity_predictor,
                reference_fasta=args.reference,
                gene_annotation_gtf=args.gtf,
                min_deletion_size=args.min_del_size,
                min_mapping_quality=args.min_mapq,
                truth_vcf=args.truth_vcf
            )

            # validate_with_hg002(
            #     pathogenicity_predictor=pathogenicity_predictor,
            #     bam_file=args.bam,
            #     chromosome=args.chr,
            #     start=args.start,
            #     end=args.end,
            #     reference_fasta=args.reference,
            #     gene_annotation_gtf=args.gtf,
            #     min_deletion_size=args.min_del_size,
            #     min_mapping_quality=args.min_mapq
            # )
    
    elif args.mode == 'inference':
        print("ERROR: Inference mode requires a trained model")
        print("Use --mode both to train and then run inference")
        return
    
    elif args.mode == 'validate':
        if not args.truth_vcf:
            print("ERROR: --truth-vcf required for validation mode")
            return
        
        if not args.validate_predictions:
            print("ERROR: --validate-predictions required for validation mode")
            return
        
        # Load predictions
        with open(args.validate_predictions) as f:
            predictions = json.load(f)
        
        # # Run validation
        # validate_with_truvari(
        #     predicted_deletions=predictions,
        #     truth_vcf=args.truth_vcf,
        #     reference_fasta=args.reference
        # )


if __name__ == "__main__":
    main()