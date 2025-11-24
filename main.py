"""
Main pipeline for fetching and processing genomic variants from ClinVar,
and training a deletion pathogenicity prediction model.
"""
from data.api import fetch_clinvar_deletions_entrez
from data.data_processor import pass_through_variants
from data.preprocessing import summarize_variants
from extraction.deletion_extraction import DeletionExtractor
from training.model import DeletionPathogenicityPredictor
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_pipeline():
    """Training pipeline: fetch ClinVar data and train pathogenicity prediction model."""
    
    print("="*70)
    print("DELETION PATHOGENICITY PREDICTION - TRAINING PIPELINE")
    print("="*70)
    
    # Configuration
    CHROMOSOME = "3" # arbitrary
    MAX_VARIANTS = 3000
    TEST_SIZE = 0.2
    CV_FOLDS = 10
    SAVE_OUTPUTS = True
    REFERENCE_FASTA = "hs37d5.fa"
    
    # Step 1: Fetch deletion variants from ClinVar
    print(f"\n[1/4] Fetching deletion variants from ClinVar...")
    print(f"Chromosome: {CHROMOSOME}")
    print(f"Maximum variants: {MAX_VARIANTS}")
    
    raw_variants = fetch_clinvar_deletions_entrez(chrom=CHROMOSOME, max_results=MAX_VARIANTS)
    
    print(f"Received {len(raw_variants)} raw deletion variants from ClinVar")
    
    if not raw_variants:
        print("ERROR: No variants fetched. Exiting.")
        return None
    
    # Save raw variants if requested
    output_dir = Path("output")
    if SAVE_OUTPUTS:
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "raw_variants.json", "w") as f:
            json.dump(raw_variants, f, indent=2, default=str)
        print(f"Saved raw variants to {output_dir / 'raw_variants.json'}")
    
    # Step 2: Process variants
    print("\n[2/4] Processing and normalizing variants...")
    
    # Pass reference genome to extract sequences
    processed_variants = pass_through_variants(raw_variants, reference_fasta=REFERENCE_FASTA)
    
    print(f"Processed {len(processed_variants)} variants")
    
    if not processed_variants:
        print("ERROR: No variants after processing. Exiting.")
        return None
    
    # Save processed variants
    if SAVE_OUTPUTS:
        with open(output_dir / "processed_variants.json", "w") as f:
            json.dump(processed_variants, f, indent=2, default=str)
        print(f"Saved processed variants to {output_dir / 'processed_variants.json'}")
    
    # Step 3: Analyze clinical significance distribution
    print("\n[3/4] Analyzing clinical significance distribution...")
    print("-" * 70)
    summarize_variants(processed_variants)
    print("-" * 70)

    # Step 4: Train pathogenicity predictor
    print(f"\n[4/4] Training pathogenicity predictor...")
    print(f"Test set size: {TEST_SIZE*100}%")
    print(f"Cross-validation folds: {CV_FOLDS}")
    
    pathogenicity_predictor = DeletionPathogenicityPredictor(threshold=0.5)
    
    try:
        # Train on deletion variants
        path_results = pathogenicity_predictor.train(
            processed_variants,
            test_size=TEST_SIZE,
            cv_folds=CV_FOLDS
        )
        
        print("\nPathogenicity Predictor Results:")
        print(f"  Test MSE:         {path_results['mse']:.4f}")
        print(f"  Test Precision:   {path_results['precision']:.4f}")
        print(f"  Test Recall:      {path_results['recall']:.4f}")
        print(f"  Test Specificity: {path_results['specificity']:.4f}")
        
        # Save model results
        if SAVE_OUTPUTS:
            pathogenic_count = sum(1 for v in processed_variants if "pathogenic" in v.get("clinical_significance", "").lower())
            benign_count = len(processed_variants) - pathogenic_count
            
            train_pathogenic = sum(1 for pred in path_results['y_train_pred'] if pred >= pathogenicity_predictor.threshold)
            test_pathogenic = sum(1 for y in path_results['y_test'] if y >= pathogenicity_predictor.threshold)
            
            results_summary = {
                'configuration': {
                    'chromosome': CHROMOSOME,
                    'max_variants': MAX_VARIANTS,
                    'test_size': TEST_SIZE,
                    'cv_folds': CV_FOLDS,
                    'threshold': pathogenicity_predictor.threshold
                },
                'cross_validation': {
                    'mse_mean': float(path_results['cv_mse_mean']),
                    'mse_std': float(path_results['cv_mse_std']),
                    'precision_mean': float(path_results['cv_precision_mean']),
                    'precision_std': float(path_results['cv_precision_std']),
                    'recall_mean': float(path_results['cv_recall_mean']),
                    'recall_std': float(path_results['cv_recall_std']),
                    'specificity_mean': float(path_results['cv_specificity_mean']),
                    'specificity_std': float(path_results['cv_specificity_std'])
                },
                'test_set': {
                    'mse': float(path_results['mse']),
                    'precision': float(path_results['precision']),
                    'recall': float(path_results['recall']),
                    'specificity': float(path_results['specificity'])
                },
                'dataset_info': {
                    'total_variants': len(processed_variants),
                    'pathogenic_count': pathogenic_count,
                    'benign_count': benign_count,
                    'pathogenic_ratio': pathogenic_count / len(processed_variants) if processed_variants else 0,
                    'training_samples': len(path_results['y_train_pred']),
                    'training_pathogenic': train_pathogenic,
                    'training_benign': len(path_results['y_train_pred']) - train_pathogenic,
                    'test_samples': len(path_results['y_test']),
                    'test_pathogenic': test_pathogenic,
                    'test_benign': len(path_results['y_test']) - test_pathogenic
                }
            }
            
            with open(output_dir / "pathogenicity_predictor_results.json", "w") as f:
                json.dump(results_summary, f, indent=2)
            print(f"Saved pathogenicity predictor results to {output_dir / 'pathogenicity_predictor_results.json'}")
        
        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETE")
        print("="*70)
        
        # Return trained model
        return pathogenicity_predictor
        
    except ValueError as e:
        print("\n" + "="*70)
        print("ERROR: Model Training Failed")
        print("="*70)
        print(f"Error: {e}")
        logger.error(f"Training error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def inference_pipeline(
    bam_file: str,
    chromosome: str,
    start: int,
    end: int,
    pathogenicity_predictor: DeletionPathogenicityPredictor,
    reference_fasta: str = "hs37d5.fa",
    gene_annotation_gtf: str = None,
    output_dir: Path = Path("output"),
    min_deletion_size: int = 1,
    min_mapping_quality: int = 20
):
    """Inference pipeline: analyze BAM file region for pathogenic deletions.
    
    Args:
        bam_file: Path to BAM file
        chromosome: Chromosome name
        start: Start position
        end: End position
        pathogenicity_predictor: Trained pathogenicity predictor
        reference_fasta: Path to reference genome
        gene_annotation_gtf: Path to GTF gene annotation file (for enriching deletions)
        output_dir: Output directory for results
        min_deletion_size: Minimum deletion size to consider (default: 1bp)
        min_mapping_quality: Minimum mapping quality for reads (default: 20)
    """
    
    print("="*70)
    print("DELETION PATHOGENICITY PREDICTION - INFERENCE PIPELINE")
    print("="*70)
    
    print(f"\n[1/2] Extracting deletions from BAM file using CIGAR strings...")
    print(f"Region: {chromosome}:{start}-{end}")
    print(f"Filters: min_size={min_deletion_size}bp, min_mapq={min_mapping_quality}")
    
    if gene_annotation_gtf:
        print(f"Gene annotations: {gene_annotation_gtf}")
    else:
        print("WARNING: No gene annotations provided - deletions will have gene='N/A'")
        print("         This may reduce prediction accuracy. Use --gtf to provide gene annotations.")
    
    # Step 1: Extract deletions directly from BAM using DeletionExtractor
    try:
        deletion_extractor = DeletionExtractor(
            reference_fasta=reference_fasta,
            gene_annotation_gtf=gene_annotation_gtf
        )
        
        # Extract all deletions from the region
        deletions = deletion_extractor.extract_deletions_from_region(
            bam_file=bam_file,
            chromosome=chromosome,
            start=start,
            end=end,
            min_deletion_length=min_deletion_size,
            min_mapping_quality=min_mapping_quality,
            skip_duplicates=True,
            skip_secondary=True,
            skip_supplementary=True,
            annotate_genes=True  # Annotate with genes if GTF provided
        )
        
        print(f"Extracted {len(deletions)} deletion variants from CIGAR strings")
        
        if deletions and gene_annotation_gtf:
            annotated_count = sum(1 for d in deletions if d.gene and d.gene != 'N/A')
            print(f"  Annotated with genes: {annotated_count}/{len(deletions)} ({annotated_count/len(deletions)*100:.1f}%)")
        
        if not deletions:
            print("No deletions found in BAM reads")
            return []
            
    except Exception as e:
        print(f"ERROR extracting deletions: {e}")
        logger.error(f"Deletion extraction error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return []
    
    # Step 2: Convert DeletionVariant objects to format expected by pathogenicity predictor
    print(f"\n[2/2] Converting deletions and predicting pathogenicity...")
    
    # Use the built-in to_variant_dict() method
    candidate_deletions = [deletion.to_variant_dict() for deletion in deletions]
    
    print(f"Converted {len(candidate_deletions)} deletions to variant format")
    
    if not candidate_deletions:
        print("No candidate deletions for pathogenicity prediction")
        return []
    
    # Save candidate deletions
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "candidate_deletions.json", "w") as f:
        # Convert to JSON-serializable format
        candidate_deletions_serializable = convert_to_json_serializable(candidate_deletions)
        json.dump(candidate_deletions_serializable, f, indent=2)
    print(f"Saved candidates to {output_dir / 'candidate_deletions.json'}")
    
    # Predict pathogenicity for deletion variants
    try:
        pathogenicity_probs = pathogenicity_predictor.predict_proba(candidate_deletions)
        
        results = []
        for variant, path_prob in zip(candidate_deletions, pathogenicity_probs):
            variant['pathogenicity_prob'] = float(path_prob)
            variant['is_pathogenic'] = bool(path_prob >= pathogenicity_predictor.threshold)
            results.append(variant)
        
        # Sort by pathogenicity probability (highest first)
        results.sort(key=lambda x: x['pathogenicity_prob'], reverse=True)
        
        # Display results
        print("\n" + "="*70)
        print("RESULTS: Pathogenic Deletion Predictions")
        print("="*70)
        
        pathogenic_deletions = [r for r in results if r['is_pathogenic']]
        
        print(f"\nDeletion variants extracted: {len(candidate_deletions)}")
        print(f"Predicted pathogenic: {len(pathogenic_deletions)}")
        
        if pathogenic_deletions:
            print(f"\nTop pathogenic deletions:")
            for i, variant in enumerate(pathogenic_deletions[:10], 1):
                print(f"\n{i}. {variant['uid']}")
                print(f"   Gene: {variant.get('gene', 'N/A')}")
                print(f"   Position: chr{variant['chr']}:{variant['start']}-{variant['end']}")
                print(f"   Size: {int(variant['end']) - int(variant['start'])} bp")
                print(f"   Pathogenicity: {variant['pathogenicity_prob']:.3f}")
                print(f"   Consequence: {variant.get('consequence', 'N/A')}")
                print(f"   Condition: {variant.get('condition', 'N/A')}")
        
        # Save results - convert to JSON-serializable format
        results_serializable = convert_to_json_serializable(results)
        
        with open(output_dir / "pathogenic_deletions.json", "w") as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nSaved results to {output_dir / 'pathogenic_deletions.json'}")
        
        print("\n" + "="*70)
        print("INFERENCE PIPELINE COMPLETE")
        print("="*70)
        
        return results
        
    except Exception as e:
        print(f"ERROR predicting pathogenicity: {e}")
        logger.error(f"Pathogenicity prediction error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main entry point - can run training or inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deletion pathogenicity prediction pipeline")
    parser.add_argument('--mode', choices=['train', 'inference', 'both'], default='train',
                       help='Pipeline mode')
    parser.add_argument('--bam', type=str, help='BAM file for inference')
    parser.add_argument('--chr', type=str, default='chr22', help='Chromosome')
    parser.add_argument('--start', type=int, default=1000000, help='Start position')
    parser.add_argument('--end', type=int, default=2000000, help='End position')
    parser.add_argument('--reference', type=str, default='hs37d5.fa', help='Reference genome')
    parser.add_argument('--gtf', type=str, help='Gene annotation GTF file (e.g., gencode.v19.annotation.gtf)')
    parser.add_argument('--min-del-size', type=int, default=1, help='Minimum deletion size (bp)')
    parser.add_argument('--min-mapq', type=int, default=20, help='Minimum mapping quality')
    
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
                min_mapping_quality=args.min_mapq
            )

if __name__ == "__main__":
    main()