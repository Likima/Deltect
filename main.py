"""
Main pipeline for fetching and processing genomic variants from ClinVar,
and training a deletion pathogenicity prediction model.
"""
from typing import Dict, List
from data.api import fetch_clinvar_deletions_entrez
from data.data_processor import pass_through_variants
from data.preprocessing import summarize_variants
from extraction.deletion_extraction import DeletionExtractor
from training.model import DeletionPathogenicityPredictor
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from validation.truvari_validator import TruvariValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_pipeline():
    """Training pipeline: fetch ClinVar data and train pathogenicity prediction model."""
    
    print("="*70)
    print("DELETION PATHOGENICITY PREDICTION - TRAINING PIPELINE")
    print("="*70)
    
    # Configuration
    CHROMOSOMES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]  # Multiple chromosomes
    MAX_VARIANTS_PER_CHR = 1000
    TEST_SIZE = 0.2
    CV_FOLDS = 5  # Reduced from 10 for faster training with larger dataset
    SAVE_OUTPUTS = True
    REFERENCE_FASTA = "hs37d5.fa"
    BALANCE_CLASSES = True
    
    # Step 1: Fetch deletion variants from ClinVar
    print(f"\n[1/4] Fetching deletion variants from ClinVar...")
    print(f"Chromosomes: {', '.join(CHROMOSOMES)}")
    print(f"Maximum variants per chromosome: {MAX_VARIANTS_PER_CHR}")
    
    all_raw_variants = []
    for chrom in CHROMOSOMES:
        print(f"  Fetching chr{chrom}...", end=" ")
        try:
            chrom_variants = fetch_clinvar_deletions_entrez(
                chrom=chrom, 
                max_results=MAX_VARIANTS_PER_CHR
            )
            all_raw_variants.extend(chrom_variants)
            print(f"✓ {len(chrom_variants)} variants")
        except Exception as e:
            print(f"✗ Error: {e}")
            logger.warning(f"Failed to fetch chr{chrom}: {e}")
    
    print(f"\nReceived {len(all_raw_variants)} total deletion variants from ClinVar")
    
    if not all_raw_variants:
        print("ERROR: No variants fetched. Exiting.")
        return None
    
    # Save raw variants if requested
    output_dir = Path("output")
    if SAVE_OUTPUTS:
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "raw_variants.json", "w") as f:
            json.dump(all_raw_variants, f, indent=2, default=str)
        print(f"Saved raw variants to {output_dir / 'raw_variants.json'}")
    
    # Step 2: Process variants
    print("\n[2/4] Processing and normalizing variants...")
    
    # Pass reference genome to extract sequences
    processed_variants = pass_through_variants(all_raw_variants, reference_fasta=REFERENCE_FASTA)
    
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
    print(f"Class balancing: {'Enabled' if BALANCE_CLASSES else 'Disabled'}")
    
    pathogenicity_predictor = DeletionPathogenicityPredictor(threshold=0.5)
    
    try:
        # Train on deletion variants with improved model
        path_results = pathogenicity_predictor.train(
            processed_variants,
            test_size=TEST_SIZE,
            cv_folds=CV_FOLDS,
            balance_classes=BALANCE_CLASSES
        )
        
        print("\n" + "="*70)
        print("PATHOGENICITY PREDICTOR RESULTS")
        print("="*70)
        
        # Cross-validation results
        print("\nCross-Validation Performance (Mean ± Std):")
        print(f"  Precision:    {path_results.get('cv_precision', 0):.4f}")
        print(f"  Recall:       {path_results.get('cv_recall', 0):.4f}")
        print(f"  F1 Score:     {path_results.get('cv_f1', 0):.4f}")
        print(f"  Specificity:  {path_results.get('cv_specificity', 0):.4f}")
        print(f"  AUC-ROC:      {path_results.get('cv_auc', 0):.4f}")
        
        # Test set results
        print("\nTest Set Performance:")
        print(f"  Precision:    {path_results.get('test_precision', 0):.4f}")
        print(f"  Recall:       {path_results.get('test_recall', 0):.4f}")
        print(f"  F1 Score:     {path_results.get('test_f1', 0):.4f}")
        print(f"  Specificity:  {path_results.get('test_specificity', 0):.4f}")
        print(f"  AUC-ROC:      {path_results.get('test_auc', 0):.4f}")
        
        # Confusion matrix
        print("\nTest Set Confusion Matrix:")
        print(f"  True Positives:  {path_results.get('test_tp', 0)}")
        print(f"  True Negatives:  {path_results.get('test_tn', 0)}")
        print(f"  False Positives: {path_results.get('test_fp', 0)}")
        print(f"  False Negatives: {path_results.get('test_fn', 0)}")
        
        # Dataset statistics
        print("\nDataset Information:")
        print(f"  Total variants:     {len(processed_variants)}")
        print(f"  Training samples:   {path_results.get('n_train', 0)}")
        print(f"  Test samples:       {path_results.get('n_test', 0)}")
        print(f"  Number of features: {path_results.get('n_features', 0)}")
        
        # Save comprehensive model results
        if SAVE_OUTPUTS:
            # Calculate dataset statistics
            pathogenic_count = sum(
                1 for v in processed_variants 
                if 'pathogenic' in v.get('clinical_significance', '').lower() 
                and 'benign' not in v.get('clinical_significance', '').lower()
            )
            benign_count = len(processed_variants) - pathogenic_count
            
            results_summary = {
                'configuration': {
                    'chromosomes': CHROMOSOMES,
                    'max_variants_per_chr': MAX_VARIANTS_PER_CHR,
                    'test_size': TEST_SIZE,
                    'cv_folds': CV_FOLDS,
                    'threshold': pathogenicity_predictor.threshold,
                    'balance_classes': BALANCE_CLASSES,
                    'reference_fasta': REFERENCE_FASTA
                },
                'cross_validation': {
                    'precision': float(path_results.get('cv_precision', 0)),
                    'recall': float(path_results.get('cv_recall', 0)),
                    'f1_score': float(path_results.get('cv_f1', 0)),
                    'specificity': float(path_results.get('cv_specificity', 0)),
                    'auc_roc': float(path_results.get('cv_auc', 0))
                },
                'test_set': {
                    'precision': float(path_results.get('test_precision', 0)),
                    'recall': float(path_results.get('test_recall', 0)),
                    'f1_score': float(path_results.get('test_f1', 0)),
                    'specificity': float(path_results.get('test_specificity', 0)),
                    'auc_roc': float(path_results.get('test_auc', 0)),
                    'confusion_matrix': {
                        'true_positives': int(path_results.get('test_tp', 0)),
                        'true_negatives': int(path_results.get('test_tn', 0)),
                        'false_positives': int(path_results.get('test_fp', 0)),
                        'false_negatives': int(path_results.get('test_fn', 0))
                    }
                },
                'dataset_info': {
                    'total_variants': len(processed_variants),
                    'raw_variants_fetched': len(all_raw_variants),
                    'pathogenic_count': pathogenic_count,
                    'benign_count': benign_count,
                    'pathogenic_ratio': pathogenic_count / len(processed_variants) if processed_variants else 0,
                    'training_samples': int(path_results.get('n_train', 0)),
                    'test_samples': int(path_results.get('n_test', 0)),
                    'n_features': int(path_results.get('n_features', 0))
                },
                'feature_importance': path_results.get('feature_importance', [])
            }
            
            # Save to JSON
            with open(output_dir / "pathogenicity_predictor_results.json", "w") as f:
                json.dump(results_summary, f, indent=2)
            print(f"\nSaved comprehensive results to {output_dir / 'pathogenicity_predictor_results.json'}")
            
            # Also save a CSV with predictions for analysis
            if 'y_test_proba' in path_results and 'y_test' in path_results:
                predictions_df = pd.DataFrame({
                    'true_label': path_results['y_test'],
                    'predicted_prob': path_results['y_test_proba'],
                    'predicted_label': path_results['y_test_pred']
                })
                predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)
                print(f"Saved test predictions to {output_dir / 'test_predictions.csv'}")
        
        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETE")
        print("="*70)
        
        # Print performance summary
        test_f1 = path_results.get('test_f1', 0)
        test_auc = path_results.get('test_auc', 0)
        
        if test_f1 >= 0.8 and test_auc >= 0.85:
            print("\n✓ Model performance: EXCELLENT")
        elif test_f1 >= 0.6 and test_auc >= 0.7:
            print("\n✓ Model performance: GOOD")
        elif test_f1 >= 0.4 and test_auc >= 0.6:
            print("\n⚠ Model performance: ACCEPTABLE")
        else:
            print("\n✗ Model performance: NEEDS IMPROVEMENT")
            print("  Consider:")
            print("  - Fetching more variants (increase MAX_VARIANTS_PER_CHR)")
            print("  - Adding more chromosomes")
            print("  - Checking data quality and feature engineering")
        
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
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR: Unexpected Training Error")
        print("="*70)
        print(f"Error: {e}")
        logger.error(f"Unexpected training error: {e}", exc_info=True)
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
    min_mapping_quality: int = 20,
    truth_vcf: str = None
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
        truth_vcf: Path to truth set VCF for Truvari validation (optional)
    """
    
    print("="*70)
    print("DELETION PATHOGENICITY PREDICTION - INFERENCE PIPELINE")
    print("="*70)
    
    print(f"\n[1/3] Extracting deletions from BAM file using CIGAR strings...")
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
            annotate_genes=True
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
    print(f"\n[2/3] Converting deletions and predicting pathogenicity...")
    
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
        
        # Step 3: Validate with Truvari if truth VCF provided
        if truth_vcf:
            print(f"\n[3/3] Validating predictions with Truvari...")
            validate_with_truvari(
                predicted_deletions=results,
                truth_vcf=truth_vcf,
                reference_fasta=reference_fasta,
                output_dir=output_dir / "validation"
            )
        else:
            print("\n[3/3] Skipping validation (no truth VCF provided)")
            print("  Use --truth-vcf to enable Truvari validation")
        
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


def validate_with_truvari(
    predicted_deletions: List[Dict],
    truth_vcf: str,
    reference_fasta: str = "hs37d5.fa",
    output_dir: Path = Path("output/validation")
) -> Dict:
    """
    Validate deletion predictions using Truvari against a truth set.
    
    Args:
        predicted_deletions: List of predicted deletion dictionaries
        truth_vcf: Path to truth set VCF (e.g., ClinVar, GIAB)
        reference_fasta: Path to reference genome
        output_dir: Output directory for validation results
        
    Returns:
        Validation metrics dictionary
    """
    print("\n" + "="*70)
    print("TRUVARI VALIDATION")
    print("="*70)
    
    validator = TruvariValidator(reference_fasta=reference_fasta)
    
    try:
        metrics = validator.validate_predictions(
            predicted_deletions=predicted_deletions,
            truth_vcf=Path(truth_vcf),
            output_dir=output_dir
        )
        
        print("\nTruvari Bench Results:")
        print(f"  Precision: {metrics['truvari_bench'].get('precision', 0):.4f}")
        print(f"  Recall: {metrics['truvari_bench'].get('recall', 0):.4f}")
        print(f"  F1 Score: {metrics['truvari_bench'].get('f1', 0):.4f}")
        
        print("\nPathogenicity Analysis:")
        path_metrics = metrics['pathogenicity_analysis']
        print(f"  True Positives: {path_metrics['true_positives']}")
        print(f"  False Positives: {path_metrics['false_positives']}")
        print(f"  False Negatives: {path_metrics['false_negatives']}")
        print(f"  Pathogenic TP: {path_metrics['pathogenic_tp']}")
        print(f"  Pathogenic FP: {path_metrics['pathogenic_fp']}")
        
        return metrics
        
    except Exception as e:
        print(f"ERROR during Truvari validation: {e}")
        logger.error(f"Truvari validation error: {e}", exc_info=True)
        return {}


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
        
        # Run validation
        validate_with_truvari(
            predicted_deletions=predictions,
            truth_vcf=args.truth_vcf,
            reference_fasta=args.reference
        )


if __name__ == "__main__":
    main()