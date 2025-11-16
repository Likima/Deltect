"""
Main pipeline for fetching and processing genomic variants from ClinVar,
and training a deletion pathogenicity prediction model.
"""
from data.api import fetch_clinvar_deletions_entrez
from data.data_processor import pass_through_variants
from data.preprocessing import summarize_variants
from data.ref_genome_data import ReferenceGenomeSampler
from training.model import DeletionPathogenicityPredictor
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution."""
    
    print("="*70)
    print("DELETION PATHOGENICITY PREDICTION PIPELINE")
    print("="*70)
    
    # Configuration
    CHROMOSOME = "22"
    MAX_VARIANTS = 3000
    TEST_SIZE = 0.2
    CV_FOLDS = 10
    SAVE_OUTPUTS = True
    BUILD_BALANCED_DATASET = True
    REFERENCE_FASTA = "hs37d5.fa"
    
    # Step 1: Fetch deletion variants from ClinVar
    print(f"\n[1/5] Fetching deletion variants from ClinVar...")
    print(f"Chromosome: {CHROMOSOME}")
    print(f"Maximum variants: {MAX_VARIANTS}")
    
    raw_variants = fetch_clinvar_deletions_entrez(chrom=CHROMOSOME, max_results=MAX_VARIANTS)
    
    print(f"Received {len(raw_variants)} raw deletion variants from ClinVar")
    
    if not raw_variants:
        print("ERROR: No variants fetched. Exiting.")
        return
    
    # Save raw variants if requested
    if SAVE_OUTPUTS:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "raw_variants.json", "w") as f:
            json.dump(raw_variants, f, indent=2, default=str)
        print(f"Saved raw variants to {output_dir / 'raw_variants.json'}")
    
    # Step 2: Process variants
    # Step 2: Process variants
    print("\n[2/5] Processing and normalizing variants...")
    
    # Pass reference genome to extract sequences
    processed_variants = pass_through_variants(raw_variants, reference_fasta=REFERENCE_FASTA)
    
    print(f"Processed {len(processed_variants)} variants")
    
    # Save processed variants
    if SAVE_OUTPUTS:
        with open(output_dir / "processed_variants.json", "w") as f:
            json.dump(processed_variants, f, indent=2, default=str)
        print(f"Saved processed variants to {output_dir / 'processed_variants.json'}")
    
    # Step 3: Analyze clinical significance distribution
    print("\n[3/5] Analyzing clinical significance distribution...")
    print("-" * 70)
    summarize_variants(processed_variants)
    print("-" * 70)

    # Step 4: Sample normal reference sequences
    normal_sequences = []
    
    if BUILD_BALANCED_DATASET:
        print(f"\n[4/5] Sampling normal reference sequences...")
        print(f"Reference genome: {REFERENCE_FASTA}")
        
        try:
            if not Path(REFERENCE_FASTA).exists():
                print(f"WARNING: Reference genome not found: {REFERENCE_FASTA}")
                print("Skipping reference genome sampling.")
            else:
                # Initialize reference sampler
                sampler = ReferenceGenomeSampler(
                    reference_fasta=REFERENCE_FASTA,
                    chromosomes=None
                )
                
                # Match STR distribution
                num_normal_samples = len(processed_variants)
                print(f"Sampling {num_normal_samples} normal sequences to match STR distribution...")
                
                normal_sequences = sampler.match_str_distribution(
                    str_variants=processed_variants,
                    ratio=1.0
                )
                
                print(f"Sampled {len(normal_sequences)} normal reference sequences")
                
                # Save normal sequences
                with open(output_dir / "normal_sequences.json", "w") as f:
                    json.dump(normal_sequences, f, indent=2)
                print(f"Saved normal sequences to {output_dir / 'normal_sequences.json'}")
                
        except Exception as e:
            print(f"ERROR sampling reference genome: {e}")
            print("Continuing without normal sequences...")
    else:
        print("\n[4/5] Skipping reference genome sampling (BUILD_BALANCED_DATASET=False)")
    
    
    # Step 4: Initialize and train model
    print(f"\n[4/5] Training deletion pathogenicity predictor...")
    print(f"Test set size: {TEST_SIZE*100}%")
    print(f"Cross-validation folds: {CV_FOLDS}")
    
    predictor = DeletionPathogenicityPredictor(threshold=0.5)
    
    try:
        # Train with cross-validation
        results = predictor.train(
            processed_variants + normal_sequences, 
            test_size=TEST_SIZE, 
            cv_folds=CV_FOLDS
        )
        
        # Step 5: Display results
        print("\n[5/5] Model training complete!")
        print("="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        
        print(f"\n{CV_FOLDS}-Fold Cross-Validation (Training Set):")
        print(f"  MSE:         {results['cv_mse_mean']:.4f} (±{results['cv_mse_std']:.4f})")
        print(f"  Precision:   {results['cv_precision_mean']:.4f} (±{results['cv_precision_std']:.4f})")
        print(f"  Recall:      {results['cv_recall_mean']:.4f} (±{results['cv_recall_std']:.4f})")
        print(f"  Specificity: {results['cv_specificity_mean']:.4f} (±{results['cv_specificity_std']:.4f})")
        
        print(f"\nHold-out Test Set:")
        print(f"  MSE:         {results['mse']:.4f}")
        print(f"  Precision:   {results['precision']:.4f}")
        print(f"  Recall:      {results['recall']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")
        
        # Save model results
        if SAVE_OUTPUTS:
            # Calculate dataset statistics
            all_data = processed_variants + normal_sequences
            pathogenic_count = sum(1 for v in all_data if v.get('is_pathogenic', False))
            benign_count = len(all_data) - pathogenic_count
            
            # Get train/test split pathogenicity distribution
            train_pathogenic = sum(1 for pred in results['y_train_pred'] if pred >= predictor.threshold)
            test_pathogenic = sum(1 for y in results['y_test'] if y >= predictor.threshold)
            
            results_summary = {
            'configuration': {
                'chromosome': CHROMOSOME,
                'max_variants': MAX_VARIANTS,
                'test_size': TEST_SIZE,
                'cv_folds': CV_FOLDS,
                'threshold': predictor.threshold
            },
            'cross_validation': {
                'mse_mean': float(results['cv_mse_mean']),
                'mse_std': float(results['cv_mse_std']),
                'precision_mean': float(results['cv_precision_mean']),
                'precision_std': float(results['cv_precision_std']),
                'recall_mean': float(results['cv_recall_mean']),
                'recall_std': float(results['cv_recall_std']),
                'specificity_mean': float(results['cv_specificity_mean']),
                'specificity_std': float(results['cv_specificity_std'])
            },
            'test_set': {
                'mse': float(results['mse']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'specificity': float(results['specificity'])
            },
            'dataset_info': {
                'total_variants': len(processed_variants),
                'total_normal_sequences': len(normal_sequences),
                'total_samples': len(all_data),
                'pathogenic_count': pathogenic_count,
                'benign_count': benign_count,
                'pathogenic_ratio': pathogenic_count / len(all_data) if all_data else 0,
                'training_samples': len(results['y_train_pred']),
                'training_pathogenic': train_pathogenic,
                'training_benign': len(results['y_train_pred']) - train_pathogenic,
                'test_samples': len(results['y_test']),
                'test_pathogenic': test_pathogenic,
                'test_benign': len(results['y_test']) - test_pathogenic
            }
            }
            
            with open(output_dir / "model_results.json", "w") as f:
                json.dump(results_summary, f, indent=2)
            print(f"\nSaved model results to {output_dir / 'model_results.json'}")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        
    except ValueError as e:
        print("\n" + "="*70)
        print("ERROR: Model Training Failed")
        print("="*70)
        print(f"Error: {e}")
        print("Possible causes:")
        print("  - Insufficient balanced data for training")
        print("  - All variants have the same clinical significance")
        print("  - Data quality issues")
        logger.error(f"Training error: {e}", exc_info=True)


if __name__ == "__main__":
    main()