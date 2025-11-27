from venv import logger
from data.api import fetch_clinvar_deletions_entrez
from data.data_processor import pass_through_variants
from data.preprocessing import summarize_variants
from training.model import DeletionPathogenicityPredictor
import json
from pathlib import Path
import pandas as pd

import config

# import hyperparameters
CHROMOSOMES = config.CHROMOSOMES
MAX_VARIANTS_PER_CHR = config.MAX_VARIANTS_PER_CHR
TEST_SIZE = config.TEST_SIZE
CV_FOLDS = config.CV_FOLDS
SAVE_OUTPUTS = config.SAVE_OUTPUTS
REFERENCE_FASTA = config.REFERENCE_FASTA
BALANCE_CLASSES = config.BALANCE_CLASSES

def train_pipeline():
    """Training pipeline: fetch ClinVar data and train pathogenicity prediction model."""
    
    print("="*70)
    print("DELETION PATHOGENICITY PREDICTION - TRAINING PIPELINE")
    print("="*70)
    
    # Configuration
    # CHROMOSOMES = ["2"]
    # CHROMOSOMES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    
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
            print(f" {len(chrom_variants)} variants")
        except Exception as e:
            print(f" Error: {e}")
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
            balance_classes=False
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