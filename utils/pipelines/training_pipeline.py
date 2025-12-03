from venv import logger

import numpy as np
from data.api import fetch_clinvar_deletions_entrez
from data.data_processor import pass_through_variants
from data.preprocessing import summarize_variants
from training.model import DeletionPathogenicityPredictor
import json
from pathlib import Path
import pandas as pd

# Import config module and get the configuration values
import config

# Access configuration from config module
CHROMOSOMES = config.CHROMOSOMES
MAX_VARIANTS_PER_CHR = config.MAX_VARIANTS_PER_CHR
TEST_SIZE = config.TEST_SIZE
CV_FOLDS = config.CV_FOLDS
SAVE_OUTPUTS = config.SAVE_OUTPUTS
REFERENCE_FASTA = config.REFERENCE_FASTA
# REMOVED: BALANCE_CLASSES - no longer needed since we always use weighted training


def train_pipeline():
    """Training pipeline: fetch ClinVar data and train pathogenicity prediction model."""
    
    print("="*70)
    print("DELETION PATHOGENICITY PREDICTION - TRAINING PIPELINE")
    print("="*70)
    
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
            print(f"{len(chrom_variants)} variants")
        except Exception as e:
            print(f"Error: {e}")
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
    print(f"Class balancing: Automatic weighted training (always enabled)")
    
    pathogenicity_predictor = DeletionPathogenicityPredictor(threshold=0.40)
    
    try:
        # Train on deletion variants with improved model
        # REMOVED: balance_classes parameter - weighted training is now automatic
        path_results = pathogenicity_predictor.train(
            processed_variants,
            test_size=TEST_SIZE,
            cv_folds=CV_FOLDS
        )
        
        print("\n" + "="*70)
        print("PATHOGENICITY PREDICTOR RESULTS")
        print("="*70)
        
        # Cross-validation results
        print("\nCross-Validation Performance:")
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
        
        # Dataset info
        print("\nDataset Information:")
        print(f"  Training samples:   {path_results.get('n_train', 0)} ({path_results.get('n_pathogenic_train', 0)} pathogenic, {path_results.get('n_benign_train', 0)} benign)")
        print(f"  Test samples:       {path_results.get('n_test', 0)} ({path_results.get('n_pathogenic_test', 0)} pathogenic, {path_results.get('n_benign_test', 0)} benign)")
        print(f"  Features:           {path_results.get('n_features', 0)}")
        print(f"  Imbalance ratio:    {path_results.get('imbalance_ratio', 0):.2f}:1 (pathogenic:benign)")
        print(f"  Class weight (benign):     {path_results.get('class_weight_benign', 0):.3f}")
        print(f"  Class weight (pathogenic): {path_results.get('class_weight_pathogenic', 0):.3f}")
        
        # Model info
        print("\nModel Information:")
        models_used = 'Random Forest + Gradient Boosting + XGBoost' if path_results.get('has_xgboost') else 'Random Forest + Gradient Boosting'
        print(f"  Ensemble:  {models_used}")
        print(f"  Threshold: {pathogenicity_predictor.threshold}")
        
        # Save model results
        if SAVE_OUTPUTS:
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            results_summary = {
                'configuration': {
                    'chromosomes': CHROMOSOMES,
                    'max_variants_per_chr': MAX_VARIANTS_PER_CHR,
                    'test_size': TEST_SIZE,
                    'cv_folds': CV_FOLDS,
                    'threshold': pathogenicity_predictor.threshold,
                    'weighted_training': True  # Always true now
                },
                'cross_validation': {
                    'precision': path_results['cv_precision'],
                    'recall': path_results['cv_recall'],
                    'f1': path_results['cv_f1'],
                    'specificity': path_results['cv_specificity'],
                    'auc': path_results['cv_auc'],
                    'tp': path_results['cv_tp'],
                    'tn': path_results['cv_tn'],
                    'fp': path_results['cv_fp'],
                    'fn': path_results['cv_fn']
                },
                'test_set': {
                    'precision': path_results['test_precision'],
                    'recall': path_results['test_recall'],
                    'f1': path_results['test_f1'],
                    'specificity': path_results['test_specificity'],
                    'auc': path_results['test_auc'],
                    'tp': path_results['test_tp'],
                    'tn': path_results['test_tn'],
                    'fp': path_results['test_fp'],
                    'fn': path_results['test_fn']
                },
                'dataset_info': {
                    'n_features': path_results['n_features'],
                    'n_train': path_results['n_train'],
                    'n_test': path_results['n_test'],
                    'n_pathogenic_train': path_results['n_pathogenic_train'],
                    'n_benign_train': path_results['n_benign_train'],
                    'n_pathogenic_test': path_results['n_pathogenic_test'],
                    'n_benign_test': path_results['n_benign_test'],
                    'imbalance_ratio': path_results['imbalance_ratio'],
                    'class_weight_benign': path_results['class_weight_benign'],
                    'class_weight_pathogenic': path_results['class_weight_pathogenic']
                },
                'model_info': {
                    'has_xgboost': path_results['has_xgboost'],
                    'feature_importances': path_results.get('feature_importances_rf_top15', [])
                }
            }
            
            # Convert to serializable format
            results_summary = convert_to_serializable(results_summary)
            
            with open(output_dir / "pathogenicity_predictor_results.json", "w") as f:
                json.dump(results_summary, f, indent=2)
            print(f"\nSaved pathogenicity predictor results to {output_dir / 'pathogenicity_predictor_results.json'}")
            
            # Also save test predictions CSV
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