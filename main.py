from data.api import fetch_clinvar_deletions_entrez
from data.data_processor import pass_through_variants
from data.preprocessing import summarize_variants
from training.model import DeletionPathogenicityPredictor

def main():
    raw_variants = fetch_clinvar_deletions_entrez(chrom="22", max_results=3000)

    all_variants = pass_through_variants(raw_variants)

    summarize_variants(all_variants)

    predictor = DeletionPathogenicityPredictor()
    
    try:
        # Train with 10-fold cross-validation
        results = predictor.train(all_variants, test_size=0.2, cv_folds=10)
        
        print("MODEL TRAINING COMPLETE!")
        print(f"\nCross-Validation Results ({results['cv_folds']}-fold):")
        print(f"   MSE: {results['cv_mse_mean']:.4f}")
        print(f"   Precision:  {results['precision']:.4f}")
        print(f"   Recall:  {results['recall']:.4f}")
        print(f"   Specificity:  {results['specificity']:.4f}")
        print(f"\nHold-out Test Set:")
        print(f"   MSE: {results['mse']:.4f}")
        print("\n Generating the Plots and/or Graphs")
        predictor.visualize_results(results)
        print("\n Visulization Task is Complete")
                
    except ValueError as e:
        print(f"\nError: {e}")
        print("   Need more balanced data for training.")

if __name__ == "__main__":
    main()