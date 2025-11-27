from venv import logger
from extraction.deletion_extraction import DeletionExtractor
from training.model import DeletionPathogenicityPredictor
from utils.conv import convert_to_json_serializable
from utils.pipelines.training_pipeline import train_pipeline

import json
import logging
from pathlib import Path
import numpy as np


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
        # if truth_vcf:
        #     print(f"\n[3/3] Validating predictions with Truvari...")
        #     validate_with_truvari(
        #         predicted_deletions=results,
        #         truth_vcf=truth_vcf,
        #         reference_fasta=reference_fasta,
        #         output_dir=output_dir / "validation"
        #     )
        # else:
        #     print("\n[3/3] Skipping validation (no truth VCF provided)")
        #     print("  Use --truth-vcf to enable Truvari validation")
        
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