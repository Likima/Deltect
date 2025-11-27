import logging
from typing import Dict, Optional

import pysam
from data.ref_genome_data import _HAS_PYSAM

logger = logging.getLogger(__name__)


def pass_through_variants(variants, reference_fasta: Optional[str] = None):
    """
    Normalize ClinVar variants into a flat record per variant.
    Optionally extract sequences from reference genome.
    
    Args:
        variants: List of raw variant dictionaries from ClinVar
        reference_fasta: Path to reference genome FASTA (optional)
        
    Returns:
        List of processed variant dictionaries
    """
    # Open reference genome if provided
    fasta = None
    if reference_fasta and _HAS_PYSAM:
        try:
            fasta = pysam.FastaFile(reference_fasta)
            logger.info(f"Opened reference genome: {reference_fasta}")
        except Exception as e:
            logger.warning(f"Could not open reference genome: {e}")
            fasta = None
    
    processed = []
    
    for v in variants:
        # Handle both dict and list responses from Entrez.read
        if isinstance(v, list):
            if not v:
                continue
            v = v[0] if len(v) == 1 else v
        
        if not isinstance(v, dict):
            continue
        
        # Extract UID - ClinVar uses accession
        uid = v.get("accession") or v.get("accession_version") or "N/A"
        
        # Extract title
        title = v.get("title") or "N/A"
        
        # Extract gene from genes array
        gene = "N/A"
        genes = v.get("genes") or []
        if isinstance(genes, list) and genes:
            first_gene = genes[0]
            if isinstance(first_gene, dict):
                gene = first_gene.get("symbol") or first_gene.get("GeneID") or "N/A"
            elif isinstance(first_gene, str):
                gene = first_gene
        elif isinstance(genes, dict):
            gene = genes.get("symbol") or genes.get("GeneID") or "N/A"
        
        # Extract genomic coordinates from variation_set
        chr_ = "N/A"
        start = "?"
        end = "?"
        assembly = "N/A"
        
        variation_set = v.get("variation_set") or []
        if isinstance(variation_set, list) and variation_set:
            first_variation = variation_set[0]
            variation_loc = first_variation.get("variation_loc") or []
            
            if variation_loc:
                # Sort by assembly preference (GRCh37 first since that's what we have)
                def rank_assembly(loc):
                    asm = str(loc.get("assembly_name") or "").upper()
                    if "GRCH37" in asm or "HG19" in asm:
                        return 0
                    if "GRCH38" in asm or "HG38" in asm:
                        return 1
                    return 2
                
                best_location = sorted(variation_loc, key=rank_assembly)[0]
                chr_ = str(best_location.get("chr") or "N/A")
                
                start_val = best_location.get("start")
                end_val = best_location.get("stop")
                
                start = str(start_val) if start_val is not None else "?"
                end = str(end_val) if end_val is not None else "?"
                assembly = str(best_location.get("assembly_name") or "N/A")
        
        # Extract clinical significance from germline_classification
        clinical_significance = "N/A"
        germline_class = v.get("germline_classification") or {}
        if isinstance(germline_class, dict):
            clinical_significance = germline_class.get("description") or "N/A"
        
        # Extract variant type from variation_set
        variant_type = "N/A"
        if isinstance(variation_set, list) and variation_set:
            first_variation = variation_set[0]
            variant_type = first_variation.get("variant_type") or "N/A"
        
        # Extract review status from germline_classification
        review_status = "N/A"
        if isinstance(germline_class, dict):
            review_status = germline_class.get("review_status") or "N/A"
        
        # Extract condition from trait_set
        condition = "N/A"
        trait_set = germline_class.get("trait_set") or [] if isinstance(germline_class, dict) else []
        if isinstance(trait_set, list) and trait_set:
            first_trait = trait_set[0]
            if isinstance(first_trait, dict):
                condition = first_trait.get("trait_name") or "N/A"
            elif isinstance(first_trait, str):
                condition = first_trait
        elif isinstance(trait_set, dict):
            condition = trait_set.get("trait_name") or "N/A"
        
        # Extract consequence from molecular_consequence_list
        consequence = "N/A"
        molecular_consequences = v.get("molecular_consequence_list") or []
        if isinstance(molecular_consequences, list) and molecular_consequences:
            consequence = molecular_consequences[0]
        elif isinstance(molecular_consequences, str):
            consequence = molecular_consequences
        
        # Extract sequence from reference genome if available
        sequence = None
        length = None
        
        if fasta and start != "?" and end != "?" and chr_ != "N/A":
            try:
                start_int = int(start)
                end_int = int(end)
                length = end_int - start_int
                
                # Normalize chromosome name (add 'chr' prefix if needed)
                chrom_variants = [chr_, f"chr{chr_}"]
                
                for chrom_name in chrom_variants:
                    if chrom_name in fasta.references:
                        sequence = fasta.fetch(chrom_name, start_int, end_int)
                        break
                
                if not sequence:
                    logger.debug(f"Could not find chromosome {chr_} in reference")
            except Exception as e:
                logger.debug(f"Could not extract sequence for {uid}: {e}")
        
        variant_record = {
            "uid": uid,
            "gene": gene,
            "title": title,
            "chr": chr_,
            "start": start,
            "end": end,
            "assembly": assembly,
            "variant_type": variant_type,
            "clinical_significance": clinical_significance,
            "review_status": review_status,
            "condition": condition,
            "consequence": consequence
        }
        
        # Add sequence if extracted
        if sequence:
            variant_record["sequence"] = sequence
            variant_record["length"] = length
        
        processed.append(variant_record)
    
    # Close reference genome
    if fasta:
        fasta.close()
    
    return processed
