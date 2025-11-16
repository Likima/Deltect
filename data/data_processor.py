"""
Process and normalize dbVar variants, extracting sequences from reference genome.
"""
from typing import List, Dict, Optional
import logging

try:
    import pysam
    _HAS_PYSAM = True
except ImportError:
    _HAS_PYSAM = False

logger = logging.getLogger(__name__)


def pass_through_variants(variants, reference_fasta: Optional[str] = None):
    """
    Normalize dbVar variants into a flat record per variant.
    Optionally extract sequences from reference genome.
    
    Args:
        variants: List of raw variant dictionaries from dbVar
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
        
        # Extract UID
        uid = v.get("uid") or v.get("Id") or v.get("SV") or "N/A"
        
        # Extract title
        title = v.get("title") or v.get("Title") or "N/A"
        
        # Extract gene
        gene = "N/A"
        gene_list = v.get("dbvargenelist") or v.get("dbVarGeneList") or []
        if isinstance(gene_list, list) and gene_list:
            first_gene = gene_list[0]
            if isinstance(first_gene, dict):
                gene = first_gene.get("name") or first_gene.get("Name") or "N/A"
            elif isinstance(first_gene, str):
                gene = first_gene
        
        # Extract genomic coordinates
        chr_ = "N/A"
        start = "?"
        end = "?"
        assembly = "N/A"
        
        placements = v.get("dbvarplacementlist") or v.get("dbVarPlacementList") or []
        if isinstance(placements, dict):
            placements = [placements]
        
        if placements:
            # Sort by assembly preference (GRCh37 first since that's what we have)
            def rank_assembly(p):
                asm = str(p.get("Assembly") or p.get("assembly") or "").upper()
                if "GRCH37" in asm or "HG19" in asm:
                    return 0
                if "GRCH38" in asm or "HG38" in asm:
                    return 1
                return 2
            
            best_placement = sorted(placements, key=rank_assembly)[0]
            chr_ = str(best_placement.get("Chr") or best_placement.get("chr") or "N/A")
            
            start_val = (best_placement.get("Chr_start") or 
                        best_placement.get("chr_start") or 
                        best_placement.get("start"))
            end_val = (best_placement.get("Chr_end") or 
                      best_placement.get("chr_end") or 
                      best_placement.get("end"))
            
            start = str(start_val) if start_val is not None else "?"
            end = str(end_val) if end_val is not None else "?"
            assembly = str(best_placement.get("Assembly") or best_placement.get("assembly") or "N/A")
        
        # Extract clinical significance
        clin_sig = v.get("dbvarclinicalsignificancelist") or v.get("dbVarClinicalSignificanceList")
        if clin_sig in (None, [], {}):
            clinical_significance = "N/A"
        else:
            clinical_significance = str(clin_sig)
        
        # Extract variant type
        var_type_list = v.get("dbvarvarianttypelist") or v.get("dbVarVariantTypeList") or []
        variant_type = var_type_list[0] if isinstance(var_type_list, list) and var_type_list else "N/A"
        
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
            "review_status": v.get("review_status", "N/A"),
            "condition": v.get("condition", "N/A"),
            "consequence": v.get("consequence", "N/A")
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

