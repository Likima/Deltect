"""
Validate deletion predictions using Truvari benchmarking.
"""
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pysam
import gzip
import shutil

logger = logging.getLogger(__name__)


class TruvariValidator:
    """Validate deletion predictions against truth sets using Truvari."""
    
    def __init__(self, reference_fasta: str):
        """
        Initialize Truvari validator.
        
        Args:
            reference_fasta: Path to reference genome FASTA
        """
        self.reference_fasta = reference_fasta
        self._check_truvari_installed()
    
    def _check_truvari_installed(self):
        """Check if Truvari is installed."""
        try:
            subprocess.run(['truvari', 'version'], 
                         capture_output=True, check=True)
            logger.info("Truvari is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Truvari is not installed. Install with: pip install truvari"
            )
    
    def deletions_to_vcf(
        self,
        deletions: List[Dict],
        output_vcf: Path,
        sample_name: str = "SAMPLE"
    ) -> Path:
        """
        Convert deletion predictions to VCF format for Truvari.
        
        Args:
            deletions: List of deletion variant dictionaries
            output_vcf: Output VCF file path
            sample_name: Sample name for VCF header
            
        Returns:
            Path to created sorted and indexed VCF file
        """
        logger.info(f"Converting {len(deletions)} deletions to VCF format")
        
        # Create VCF header
        header = pysam.VariantHeader()
        header.add_line('##fileformat=VCFv4.2')
        header.add_line(f'##reference={self.reference_fasta}')
        header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">')
        header.add_line('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Difference in length between REF and ALT alleles">')
        header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">')
        header.add_line('##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">')
        header.add_line('##INFO=<ID=PATHPROB,Number=1,Type=Float,Description="Pathogenicity probability">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        
        # Add contigs from reference
        try:
            with pysam.FastaFile(self.reference_fasta) as fasta:
                for contig in fasta.references:
                    length = fasta.get_reference_length(contig)
                    header.add_line(f'##contig=<ID={contig},length={length}>')
        except Exception as e:
            logger.warning(f"Could not read reference contigs: {e}")
        
        # Add sample
        header.add_sample(sample_name)
        
        # Write VCF records
        records_written = 0
        with pysam.VariantFile(str(output_vcf), 'w', header=header) as vcf_out:
            for deletion in deletions:
                try:
                    chrom = str(deletion.get('chr', '')).replace('chr', '')
                    if not chrom or chrom == 'N/A':
                        continue
                    
                    start = int(deletion.get('start', 0))
                    end = int(deletion.get('end', 0))
                    
                    if start <= 0 or end <= start:
                        continue
                    
                    svlen = -(end - start)  # Negative for deletions
                    
                    # Get reference base at position
                    ref_base = 'N'
                    try:
                        with pysam.FastaFile(self.reference_fasta) as fasta:
                            # Try with and without 'chr' prefix
                            for chr_variant in [chrom, f'chr{chrom}']:
                                if chr_variant in fasta.references:
                                    ref_base = fasta.fetch(chr_variant, start - 1, start).upper()
                                    break
                    except Exception:
                        pass
                    
                    if not ref_base or ref_base == '':
                        ref_base = 'N'
                    
                    # Create VCF record
                    # Use stop parameter instead of setting END in INFO
                    record = vcf_out.new_record(
                        contig=chrom,
                        start=start - 1,  # VCF is 0-based for start
                        stop=end,  # This sets the END field properly
                        alleles=(ref_base, '<DEL>'),  # Symbolic allele
                        id=deletion.get('uid', f'DEL_{chrom}_{start}_{end}')
                    )
                    
                    # Add INFO fields (don't set END manually)
                    record.info['SVTYPE'] = 'DEL'
                    record.info['SVLEN'] = svlen
                    
                    if deletion.get('gene') and deletion.get('gene') != 'N/A':
                        record.info['GENE'] = deletion.get('gene')
                    
                    if 'pathogenicity_prob' in deletion:
                        record.info['PATHPROB'] = float(deletion['pathogenicity_prob'])
                    
                    # Add genotype (assume heterozygous)
                    record.samples[sample_name]['GT'] = (0, 1)
                    
                    vcf_out.write(record)
                    records_written += 1
                
                except Exception as e:
                    logger.debug(f"Skipped deletion {deletion.get('uid', 'unknown')}: {e}")
                    continue
        
        logger.info(f"Wrote {records_written} deletions to VCF")
        
        if records_written == 0:
            raise ValueError("No valid deletions were written to VCF")
        
        sorted_vcf = output_vcf.with_suffix('.sorted.vcf.gz')
        
        try:
            result = subprocess.run([
                'bcftools', 'sort',
                '-o', str(sorted_vcf),
                '-O', 'z',
                str(output_vcf)
            ], capture_output=True, text=True, check=True)
            
            logger.info(f"bcftools sort output: {result.stdout}")
            if result.stderr:
                logger.warning(f"bcftools sort warnings: {result.stderr}")
            
            # Index the sorted VCF
            subprocess.run([
                'bcftools', 'index',
                '-t',  # tabix index
                str(sorted_vcf)
            ], check=True, capture_output=True)
            
            logger.info(f"Created sorted and indexed VCF with bcftools: {sorted_vcf}")
            
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            if isinstance(e, subprocess.CalledProcessError):
                logger.warning(f"bcftools failed: {e.stderr}")
            logger.info("bcftools failed, using Python-based sorting")
            # Pure Python fallback - read, sort, and write
            self._sort_and_compress_vcf_python(output_vcf, sorted_vcf)
        
        return sorted_vcf
    
    def _sort_and_compress_vcf_python(self, input_vcf: Path, output_vcf: Path):
        """
        Sort and compress VCF file using pure Python (no external tools).
        
        Args:
            input_vcf: Input VCF file
            output_vcf: Output compressed VCF file
        """
        logger.info("Sorting and compressing VCF with Python")
        
        # Read all records
        records = []
        header = None
        
        with pysam.VariantFile(str(input_vcf)) as vcf_in:
            header = vcf_in.header
            for record in vcf_in:
                records.append(record)
        
        logger.info(f"Read {len(records)} records, sorting...")
        
        # Sort records by contig and position
        # Create contig order from header
        contig_order = {contig: i for i, contig in enumerate(header.contigs.keys())}
        
        def sort_key(record):
            contig_idx = contig_order.get(record.contig, 999999)
            return (contig_idx, record.start)
        
        records.sort(key=sort_key)
        
        logger.info(f"Writing sorted records to {output_vcf}")
        
        # Write sorted records to compressed VCF
        with pysam.VariantFile(str(output_vcf), 'w', header=header) as vcf_out:
            for record in records:
                vcf_out.write(record)
        
        # Index the compressed VCF
        logger.info(f"Indexing {output_vcf}")
        try:
            pysam.tabix_index(str(output_vcf), preset='vcf', force=True)
            logger.info(f"Created sorted and indexed VCF: {output_vcf}")
        except Exception as e:
            logger.warning(f"Could not create tabix index: {e}")
            logger.info("VCF may still work with Truvari without index")
    
    def run_truvari_bench(
        self,
        predicted_vcf: Path,
        truth_vcf: Path,
        output_dir: Path,
        size_min: int = 1,
        size_max: int = 1000000,
        pctsize: float = 0.7,
        pctseq: float = 0.7,
        pctovl: float = 0.0,
        typeignore: bool = False
    ) -> Dict:
        """
        Run Truvari bench to compare predictions against truth set.
        
        Args:
            predicted_vcf: VCF with predicted deletions
            truth_vcf: VCF with truth set deletions (e.g., from ClinVar, GIAB)
            output_dir: Output directory for Truvari results
            size_min: Minimum SV size to consider
            size_max: Maximum SV size to consider
            pctsize: Min percent size similarity (default: 0.7)
            pctseq: Min percent sequence similarity (default: 0.7)
            pctovl: Min percent reciprocal overlap (default: 0.0)
            typeignore: Ignore variant type matching
            
        Returns:
            Dictionary with Truvari metrics
        """
        # Remove existing output directory if it exists
        output_dir = Path(output_dir)
        if output_dir.exists():
            logger.info(f"Removing existing Truvari output directory: {output_dir}")
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                logger.error(f"Failed to remove existing directory: {e}")
                try:
                    import os
                    os.system(f'rm -rf "{output_dir}"')
                except:
                    pass
        
        # Split multi-allelic variants in truth VCF if needed
        truth_vcf = Path(truth_vcf)
        split_truth_vcf = output_dir.parent / f"{truth_vcf.stem}_split.vcf.gz"
        
        if not split_truth_vcf.exists():
            logger.info(f"Splitting multi-allelic variants in truth VCF: {truth_vcf}")
            try:
                # Try bcftools first
                subprocess.run([
                    'bcftools', 'norm',
                    '-m', '-any',  # Split multi-allelic sites
                    '-o', str(split_truth_vcf),
                    '-O', 'z',
                    str(truth_vcf)
                ], check=True, capture_output=True)
                
                # Index the split VCF
                subprocess.run([
                    'bcftools', 'index', '-t', str(split_truth_vcf)
                ], check=True, capture_output=True)
                
                logger.info(f"Created split truth VCF: {split_truth_vcf}")
                truth_vcf = split_truth_vcf
                
            except FileNotFoundError:
                logger.warning("bcftools not found, trying Python-based splitting")
                # Python fallback using pysam
                try:
                    self._split_multiallelic_vcf(truth_vcf, split_truth_vcf)
                    truth_vcf = split_truth_vcf
                except Exception as e:
                    logger.warning(f"Could not split multi-allelic variants: {e}")
                    logger.info("Proceeding with original truth VCF - may cause errors")
        else:
            logger.info(f"Using existing split truth VCF: {split_truth_vcf}")
            truth_vcf = split_truth_vcf
        
        cmd = [
            'truvari', 'bench',
            '-b', str(truth_vcf),
            '-c', str(predicted_vcf),
            '-o', str(output_dir),
            '-f', str(self.reference_fasta),
            '--sizemin', str(size_min),
            '--sizemax', str(size_max),
            '--pctsize', str(pctsize),
            '--pctseq', str(pctseq),
            '--pctovl', str(pctovl),
        ]
        
        if typeignore:
            cmd.append('--typeignore')
        
        logger.info(f"Running Truvari bench: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Truvari bench completed successfully")
            logger.debug(result.stdout)
            
            # Parse summary.json
            summary_file = output_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    metrics = json.load(f)
                return metrics
            else:
                logger.warning("Truvari summary.json not found")
                return {}
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Truvari bench failed: {e.stderr}")
            raise

    def _split_multiallelic_vcf(self, input_vcf: Path, output_vcf: Path):
        """
        Split multi-allelic variants using pysam (Python fallback).
        
        Args:
            input_vcf: Input VCF with multi-allelic variants
            output_vcf: Output VCF with split variants
        """
        logger.info(f"Splitting multi-allelic variants with Python: {input_vcf}")
        
        with pysam.VariantFile(str(input_vcf)) as vcf_in:
            header = vcf_in.header
            
            with pysam.VariantFile(str(output_vcf), 'w', header=header) as vcf_out:
                for record in vcf_in:
                    # If bi-allelic or no ALT alleles, write as-is
                    if len(record.alts) <= 1:
                        vcf_out.write(record)
                        continue
                    
                    # Split multi-allelic into separate records
                    for i, alt in enumerate(record.alts):
                        # Create new record for each ALT allele
                        new_record = vcf_out.new_record(
                            contig=record.contig,
                            start=record.start,
                            stop=record.stop,
                            alleles=(record.ref, alt),
                            id=record.id,
                            qual=record.qual,
                            filter=record.filter
                        )
                        
                        # Copy INFO fields
                        for key in record.info:
                            try:
                                new_record.info[key] = record.info[key]
                            except:
                                pass
                        
                        # Handle genotypes if present
                        if record.samples:
                            for sample in record.samples:
                                gt = record.samples[sample].get('GT', None)
                                if gt:
                                    # Remap genotype for this allele
                                    # Original GT with allele i+1 becomes 1 in new record
                                    new_gt = tuple(
                                        1 if g == i + 1 else (0 if g == 0 else None)
                                        for g in gt
                                    )
                                    new_record.samples[sample]['GT'] = new_gt
                        
                        vcf_out.write(new_record)
        
        # Index the output
        logger.info(f"Indexing split VCF: {output_vcf}")
        try:
            pysam.tabix_index(str(output_vcf), preset='vcf', force=True)
        except Exception as e:
            logger.warning(f"Could not index split VCF: {e}")

    def calculate_pathogenicity_metrics(
        self,
        truvari_output_dir: Path,
        predicted_deletions: List[Dict],
        pathogenicity_threshold: float = 0.5
    ) -> Dict:
        """
        Calculate pathogenicity prediction metrics from Truvari results.
        
        Args:
            truvari_output_dir: Directory with Truvari bench output
            predicted_deletions: List of predicted deletions with pathogenicity scores
            pathogenicity_threshold: Threshold for pathogenic classification
            
        Returns:
            Dictionary with pathogenicity-specific metrics
        """
        # Load Truvari results
        tp_vcf = truvari_output_dir / 'tp-call.vcf.gz'
        fp_vcf = truvari_output_dir / 'fp.vcf.gz'
        fn_vcf = truvari_output_dir / 'fn.vcf.gz'
        
        metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'pathogenic_tp': 0,
            'pathogenic_fp': 0,
            'benign_tp': 0,
            'benign_fp': 0
        }
        
        # Create ID to pathogenicity mapping
        id_to_path = {
            d.get('uid'): d.get('pathogenicity_prob', 0.0)
            for d in predicted_deletions
        }
        
        # Count true positives
        if tp_vcf.exists():
            try:
                with pysam.VariantFile(str(tp_vcf)) as vcf:
                    for record in vcf:
                        metrics['true_positives'] += 1
                        path_prob = id_to_path.get(record.id, 0.0)
                        if path_prob >= pathogenicity_threshold:
                            metrics['pathogenic_tp'] += 1
                        else:
                            metrics['benign_tp'] += 1
            except Exception as e:
                logger.warning(f"Could not read TP VCF: {e}")
        
        # Count false positives
        if fp_vcf.exists():
            try:
                with pysam.VariantFile(str(fp_vcf)) as vcf:
                    for record in vcf:
                        metrics['false_positives'] += 1
                        path_prob = id_to_path.get(record.id, 0.0)
                        if path_prob >= pathogenicity_threshold:
                            metrics['pathogenic_fp'] += 1
                        else:
                            metrics['benign_fp'] += 1
            except Exception as e:
                logger.warning(f"Could not read FP VCF: {e}")
        
        # Count false negatives
        if fn_vcf.exists():
            try:
                with pysam.VariantFile(str(fn_vcf)) as vcf:
                    for record in vcf:
                        metrics['false_negatives'] += 1
            except Exception as e:
                logger.warning(f"Could not read FN VCF: {e}")
        
        # Calculate derived metrics
        total_predicted = metrics['true_positives'] + metrics['false_positives']
        total_truth = metrics['true_positives'] + metrics['false_negatives']
        
        if total_predicted > 0:
            metrics['precision'] = metrics['true_positives'] / total_predicted
        else:
            metrics['precision'] = 0.0
        
        if total_truth > 0:
            metrics['recall'] = metrics['true_positives'] / total_truth
            metrics['sensitivity'] = metrics['recall']
        else:
            metrics['recall'] = 0.0
            metrics['sensitivity'] = 0.0
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = (
                2 * metrics['precision'] * metrics['recall'] /
                (metrics['precision'] + metrics['recall'])
            )
        else:
            metrics['f1_score'] = 0.0
        
        return metrics
    
    def validate_predictions(
        self,
        predicted_deletions: List[Dict],
        truth_vcf: Path,
        output_dir: Path,
        pathogenicity_threshold: float = 0.5
    ) -> Dict:
        """
        Complete validation workflow: convert to VCF, run Truvari, analyze results.
        
        Args:
            predicted_deletions: List of predicted deletion dictionaries
            truth_vcf: Path to truth set VCF
            output_dir: Output directory
            pathogenicity_threshold: Threshold for pathogenic classification
            
        Returns:
            Comprehensive validation metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert predictions to VCF
        predicted_vcf = output_dir / 'predicted_deletions.vcf'
        sorted_vcf = self.deletions_to_vcf(predicted_deletions, predicted_vcf)
        
        # Run Truvari bench
        truvari_dir = output_dir / 'truvari_bench'
        truvari_metrics = self.run_truvari_bench(
            predicted_vcf=sorted_vcf,
            truth_vcf=truth_vcf,
            output_dir=truvari_dir
        )
        
        # Calculate pathogenicity-specific metrics
        path_metrics = self.calculate_pathogenicity_metrics(
            truvari_output_dir=truvari_dir,
            predicted_deletions=predicted_deletions,
            pathogenicity_threshold=pathogenicity_threshold
        )
        
        # Combine metrics
        combined_metrics = {
            'truvari_bench': truvari_metrics,
            'pathogenicity_analysis': path_metrics,
            'summary': {
                'total_predicted': len(predicted_deletions),
                'total_pathogenic_predicted': sum(
                    1 for d in predicted_deletions 
                    if d.get('pathogenicity_prob', 0) >= pathogenicity_threshold
                ),
                'truvari_precision': truvari_metrics.get('precision', 0),
                'truvari_recall': truvari_metrics.get('recall', 0),
                'truvari_f1': truvari_metrics.get('f1', 0)
            }
        }
        
        # Save metrics
        with open(output_dir / 'validation_metrics.json', 'w') as f:
            json.dump(combined_metrics, f, indent=2)
        
        logger.info(f"Validation complete. Metrics saved to {output_dir / 'validation_metrics.json'}")
        
        return combined_metrics