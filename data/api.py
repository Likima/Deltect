"""
Optimized API client with batch fetching for faster data retrieval.
"""
import time
from typing import List, Dict, Optional
from Bio import Entrez
from dotenv import dotenv_values
import logging

logger = logging.getLogger(__name__)


class ClinVarClient:
    """Client for fetching variants from NCBI dbVar database."""
    
    # Rate limiting
    MAX_REQUESTS_PER_SECOND = 3
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2
    
    # Batch size for fetching (Entrez allows up to 500 per request)
    BATCH_SIZE = 200  # Conservative batch size for stability
    
    def __init__(self, env_file: str = ".env"):
        """Initialize dbVar client."""
        env = dotenv_values(env_file)
        self.api_key = env.get("ENTREZ_API_KEY")
        self.email = env.get("ENTREZ_EMAIL")
        
        if not self.email:
            raise ValueError("ENTREZ_EMAIL must be set in .env file")
        
        Entrez.api_key = self.api_key
        Entrez.email = self.email
        self._last_request_time = 0
        
        logger.info(f"Initialized DbVarClient with email: {self.email}")
    
    def _rate_limit(self):
        """Implement rate limiting for API requests."""
        elapsed = time.time() - self._last_request_time
        min_interval = 1.0 / self.MAX_REQUESTS_PER_SECOND
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    def _fetch_variants_batch(self, variant_ids: List[str]) -> List[Dict]:
        """Fetch multiple variants in a single API call.
        
        Args:
            variant_ids: List of variant IDs to fetch
            
        Returns:
            List of variant dictionaries
        """
        if not variant_ids:
            return []
        
        self._rate_limit()
        
        # Join IDs with commas for batch request
        id_string = ",".join(variant_ids)
        
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                with Entrez.esummary(db="clinVar", id=id_string, rettype="xml") as stream:
                    # Entrez.read returns a list when fetching multiple IDs
                    results = Entrez.read(stream, validate=False, ignore_errors=True)
                
                # Results may be a list or a single dict
                if isinstance(results, list):
                    return results
                elif isinstance(results, dict):
                    # Check if it's a DocumentSummarySet
                    if 'DocumentSummarySet' in results:
                        doc_set = results['DocumentSummarySet']
                        if 'DocumentSummary' in doc_set:
                            return doc_set['DocumentSummary']
                    return [results]
                else:
                    logger.warning(f"Unexpected result type: {type(results)}")
                    return []
                    
            except Exception as e:
                if attempt < self.RETRY_ATTEMPTS - 1:
                    logger.warning(
                        f"Batch fetch retry {attempt + 1}/{self.RETRY_ATTEMPTS}: {e}"
                    )
                    time.sleep(self.RETRY_DELAY * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch batch after {self.RETRY_ATTEMPTS} attempts: {e}")
                    return []
        
        return []
    
    def fetch_deletion_variants(
        self,
        chr: str = "22",
        max_results: int = 500
    ) -> List[Dict]:
        """Fetch deletion variants from ClinVar with 2/3 pathogenic and 1/3 non-pathogenic.
        
        Args:
            chr: Chromosome number
            max_results: Maximum total number of results to fetch
            
        Returns:
            List of variant dictionaries (2/3 pathogenic, 1/3 non-pathogenic)
        """
        # Calculate split: 2/3 pathogenic, 1/3 non-pathogenic
        pathogenic_count = int(max_results * 2 / 3)
        non_pathogenic_count = max_results - pathogenic_count
        
        logger.info(f"Fetching {pathogenic_count} pathogenic and {non_pathogenic_count} non-pathogenic variants")
        
        # Base search term (common to both)
        base_term = (
            f'"{chr}"[Chromosome] '
            f'AND "deletion"[Type of variation] '
            f'AND ("criteria provided, multiple submitters, no conflicts"[Review status] '
            f'OR "criteria provided, single submitter"[Review status])'
        )
        
        # Pathogenic search term
        pathogenic_term = (
            f'{base_term} '
            f'AND ("pathogenic"[Clinical significance] '
            f'OR "likely pathogenic"[Clinical significance])'
        )
        
        # Non-pathogenic search term
        non_pathogenic_term = (
            f'{base_term} '
            f'AND ("benign"[Clinical significance] '
            f'OR "likely benign"[Clinical significance])'
        )
        
        all_variants = []
        
        # Fetch pathogenic variants
        logger.info(f"Searching for pathogenic variants: {pathogenic_term}")
        pathogenic_variants = self._fetch_variants_by_term(pathogenic_term, pathogenic_count)
        all_variants.extend(pathogenic_variants)
        logger.info(f"Fetched {len(pathogenic_variants)} pathogenic variants")
        
        # Fetch non-pathogenic variants
        logger.info(f"Searching for non-pathogenic variants: {non_pathogenic_term}")
        non_pathogenic_variants = self._fetch_variants_by_term(non_pathogenic_term, non_pathogenic_count)
        all_variants.extend(non_pathogenic_variants)
        logger.info(f"Fetched {len(non_pathogenic_variants)} non-pathogenic variants")
        
        logger.info(f"Total variants fetched: {len(all_variants)} "
                    f"(pathogenic: {len(pathogenic_variants)}, non-pathogenic: {len(non_pathogenic_variants)})")
        
        return all_variants

    def _fetch_variants_by_term(self, search_term: str, max_results: int) -> List[Dict]:
        """Helper method to fetch variants for a specific search term.
        
        Args:
            search_term: Complete search query
            max_results: Maximum number of results to fetch
            
        Returns:
            List of variant dictionaries
        """
        # Search for variant IDs
        self._rate_limit()
        try:
            with Entrez.esearch(db="clinvar", term=search_term, retmax=max_results) as stream:
                record = Entrez.read(stream, validate=False, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error searching ClinVar: {e}")
            return []
        
        variant_ids = record.get("IdList", [])
        
        if not variant_ids:
            logger.warning(f"No variants found for term: {search_term}")
            return []
        
        logger.info(f"Found {len(variant_ids)} variant IDs, fetching in batches of {self.BATCH_SIZE}...")
        
        # Fetch variants in batches
        all_variants = []
        total_ids = len(variant_ids)
        
        for i in range(0, total_ids, self.BATCH_SIZE):
            batch_ids = variant_ids[i:i + self.BATCH_SIZE]
            batch_num = (i // self.BATCH_SIZE) + 1
            total_batches = (total_ids + self.BATCH_SIZE - 1) // self.BATCH_SIZE
            
            logger.info(
                f"Fetching batch {batch_num}/{total_batches} "
                f"({len(batch_ids)} variants, {i+len(batch_ids)}/{total_ids} total)"
            )
            
            batch_results = self._fetch_variants_batch(batch_ids)
            
            if batch_results:
                all_variants.extend(batch_results)
                logger.info(f"Batch {batch_num} successful ({len(batch_results)} variants)")
            else:
                logger.warning(f"Batch {batch_num} returned no results")
        
        return all_variants
        
    def get_variant_stats(self, variants: List[Dict]) -> Dict:
        """Get statistics about fetched variants."""
        if not variants:
            return {"total": 0}
        
        stats = {
            "total": len(variants),
            "has_gene": sum(1 for v in variants if v.get("dbvargenelist")),
            "has_placement": sum(1 for v in variants if v.get("dbvarplacementlist")),
            "has_clinsig": sum(1 for v in variants if v.get("dbvarclinicalsignificancelist")),
        }
        
        variant_types = {}
        for v in variants:
            vtype = v.get("variant_type", "unknown")
            variant_types[vtype] = variant_types.get(vtype, 0) + 1
        
        stats["variant_types"] = variant_types
        
        return stats


def fetch_clinvar_deletions_entrez(chrom, max_results=500):
    """Legacy function for backward compatibility."""
    client = ClinVarClient()
    variants = client.fetch_deletion_variants(chrom, max_results=max_results)
    
    stats = client.get_variant_stats(variants)
    logger.info(f"\nVariant Statistics:")
    logger.info(f"  Total variants: {stats['total']}")
    logger.info(f"  With gene info: {stats['has_gene']}")
    logger.info(f"  With placement: {stats['has_placement']}")
    logger.info(f"  With clinical significance: {stats['has_clinsig']}")
    
    if stats.get('variant_types'):
        logger.info(f"  Variant types: {stats['variant_types']}")
    
    return variants