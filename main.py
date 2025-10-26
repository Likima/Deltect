from data.api import fetch_clinvar_deletions_entrez
from data.data_processor import pass_through_variants
from data.preprocessing import summarize_variants

def main():
    raw_variants = fetch_clinvar_deletions_entrez(chrom="22", max_results=500)
    ##print(f"Raw variants fetched: {len(raw_variants)}")

    all_variants = pass_through_variants(raw_variants)
    ##print(f"Variants after processing: {len(all_variants)}")

    summarize_variants(all_variants)

    if all_variants:
        print("\nFinal Variant Report:")
        for v in all_variants:
            print(f"• ID: {v.get('uid', 'N/A')} | "
                  f"Gene: {v.get('gene', 'N/A')} | "
                  f"Title: {v.get('title', 'N/A')} | "
                  f"Location: {v.get('chr', 'N/A')}:{v.get('start', '?')}-{v.get('end', '?')} | "
                  f"Significance: {v.get('clinical_significance', 'N/A')} | "
                  f"Condition: {v.get('condition', 'N/A')}")
    else:
        print("No variants found.")

if __name__ == "__main__":
    main()