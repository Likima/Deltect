from Bio import Entrez
from dotenv import dotenv_values

config = dotenv_values(".env")
api_key = config.get("ENTREZ_API_KEY")
email = config.get("ENTREZ_EMAIL")

        
Entrez.api_key = api_key
Entrez.email = email

# List of human chromosomes
chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

total_entries_pathogenic = 0
total_entries_non_pathogenic = 0

non_pathogenic_count_list = []

for chrom in chromosomes:
    search_term1 = (f'"{chrom}"[Chromosome] AND "deletion"[Type of variation]'
                   f' AND ("clinsig pathogenic"[Properties] OR "clinsig likely pathogenic"[Properties])' 
    )
    search_term2 = (f'"{chrom}"[Chromosome] AND "deletion"[Type of variation]'
                    f' AND ("clinsig benign"[Properties] OR "clinsig likely benign"[Properties])')
    
    # print(search_term)

    with Entrez.esearch(db="clinvar", term=search_term1, retmax=10000) as stream:
        record = Entrez.read(stream)
        count = int(record["Count"])
        print(f"Chromosome {chrom}: {count} entries (pathogenic)")
        total_entries_pathogenic += count

    with Entrez.esearch(db="clinvar", term=search_term2, retmax=10000) as stream:
        record = Entrez.read(stream)
        count = int(record["Count"])
        print(f"Chromosome {chrom}: {count} entries (non pathogenic)")
        non_pathogenic_count_list.append(count)
        total_entries_non_pathogenic += count

print(f"\nTotal pathogenic across all chromosomes: {total_entries_pathogenic}")
print(f"\nTotal non pathogenic across all chromosomes: {total_entries_non_pathogenic}")

print(max(non_pathogenic_count_list))