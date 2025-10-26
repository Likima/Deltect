import requests
import json

base_url = "https://clinicaltables.nlm.nih.gov/api/clinvar/v3/search"

# Query parameters (same filters as before)
params = {
    'terms': '',  # Required field
    'q': 'Type:deletion AND SeqID:NC_000022.10',  # Filter for deletion type variants
    'df': 'Name,Type,SeqID,FeatureStart,FeatureEnd,clinical_int',  # Display fields
    'ef': 'Dbxref,phenotype,Zygosity,var_origin',  # Extra fields to return
    'maxList': 100  # Limit results
}

def fetch_clinvar_deletions():
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None