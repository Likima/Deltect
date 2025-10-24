import requests
import json

base_url = "https://clinicaltables.nlm.nih.gov/api/dbvar/v3/search"

# Query parameters - requesting all available fields
params = {
    'terms': '',  # Required field
    'q': 'Type:deletion',  # Filter for deletion type variants on chromosome 21
    'df': 'Name,Type,SeqID,FeatureStart,FeatureEnd,clinical_int,Alias,ID,parent',  # All display fields
    'ef': 'Dbxref,phenotype,Zygosity,var_origin,ciend,cipos,copy_number,End_range,gender,sampleset_name,sampleset_type,Start_range',  # All extra fields
    'maxList': 100  # Limit results
}

response = requests.get(base_url, params=params)

if response.status_code == 200:
    data = response.json()
    
    total_count = data[0]  # Total number of results
    codes = data[1]  # Array of codes (Name field)
    extra_data = data[2]  # Hash of extra fields
    display_strings = data[3]  # Display field arrays
    
    print(f"Total deletions found: {total_count}\n")
    
    for i in range(len(codes)):
        print(f"\n{'='*80}")
        print(f"Variant {i+1}:")
        print(f"{'='*80}")
        
        # Display fields
        print(f"  Name: {codes[i]}")
        print(f"  Display String: {display_strings[i]}")
        
        # Extra fields (check if they exist before printing)
        for field in extra_data.keys():
            if i < len(extra_data[field]):
                value = extra_data[field][i]
                print(f"  {field}: {value}")
        
        print()
else:
    print(f"Error: {response.status_code}")