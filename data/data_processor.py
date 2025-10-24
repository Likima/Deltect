
# Indices of display field arrays
NAME = 0
TYPE = 1
SEQ_ID = 2
FEATURE_START = 3
FEATURE_END = 4
CLINICAL_INT = 5

def process_api_data(data):

    processed = [] # Array for formatted data

    total_count = data[0]  # Total number of results
    codes = data[1]  # Array of codes (Name field)
    extra_data = data[2]  # Hash of extra fields
    display_strings = data[3]  # Display field arrays
    
    # print(f"Total deletions found: {total_count}\n")
    
    for i in range(len(codes)):
        # print(f"Variant {i+1}:")
        # print(f"Name: {codes[i]}")
        # print(f"Display: {display_strings[i]}")
        # print(f"Phenotype: {extra_data['phenotype'][i]}")
        # print(f"Zygosity: {extra_data['Zygosity'][i]}")
        # print(f"Origin: {extra_data['var_origin'][i]}")
        # print(f"dbVar Link: {extra_data['Dbxref'][i]}")
        # print()
        current_fields = display_strings[i]

        processed.append(
            {
                'variant': i+1,
                'name': current_fields[NAME],
                'type': current_fields[TYPE],
                'seq_id': current_fields[SEQ_ID],
                'feature_start': current_fields[FEATURE_START],
                'feature_end': current_fields[FEATURE_END],
                'feature_length': calculate_length (
                                        current_fields[FEATURE_START], 
                                        current_fields[FEATURE_END]
                                ),
                'clinical_intervention': current_fields[CLINICAL_INT],
                'phenotype': extra_data['phenotype'][i],
                'zygosity': extra_data['Zygosity'][i],
                'origin': extra_data['var_origin'][i],
                'dbvar_link': extra_data['Dbxref'][i]
            }
        )

    # print(processed)
    return processed


def calculate_length(feature_start: int, feature_end: int): 
    return str(int(feature_end) - int(feature_start))