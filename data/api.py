import requests
from dotenv import dotenv_values
env = dotenv_values(".env")
ENTREZ_API_KEY = env.get("ENTREZ_API_KEY")

def fetch_clinvar_deletions_entrez(chrom="22", max_results=500):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "clinvar",
        "term": f"deletion[variant type] AND {chrom}[chromosome]",
        "retmax": max_results,
        "retmode": "json"
    }

    try:
        search_resp = requests.get(search_url, params=search_params)
        search_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error during search request: {e}")
        return []

    search_data = search_resp.json()
    ids = search_data.get("esearchresult", {}).get("idlist", [])

    if not ids:
        return []

    batch_size = 50
    summaries = []
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            "db": "clinvar",
            "id": ",".join(batch_ids),
            "retmode": "json",
            "api_key": ENTREZ_API_KEY
        }

        try:
            s_resp = requests.get(summary_url, params=summary_params)
            s_resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error during summary request for batch {i//batch_size + 1}: {e}")
            continue

        s_data = s_resp.json()
        result_data = s_data.get("result", {})

        for vid in batch_ids:
            variant = result_data.get(vid)
            if variant:
                summaries.append(variant)

    if len(summaries) == 0:
        print("No usable variant summaries found in API response.")

    return summaries