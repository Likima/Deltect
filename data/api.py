import requests

def fetch_clinvar_deletions_entrez(chrom="22", max_results=50):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "clinvar",
        "term": f"deletion[variant type] AND {chrom}[chromosome]",
        "retmax": max_results,
        "retmode": "json"
    } ##Gets 50 examples with the query

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

    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    summary_params = {
        "db": "clinvar",
        "id": ",".join(ids),
        "retmode": "json"
    } ##This was to get other fields

    try:
        s_resp = requests.get(summary_url, params=summary_params)
        s_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error during summary request: {e}")
        return []

    s_data = s_resp.json()
    summaries = []
    result_data = s_data.get("result", {})

    for vid in ids:
        variant = result_data.get(vid)
        if variant:
            summaries.append(variant)

    if len(summaries) == 0:
        print("No usable variant summaries found in API response.")

    return summaries ##Return what was queried