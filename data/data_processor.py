def pass_through_variants(variants):
    processed = []

    for v in variants:
        uid = v.get("uid", "N/A")
        title = v.get("title", "N/A")
        gene = "N/A"
        if isinstance(v.get("genes"), list) and v["genes"]:
            gene = v["genes"][0].get("symbol", "N/A")

        chr_, start, stop = "N/A", "?", "?"
        if isinstance(v.get("variation_set"), list) and v["variation_set"]:
            var_loc_list = v["variation_set"][0].get("variation_loc", [])
            if var_loc_list:
                loc = var_loc_list[0]
                chr_ = loc.get("chr", "N/A")
                start = loc.get("start", "?")
                stop = loc.get("stop", "?")

        significance = v.get("germline_classification", {}).get("description", "N/A")
        review_status = v.get("germline_classification", {}).get("review_status", "N/A")

        trait = "N/A"
        trait_set = v.get("germline_classification", {}).get("trait_set", [])
        if trait_set and isinstance(trait_set, list):
            trait = trait_set[0].get("trait_name", "N/A")

        consequence_list = v.get("molecular_consequence_list", [])
        consequence = consequence_list[0] if consequence_list else "N/A"

        processed.append({
            "uid": uid,
            "gene": gene,
            "title": title,
            "chr": chr_,
            "start": start,
            "end": stop,
            "clinical_significance": significance,
            "review_status": review_status,
            "condition": trait,
            "consequence": consequence
        })

    return processed