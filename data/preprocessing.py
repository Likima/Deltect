def summarize_variants(variants):
    counts = {}
    for v in variants:
        significance = v.get("clinical_significance", "N/A") ##Gets the Classification/Review status
        if isinstance(significance, list):
            for item in significance:
                counts[item] = counts.get(item, 0) + 1
        else:
            counts[significance] = counts.get(significance, 0) + 1

    for k, v in counts.items():
        print(f"  - {k}: {v}")