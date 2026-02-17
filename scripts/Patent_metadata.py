import json
import pandas as pd
path = "/scratch/brbloemen/ARGUMENT/Patent_search/claims_FoodFeedSuppVitEnz_C12N.json"

with open(path, 'r', encoding='utf-8') as f_in:
    rows = []

    counter = 0
    for line in f_in:
        if not line.strip():
            continue
        
        # store patent information
        patent = json.loads(line)

        claims_data = patent.get('claims', [])
        
        if claims_data != None and len(claims_data) > 0:
            claims_data = claims_data[0]["claims"]
            all_claims_text = " ".join([
                " ".join(claim.get('claim_text', [])) 
                for claim in claims_data
            ])        
        else:
            all_claims_text = ""

        desc_data = patent.get('description', {})
        if isinstance(desc_data, dict):
            description_text = desc_data.get('text', "")
        else:
            description_text = str(desc_data)

        word_count_claims = len(all_claims_text.split())
        word_count_desc = len(description_text.split())

        rows.append({
            'lens_id': patent.get('lens_id'),
            'date_published': patent.get('date_published'),
            'jurisdiction': patent.get('jurisdiction'),
            'doc_number': patent.get('doc_number'),
            'title': patent.get('title', {}).get('text', 'N/A'), # Useful for context
            'word_count_claims': word_count_claims, # Added
            'word_count_desc': word_count_desc,     # Added
            'word_count_total': word_count_claims + word_count_desc # Added
        })

patents = pd.DataFrame(rows)
patents.to_csv('../results/claimsFoodFeedVitSuppEnz_C12N_patents_metadata.csv', index=False)
