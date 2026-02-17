import json
import csv
import pandas as pd
from flashtext import KeywordProcessor
from openai import OpenAI
import os
from dotenv import load_dotenv

# --- Initialize Nebius Client (uses openAI system) ---
load_dotenv()
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# --- Load Antibiotic Names ---
antibiotic_file = '../CARD_ontology/antibiotics_CARD_aro.csv'
antibiotics = []
with open(antibiotic_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        antibiotics.append(row[0].lower())


processor = KeywordProcessor(case_sensitive=False)
for name in antibiotics:
    processor.add_keyword(name)


# --- Load Resistance Genes and Terms ---
resistance_gene_file = '../CARD_ontology/aro.tsv'
resistance_genes = pd.read_csv(resistance_gene_file, sep='\t')["Name"]
resistance_terms = ["selection marker", "selectable marker", "marker gene"]

# # --- llm-classifier function ---
# def categorize_with_llm(text_snippet):
#     prompt = f"""
#     Analyze this patent text regarding antibiotic selection markers:
#     "{text_snippet}"
#     Does this text indicate the USE of an antibiotic marker, or the AVOIDANCE of one?
#     Return JSON: {{"category": "USE" or "AVOIDANCE", "reason": "..."}}
#     """
    
#     response = openai.chat.completions.create(
#         model="gpt-4o-mini", # Use a smaller model for cost efficiency
#         messages=[{"role": "user", "content": prompt}],
#         response_format={ "type": "json_object" }
#     )
#     return json.loads(response.choices[0].message.content)

# --- Process Patent Data ---
output_file = 'patent_antibiotic_analysis.csv'
headers = ['lens_id', 'date_published', 'jurisdiction', 'doc_number', 'antibiotic_counts', 'total_hits']
results = []
path = "/scratch/brbloemen/ARGUMENT/Patent_search/claims_FoodFeedSuppVitEnz_C12N.json"

with open(path, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        if not line.strip():
            continue
            
        patent = json.loads(line)
        
        # 1. Extract and Process Claims
        claims_data = patent.get('claims', [])
        
        if claims_data != None and len(claims_data) > 0:
            claims_data = claims_data[0]["claims"]
            all_claims_text = " ".join([
                " ".join(claim.get('claim_text', [])) 
                for claim in claims_data
            ])        
            found_claims = processor.extract_keywords(all_claims_text)
        else:
            all_claims_text = ""
            found_claims = []

        # 2. Extract and Process Description
        # Lens descriptions can be a string or a list/dict depending on the version
        desc_data = patent.get('description', {})
        if isinstance(desc_data, dict):
            description_text = desc_data.get('text', "")
        else:
            description_text = str(desc_data)
            
        found_desc = processor.extract_keywords(description_text)

        word_count_claims = len(all_claims_text.split())
        word_count_desc = len(description_text.split())
        
        # 3. Build the row dictionary
        patent_entry = {
            'lens_id': patent.get('lens_id'),
            'date_published': patent.get('date_published'),
            'jurisdiction': patent.get('jurisdiction'),
            'doc_number': patent.get('doc_number'),
            'title': patent.get('title', {}).get('text', 'N/A'), # Useful for context
            'word_count_claims': word_count_claims, # Added
            'word_count_desc': word_count_desc,     # Added
            'word_count_total': word_count_claims + word_count_desc # Added
        }
        
        # Dynamically add columns for each antibiotic
        for marker in antibiotics:
            patent_entry[f'{marker}'] = found_claims.count(marker)
            patent_entry[f'{marker}'] += found_desc.count(marker)
        
        patent_entry['total_hits'] = len(found_claims) + len(found_desc)
        
        results.append(patent_entry)

# --- Create the DataFrame ---
df = pd.DataFrame(results)

# Optional: Set lens_id as index
df.set_index('lens_id', inplace=True)

# Preview the results

print(df.sort_values('total_hits', ascending=False).head(10))

# --- Save to CSV ---
output_file = 'patent_antibiotics_claimsFoodFeedVitSuppEnz_C12N.csv'
df.to_csv(output_file)


