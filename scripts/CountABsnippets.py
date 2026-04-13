import pandas as pd
import csv
from flashtext import KeywordProcessor

"""
This script is to be run on the output csv file from the LLM classification step, which contains the 'snippet_text' column.
It will count the number of unique antibiotic names found in each snippet and also list which antibiotics were found.
"""
# 1. Load your existing results
input_csv = '../results/claimsFoodFeedVitSuppEnz_C12N_snip_class_gptoss20b.tsv'
df = pd.read_csv(input_csv, sep='\t')

# 2. Setup FlashText
# --- Load Antibiotic Names ---
antibiotic_file = '../CARD_ontology/antibiotics_list.txt'
antibiotics = []
with open(antibiotic_file, 'r') as f:
    for line in f:
        antibiotics.append(line.rstrip().lower())


processor = KeywordProcessor(case_sensitive=False)
for name in antibiotics:
    processor.add_keyword(name)


# 3. Define a helper function to apply to the 'snippet_text' column
def get_antibiotic_stats(text):
    if not isinstance(text, str): return pd.Series([0, ""])
    
    found = processor.extract_keywords(text)
    unique = sorted(list(set(found)))
    
    return pd.Series([len(unique), ", ".join(unique)])

# 4. Apply it efficiently
df[['antibiotic_count', 'antibiotics_found']] = df['snippet_text'].apply(get_antibiotic_stats)

# 5. Save the updated file
output_csv = input_csv.replace('.tsv', '_with_counts.tsv')
df.to_csv(output_csv, index=False)

print(df[['patent_id', 'antibiotic_count', 'antibiotics_found']].head())