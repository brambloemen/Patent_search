import csv
# Read the contents of the file
file_path = "/scratch/brbloemen/ARGUMENT/Patent_search/CARD_ontology/aro.obo"

classes = set()
with open(file_path, "r") as file:
    for line in file.readlines():
        if line.startswith("id: ARO:"):
            aro = line.replace("id: ", "").strip()
        if line.startswith("name: "):
            # Extract the name part
            name = line.replace("name: ", "").strip()
        
        if line.startswith("is_a: ARO:1000003 ! antibiotic molecule"):
            filter_criterium = "is_a: " + aro + " ! " + name
            classes.add(filter_criterium)

antibiotics = []
with open(file_path, "r") as file:
    name = str()
    for line in file.readlines():
        if line.startswith("name: "):
            # Extract the name part
            name = line.replace("name: ", "").strip()
            
            # Remove the suffix " antibiotic" if present
            # if name.endswith(" antibiotic"):
            #     name = name.rsplit(" antibiotic", 1)[0]

        
        if line.rstrip() in classes:
            antibiotics.append(name)

# Write the antibiotics to a CSV file
output_file = "/scratch/brbloemen/ARGUMENT/Patent_search/CARD_ontology/antibiotics_CARD_aro.csv"
with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for antibiotic in antibiotics:
        writer.writerow([antibiotic])


