import csv
import obonet
import networkx as nx

def export_antibiotics_to_txt(file_path="../CARD_ontology/aro.obo", output_filename="../CARD_ontology/antibiotics_list.txt"):
   
    # Read the ontology into a networkx graph
    graph = obonet.read_obo(file_path)
    
    root_id = "ARO:1000003"
    
    if root_id not in graph:
        print(f"Error: Could not find {root_id} in the ontology.")
        return
    
# Create a subgraph that ONLY contains 'is_a' edges
    # This effectively strips away resistance mechanisms and targets
    is_a_graph = nx.MultiDiGraph([
        (u, v, key) for u, v, key in graph.edges(keys=True) 
        if key == 'is_a'
    ])
    
    # Add back the node attributes (like 'name') from the original graph
    for node in is_a_graph.nodes():
        if node in graph.nodes:
            is_a_graph.nodes[node].update(graph.nodes[node])
    
    # Find all nodes that point to the root via 'is_a' relationships
    # In obonet/networkx, child -> parent is the direction, so we use ancestors
    antibiotic_ids = nx.ancestors(is_a_graph, root_id)
    
    # Create the clean list
    antibiotics = []
    exclusions = set(["linoleic acid", "palmitic acid", "oleic acid", "peptide antibiotic", "phosphonic acid antibiotic",
                      "polyamine antibiotic", "polyene antibiotic", "lipopeptide antibiotic"])
    for node_id in antibiotic_ids:
        name = is_a_graph.nodes[node_id].get("name")
        if name:
            # exclude certain entries that are not actual antibiotics or are too generic
            if name in exclusions:
                continue
            if name.endswith("with antibiotic activity") or name.startswith("antibiotic") or name.endswith("antibiotics"):
                continue
            if name.endswith(" antibiotic"):
                name = name.rsplit(" antibiotic", 1)[0]
            
            antibiotics.append(name)
    
    # Sort alphabetically for a cleaner file
    antibiotics.sort()

    with open(output_filename, "w", encoding="utf-8") as file:
        for name in antibiotics:
            file.write(f"{name}\n")
                
    print(f"\nDone! Saved {len(antibiotics)} filtered antibiotic names to '{output_filename}'.")

if __name__ == "__main__":
    export_antibiotics_to_txt()