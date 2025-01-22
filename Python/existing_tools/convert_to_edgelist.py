import numpy as np
import pandas as pd
import csv
from dataclasses import make_dataclass
from tqdm import tqdm
import pickle

def read_S(s_path:str)->list[str]:
    with open(s_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        S_nodes = [row for row in csv_reader]

    S_nodes = np.array(S_nodes)
    S_nodes = S_nodes.flatten()
    return S_nodes

Disease = make_dataclass("Disease", [("name", str), ("adj_path", str), ("s_path", str)])

arthritis = Disease("arthritis", "data_rwr/Arthritis_exp_adj.csv", "data_rwr/Arthritis_exp_s.csv")
asthma = Disease("asthma","data_rwr/Asthma_exp_adj.csv", "data_rwr/Asthma_exp_s.csv")
chron_pulmo = Disease("chronic_obstructive_pulmonary_disease","data_rwr/Chronic_Obstructive_Pulmonary_Disease_exp_adj.csv", "data_rwr/Chronic_Obstructive_Pulmonary_Disease_exp_s.csv")
dilated_cardiomyopath = Disease("dilated_cardiomyopathy", "data_rwr/Dilated_Cardiomyopathy_exp_adj.csv", "data_rwr/Dilated_Cardiomyopathy_exp_s.csv")
breast_carcinoma = Disease("invasive_breast_carcinoma", "data_rwr/Invasive_Breast_Carcinoma_exp_adj.csv", "data_rwr/Invasive_Breast_Carcinoma_exp_s.csv" )
lung_adenocarcinoma = Disease("lung_adenocarcinoma", "data_rwr/Lung_Adenocarcinoma_exp_adj.csv", "data_rwr/Lung_Adenocarcinoma_exp_s.csv")
psoriasis = Disease("psoriasis", "data_rwr/Psoriasis_exp_adj.csv", "data_rwr/Psoriasis_exp_s.csv")
ulcerative_colitis = Disease("ulcerative_colitis", "data_rwr/Ulcerative_colitis_exp_adj.csv", "data_rwr/Ulcerative_colitis_exp_s.csv")
type_2_diabetes = Disease("type_2_diabetes", "data_rwr/Type_2_diabetes_exp_adj.csv", "data_rwr/Type_2_diabetes_exp_s.csv")

disease_list = [type_2_diabetes]

# disease_list = [arthritis, asthma, chron_pulmo, dilated_cardiomyopath, breast_carcinoma,
#               lung_adenocarcinoma, psoriasis, ulcerative_colitis, type_2_diabetes]

# Define the adjacency matrix as a pandas DataFrame
# Example graph adjacency matrix
data = {
    'c1': {'c1': 0, 'c2': 1.5, 'c3': 0, 'c4': 3.5, 'c5': 0, 'c6': 0},
    'c2': {'c1': 1.5, 'c2': 0, 'c3': 6.0, 'c4': 0, 'c5': 0, 'c6': 0},
    'c3': {'c1': 0, 'c2': 6.0, 'c3': 0, 'c4': 1.0, 'c5': 0, 'c6': 2.0},
    'c4': {'c1': 3.5, 'c2': 0, 'c3': 1.0, 'c4': 0, 'c5': 2.5, 'c6': 0},
    'c5': {'c1': 0, 'c2': 0, 'c3': 0, 'c4': 2.5, 'c5': 0, 'c6': 1.0},
    'c6': {'c1': 0, 'c2': 0, 'c3': 2.0, 'c4': 0, 'c5': 1.0, 'c6': 0}
}
#adj_matrix = pd.DataFrame(data)
#S = ['c2','c3']  # Example: nodes 2 and 3 are considered guilty

for disease in disease_list:
    disease_name, adj_path, S_path = disease.name, disease.adj_path, disease.s_path
    output_path  = f"data_rwr/{disease_name}_edge_list.txt"

    #adj_path = "/home/op140096d/code/CTD/data/arthritis/Arthritis_large_adj.csv"
    adj_matrix = pd.read_csv(adj_path, skip_blank_lines=True)
    labels = adj_matrix.columns
    labels_to_ints_dict = {label: i for i, label in enumerate(labels)}
    ints_to_labels_dict = {i: label for i, label in enumerate(labels)}

    # Rename the rows and columns of the adjacency matrix to integers
    adj_matrix = adj_matrix.rename(columns=labels_to_ints_dict, index=labels_to_ints_dict)

    #print(adj_matrix[[0,1,2,3]].head(3))

    #S_path = "/home/op140096d/code/CTD/data/arthritis/anchors_42.csv"
    S = read_S(S_path)
    S = [labels_to_ints_dict[node] for node in S]

    # Convert adjacency matrix to edge list format
    edge_list = []
    row_iter = tqdm(adj_matrix.index, desc="Processing rows")
    for source in row_iter:
        col_iter = tqdm(adj_matrix.columns, desc=f"Processing columns for source {source}", leave=False)
        for target in col_iter:
            weight = adj_matrix.loc[source, target]
            if weight > 0:  # Include only edges with non-zero weight
                edge_list.append((source, target, weight))

    # Step 3: Write edge list to a file in the required format
    with open(output_path, "w") as f:
        for source, target, weight in edge_list:
            f.write(f"{source}\t{target}\t{weight}\n")