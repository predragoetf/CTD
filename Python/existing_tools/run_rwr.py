from pyrwr.rwr import RWR
from pyrwr.ppr import PPR
from pyrwr.pagerank import PageRank
import numpy as np
import pandas as pd
import time
from multiprocessing import shared_memory, Lock
from concurrent.futures import ProcessPoolExecutor
from dataclasses import make_dataclass
import csv
import random
import math
import argparse as ap

#TODO: reuse the read_s, split_80_20 and ndcg functions from the other file

def read_S(s_path:str)->list[str]:
    with open(s_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        S_nodes = [row for row in csv_reader]

    S_nodes = np.array(S_nodes)
    S_nodes = S_nodes.flatten()
    return S_nodes

def split_80_20(array:list[str], generator_seed:int=42)->tuple[list[str],list[str]]:
    # Calculate the split index (80% of the list)
    split_index = int(len(array) * 0.8)

    # Shuffle the list randomly
    random.seed(generator_seed)
    random.shuffle(array)

    # Split the list into two lists
    anchors = array[:split_index]  # First 80% elements
    targets = array[split_index:]  # Remaining 20% elements

    return (anchors, targets)

def ndcg(rankings:pd.DataFrame, marked:list[str], relevance_scores:dict=None)->float:
    if (relevance_scores is None):
        relevance_scores = {node:1 for node in marked} 
    indices_of_marked: list[int] = rankings.index[rankings['Node_id'].isin(marked)].tolist()
    print(f"Positions of marked nodes are {indices_of_marked}")
    dcg_scores = [1/math.log2(2+rank) for rank in indices_of_marked]#TODO use relevance scores!
    dcg = sum(dcg_scores)
    ideal_dcg = sum([1/math.log2(2+rank) for rank in range(0, len(marked))])#TODO use relevance scores!

    ndcg_score = dcg / ideal_dcg
    print(f"The ndcg score is {ndcg_score}.")
    return ndcg_score

parser = ap.ArgumentParser(
        description="A script that tests existing GBA implementations.")

parser.add_argument(
        "-a",
        "--algorithm",
        help="rwr, pagerank, ppr", 
        type=str,
        default="rwr",
    )

args = parser.parse_args()
algorithm = args.algorithm

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

disease_list = [arthritis, asthma, chron_pulmo, dilated_cardiomyopath, breast_carcinoma, lung_adenocarcinoma, psoriasis, ulcerative_colitis, type_2_diabetes]
split_seeds = [42, 45, 55, 100, 420, 20, 40, 18, 22, 157]

for disease in disease_list:
    results_list: list[dict] = []
    for seed in split_seeds:
        print(f"\nRunning {algorithm} for {disease.name} with seed {seed}...")

        edge_list = f"data_rwr/{disease.name}_edge_list.txt"
        test_name = f"{algorithm}_{disease.name}_{seed}"
        S = read_S(disease.s_path)
        

        adj_matrix = pd.read_csv(disease.adj_path, skip_blank_lines=True)
        labels = adj_matrix.columns
        labels_to_ints_dict = {label: i for i, label in enumerate(labels)}
        ints_to_labels_dict = {i: label for i, label in enumerate(labels)}

        if (algorithm == "rwr"):
            # Initialize the RWR model
            model = RWR()
        elif (algorithm == "ppr"):
            # Initialize the personalized PageRank model
            model = PPR()
        elif (algorithm == "pagerank"):
            # Initialize the PageRank model
            model = PageRank()

        # Read the graph from the generated file
        model.read_graph(edge_list, graph_type="undirected")

        # Define the "guilty nodes" (seed nodes)
        num_nodes = model.node_ids.shape[0]
        S_remap = [labels_to_ints_dict[node_label] for node_label in S]
        anchors, targets = split_80_20(S_remap, seed)
        num_in_S = len(S_remap)

        total_time_start = time.time()
        # Create shared memory array
        agg_shape = (num_nodes,)
        shared_array = shared_memory.SharedMemory(create=True, size=np.zeros(agg_shape).nbytes)
        agg_shared = np.ndarray(agg_shape, dtype=np.float64, buffer=shared_array.buf)
        #agg_shared[:] = 0  # Initialize to zero

        # Lock to ensure thread-safety
        lock = Lock()

        def compute_and_update(seed):
            scores = model.compute(seed)
            with lock:  # Ensure thread safety
                np.add(agg_shared, scores, out=agg_shared)

        with ProcessPoolExecutor() as executor:
            executor.map(compute_and_update, anchors)

        # Normalize the shared array
        agg_shared /= num_in_S

        total_time_end = time.time()

        total_time = total_time_end - total_time_start
        print(f"\n{algorithm} tests completed in {total_time} seconds.")

        results_array = agg_shared
        
        # Rank nodes by scores in descending order
        ranked_nodes = np.argsort(-results_array)

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(columns=['Node_id', 'Importance'])

        print("Ranking nodes based on guilt by association:")
        for rank, node_index in enumerate(ranked_nodes):
            node_id = model.node_ids[node_index]
            original_label = ints_to_labels_dict[node_id]
            score = results_array[node_index]

            #print(f"Rank {rank + 1}: Node {original_label}, Score: {score:.4f}")

            # Append row to DataFrame
            results_df.loc[len(results_df)] = {'Node_id': original_label, 'Importance': score}

        # Save the results to a CSV file
        results_df.to_csv(f"./Python/existing_tools/results/{algorithm}/{test_name}_results.csv", index=False)

        # Perform ndcg calculation
        targets_original_labels = [ints_to_labels_dict[node_id] for node_id in targets]
        ndcg_score = ndcg(results_df, targets_original_labels)

        # Compute ranks of target nodes
        ranks:list = results_df.index[results_df['Node_id'].isin(targets_original_labels)].tolist()

        # TODO: create benchmark results file
        new_test_result = {'test_name':f"{test_name}_rwr", 
                      'ndgc':        ndcg_score, 
                      'time':total_time, 
                      'split_name':f"generated_{seed}",
                      'ranks': ','.join(map(str, ranks))}
        results_list.append(new_test_result)

        # Check if shared_array is closed and unlinked properly
        try:
            shared_array.close()
            shared_array.unlink()
            print("Shared array closed and unlinked successfully.")
        except Exception as e:
            print(f"Error closing or unlinking shared array: {e}")

    results_dtype_dict = {'test_name':'str', 
                        'ndgc':'float', 
                        'time':'float',  
                        'split_name':'str',
                        'ranks': 'str'}
    results_columns = ['test_name', 'ndgc', 'time', 'split_name', 'ranks']
    benchmark_results_df:pd.DataFrame = pd.DataFrame(data=results_list, columns=results_columns)
    benchmark_results_df.astype(results_dtype_dict)
    benchmark_results_df.to_csv(f"Python/existing_tools/results/{algorithm}/benchmark_results_{algorithm}_{disease.name}.csv", index=False, header=True)
    
