import argparse
import collections
from tqdm import tqdm
import simplejson as json
from typing import Iterable
import nmslib

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(list_1, list_2):
    cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
    return cos_sim

def create_vector_space(embedd_path, metadados_path):
    """
    Create a vector space from the embeddings.
    """
    df = pd.read_csv(metadados_path, sep='\t')
    df = df.dropna()
    df = df.reset_index(drop=True)

    embeddings = np.loadtxt(embedd_path)
    embeddings = np.nan_to_num(embeddings)

    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(embeddings)
    index.createIndex({'post': 2}, print_progress=True)

    return index, df

def read_evaluatio_dataset():
    """
    Read the evaluation dataset.
    """
    df = pd.read_csv("data/evaluation/eval_users.csv")
    df["business_with_5"] = df["business_with_5"].apply(eval)#.apply(lambda x: x.split(","))
    df["reclist"] = df["reclist"].apply(eval)#$.apply(lambda x: x.split(","))
    return df


if __name__ == '__main__':
    """Convert a yelp dataset file from json to csv."""

    parser = argparse.ArgumentParser(
            description='Avaliação de Embeddings',
            )

    parser.add_argument('embeddings_path', type=str, help='Arquivo de embeddings')
    parser.add_argument('metadados_path', type=str, help='Arquivo de metadados')

    args = parser.parse_args()

    # Create a vector space from the embeddings
    index, df = create_vector_space(args.embeddings_path, args.metadados_path)
    print(df.head())

    # Create a Map Index
    item_code_id = {i: code for i, code in enumerate(df['business_id'])}
    item_id_code = {code: i for i, code in enumerate(df['business_id'])}

    # Load Evaluation Dataset
    eval_users = read_evaluatio_dataset()

    for i, row in tqdm(eval_users.iterrows()):
        print(row)

        # create context array
        code_context  = [item_id_code[str(id)] for id in row.business_with_5] # mapping ids to codes for table lookup
        context_array = []

        for c in code_context:
            context_array.append(np.array(index[c]))
        context_array = np.array(context_array).mean(axis=0)
    
        # rank
        code_reclist  = [item_id_code[str(id)] for id in row.reclist] # mapping ids to codes for table lookup
        rank_bussiness_id = {}
        for bussiness_id in row.reclist:
            cos_sim = cosine_similarity(np.array(index[item_id_code[str(bussiness_id)]]), context_array)
            bussiness_id[bussiness_id] = cos_sim
        
        # order list from dict
        rank_bussiness_id = {k: v for k, v in sorted(rank_bussiness_id.items(), key=lambda item: item[1], reverse=True)}