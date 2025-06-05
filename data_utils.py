import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas.api.types import is_datetime64_any_dtype

import os
import urllib.request
import zipfile
import shutil
import gzip
import json

from tqdm import tqdm


def download_data(dataset_name=None):
    os.makedirs('data', exist_ok=True)

    if dataset_name == 'movielens':
        target_csv_path = os.path.join('data', 'ml-25m.csv')
        if os.path.exists(target_csv_path):
            print(f"{dataset_name} data already exists.")
            return

        print("Downloading MovieLens 25M dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        zip_path = "ml-25m.zip"
        extract_dir = "ml-25m"

        urllib.request.urlretrieve(url, zip_path)
        print("Download completed.")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Unzip completed.")

        src_ratings = os.path.join(extract_dir, "ml-25m", "ratings.csv")
        shutil.copy(src_ratings, target_csv_path)
        print(f"ratings.csv copied to {target_csv_path}")

        os.remove(zip_path)
        shutil.rmtree(extract_dir)

    elif dataset_name == 'netflix':

        output_csv_path = os.path.join('data', 'netflix.csv')
        if os.path.exists(output_csv_path):
            print(f"{dataset_name} data already exists.")
            return output_csv_path
            
        zip_files = [
            "combined_data_1.txt",
            "combined_data_2.txt",
            "combined_data_3.txt",
            "combined_data_4.txt"
        ]
        extracted_files = []

        for zip_path in zip_files:
            if not os.path.exists(zip_path):
                print(f"Missing zip file: {zip_path}")
                return None
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                name = zip_ref.namelist()[0]
                zip_ref.extractall("data/")
                extracted_files.append(os.path.join("data", name))        

        print("Processing Netflix dataset...")
        rows = []
        current_movie_id = None

        for file in extracted_files:
            print(f"Reading {file}...")
            with open(file, 'r', encoding='latin1') as f:
                for line in tqdm(f, desc=os.path.basename(file)):
                    line = line.strip()
                    if line.endswith(':'):
                        current_movie_id = int(line[:-1])
                    else:
                        user_id, rating, date = line.split(',')
                        rows.append([int(user_id), current_movie_id, int(rating), date])

        df = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
        df.to_csv(output_csv_path, index=False)
        print(f"Saved processed file to {output_csv_path}")
        
        for file in extracted_files:
            os.remove(file)
        print("Temporary extracted .txt files have been deleted.")
    

def convert_jsonl_gz_to_csv(input_path, output_csv_path):
    data = []

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Converted and saved to {output_csv_path}")
    
    
def preprocess_data(args, generate_edge_list):

    if args.data_name == 'movielens':
        data_path = os.path.join('data', 'ml-25m.csv')
    elif args.data_name == 'netflix':
        data_path = os.path.join('data', 'netflix.csv')
    else:
        raise ValueError(f"Unknown dataset name: {args.data}")
    
    data = pd.read_csv(data_path)
    data = data[data['rating'] >= args.rate]

    filtered_data = data.copy()
    while True:
        user_interaction = filtered_data['userId'].value_counts()
        filtered_users_idx = user_interaction[user_interaction >= args.interaction_threshold].index
        filtered_data = filtered_data[filtered_data['userId'].isin(filtered_users_idx)]

        item_interaction = filtered_data['movieId'].value_counts()
        filtered_items_idx = item_interaction[item_interaction >= args.interaction_threshold].index
        filtered_data = filtered_data[filtered_data['movieId'].isin(filtered_items_idx)]

        if len(filtered_users_idx) == len(user_interaction) and len(filtered_items_idx) == len(item_interaction):
            break

    user_id_map = {id_: i for i, id_ in enumerate(filtered_data['userId'].unique())}
    item_id_map = {id_: i for i, id_ in enumerate(filtered_data['movieId'].unique())}

    filtered_data['userId'] = filtered_data['userId'].map(user_id_map)
    filtered_data['movieId'] = filtered_data['movieId'].map(item_id_map)

    num_users = len(user_id_map)
    num_items = len(item_id_map)
    
    unique_interactions = filtered_data[['userId', 'movieId']].drop_duplicates().shape[0]
    density = np.round((unique_interactions / (num_users * num_items)) * 100, 4)

    print(f"INTERACTION: {filtered_data.shape[0]} || USERS: {num_users} | ITEMS: {num_items} || DENSITY: {density} %")
    
    edge_index_series, edge_index_series_2 = generate_edge_list(
        filtered_data,
        coluser='userId',
        colitem='movieId',
        coltime='timestamp',
        timepoints=args.all_time_steps,
        num_users=num_users
    )

    print(f'0th time point interaction: {edge_index_series[0].shape[1]}')

    return filtered_data, edge_index_series, edge_index_series_2, num_users, num_items


def generate_edge_list(data, coluser, colitem, coltime, timepoints, num_users):
    
    if not is_datetime64_any_dtype(data[coltime]):
        data[coltime] = pd.to_datetime(data[coltime])
    
    data['time_point'] = pd.qcut(data[coltime], timepoints, labels = False)
    
    edge_index_series = []
    for t in range(timepoints):
        temp_data = data[data['time_point'] == t]
        edge_index = torch.tensor(np.array([temp_data[coluser].values, 
                                            temp_data[colitem].values + num_users]), 
                                  dtype = torch.long)
        edge_index_series.append(edge_index)
        
    edge_index_series_2 = []
    for edge_list in edge_index_series:
        list1 = torch.cat([edge_list[0], edge_list[1]])
        list2 = torch.cat([edge_list[1], edge_list[0]])
        new_edge_list = torch.stack([list1, list2])
        
        edge_index_series_2.append(new_edge_list)
        
    return edge_index_series, edge_index_series_2


def generate_sub_edge_list(edge_index_series, device, dropout_rate=0.4, num_subgraph=2):
    
    sub_edge_index_series_1 = []
    sub_edge_index_series_2 = []
    for edge_index in edge_index_series:
        edge_index = edge_index.to(device)
        
        sub_edge_indices = []
        for _ in range(num_subgraph):
            # edge dropout
            num_edges = edge_index.shape[1]
            keep_edges = int(num_edges * (1 - dropout_rate))
            selected_indices = np.random.choice(num_edges, keep_edges, replace=False)
            edge_index = edge_index[:, selected_indices]
            sub_edge_indices.append(edge_index)
        
        sub_edge_index_per_time = []    
        for sub_edge_index in sub_edge_indices:
            list1 = torch.cat([sub_edge_index[0], sub_edge_index[1]])
            list2 = torch.cat([sub_edge_index[1], sub_edge_index[0]])
            new_edge_list = torch.stack([list1, list2])
            
            sub_edge_index_per_time.append(new_edge_list)
            
        sub_edge_index_series_1.append(sub_edge_index_per_time[0])
        sub_edge_index_series_2.append(sub_edge_index_per_time[1])
        
    return sub_edge_index_series_1, sub_edge_index_series_2
