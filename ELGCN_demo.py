import torch
import argparse
from model import *
from utils import *
from data_utils import *

import warnings
warnings.filterwarnings('ignore')

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
print("Current working directory:", os.getcwd())

device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
print(f"TORCH: {torch.cuda.is_available()} | GPU: {device}")

parser = argparse.ArgumentParser(description="ELGCN Config")

# DATA PARAMETERS
parser.add_argument('--data', type=str, default='movielens', choices=['movielens', 'netflix'])
parser.add_argument('--rate', type=int, default=4)
parser.add_argument('--interaction_threshold', type=int, default=500)
parser.add_argument('--all_time_steps', type=int, default=59)

# MODEL PARAMETERS
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--agg_mode', type=str, choices=['mean', 'attention'], default='mean')
parser.add_argument('--first', type=str, choices=['GRU', 'GCN'], default='GCN')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--EMA', type=bool, choices=[True, False], default=True)
parser.add_argument('--ema_alpha', type=float, default=0.7)

# TIME SETTINGS
parser.add_argument('--time_window', type=int, default=5)
parser.add_argument('--tr_first_time', type=int, default=5)
parser.add_argument('--tr_last_time', type=int, default=40)
parser.add_argument('--val_first_time', type=int, default=40)
parser.add_argument('--val_last_time', type=int, default=45)
parser.add_argument('--ts_first_time', type=int, default=45)
parser.add_argument('--ts_last_time', type=int, default=55)

# TRAINING SETTINGS
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--improvement_threshold', type=float, default=0.03)
parser.add_argument('--multi_sampling', type=bool, choices=[True, False], default=True)
parser.add_argument('--one_batch_sample', type=bool,choices=[True, False], default=False)
parser.add_argument('--recent_loss', type=int, default=3)
parser.add_argument('--sgl', type=bool, choices=['True', 'False'], default=False)
parser.add_argument('--cont_loss', type=str, choices=['vanilla', 'num_batch', 'ratio_batch'], default='ratio_batch')
parser.add_argument('--ssl_lambda', type=float, default=0.01)

# SAVE/MODEL VERSION
parser.add_argument('--model_ver', type=str, default='demo')
parser.add_argument('--model_name', type=str, default='elgcn')

args = parser.parse_args()

# DATA DOWNLOAD
download_data(args.data)

# DATA GENERATE
filtered_data, edge_index_series, edge_index_series_2,\
    num_users, num_items = preprocess_data(args=args,
                                                     generate_edge_list=generate_edge_list)

# DEFINE MODEL
model = ELGCN(
    num_users=num_users,
    num_items=num_items,
    emb_dim=args.embedding_dim,
    num_layers=args.num_layers,
    agg_mode=args.agg_mode,
    first=args.first,
    device=device,
    EMA=args.EMA,
    ema_alpha=args.ema_alpha
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate
)

# TRAIN MODEL
best_model_state, model_path = model.train_model(
    edge_index_series=edge_index_series,
    edge_index_series_2=edge_index_series_2,
    optimizer=optimizer,
    sample_mini_batch=sample_mini_batch,
    bpr_loss=bpr_loss,
    multi_negative_bpr_loss=multi_negative_bpr_loss,
    one_batch_sample = args.one_batch_sample,
    num_users=num_users,
    num_items=num_items,
    batch_size=args.batch_size,
    time_steps=args.time_window,
    tr_first_time=args.tr_first_time,
    tr_last_time=args.tr_last_time,
    val_first_time=args.val_first_time,
    val_last_time=args.val_last_time,
    epochs=args.epochs,
    patience=args.patience,
    improvement_threshold=args.improvement_threshold,
    recent_loss=args.recent_loss,
    model_ver=args.model_ver,
    multi_sampling=args.multi_sampling,
    sgl=args.sgl,
    cont_loss=args.cont_loss,
    ssl_lambda=args.ssl_lambda,
    data_name=args.data_name,
    model_name=args.model_name
)

# TEST MODEL
results_df = model.test_model(
    model_path=model_path,
    edge_index_series_2=edge_index_series_2,
    edge_index_series=edge_index_series,
    recall_at_k=recall_at_k,
    ndcg_at_k=ndcg_at_k,
    time_steps=args.time_window,
    ts_first_time=args.ts_first_time,
    ts_last_time=args.ts_last_time,
    device=device,
    model_ver=args.model_ver,
    model_name=args.model_name,
    data_name=args.data
)
