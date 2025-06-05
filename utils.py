import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from data_utils import *


def bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users, regular = True, lambda_reg = 0.01):
    
    pos_scores = (user_embs[edge_index_t_batch[0, :]] * 
                  item_embs[edge_index_t_batch[1, :] - num_users]).sum(dim=1)
    
    neg_scores = (user_embs[edge_index_t_batch[0, :]] * 
                  item_embs[edge_index_t_batch[2, :] - num_users]).sum(dim=1)
    
    if regular == True:
        reg = (user_embs.norm(2).pow(2) + item_embs.norm(2).pow(2)) / user_embs.size(0)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores)) + lambda_reg * reg
    else:
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    

def multi_negative_bpr_loss(user_embs, item_embs, edge_index_batch, num_users, 
                            regular = True, temperature = 0.2, lambda_reg = 0.01):
    """
    edge_index_batch: (2 + K, B)
        [user, pos_item, neg_item1, neg_item2, ..., neg_itemK]
    """
    user_idx = edge_index_batch[0]                   # (B,)
    pos_item_idx = edge_index_batch[1] - num_users   # (B,)
    neg_item_idx = edge_index_batch[2:] - num_users  # (K, B)

    u_emb = user_embs[user_idx]                      # (B, D)
    pos_emb = item_embs[pos_item_idx]                # (B, D)
    neg_emb = item_embs[neg_item_idx]                # (K, B, D)

    # Positive score: (B,)
    pos_score = torch.sum(u_emb * pos_emb, dim=1)

    # Negative scores: (K, B)
    neg_score = torch.sum(neg_emb * u_emb.unsqueeze(0), dim=2)

    all_scores = torch.cat([pos_score.unsqueeze(0), neg_score], dim=0)  
    log_prob = F.log_softmax(all_scores / temperature, dim=0) 

    if regular: 
        reg = (user_embs.norm(2).pow(2) + item_embs.norm(2).pow(2)) / user_embs.size(0)
        loss = -log_prob[0].mean() + lambda_reg * reg
    else:
        loss = -log_prob[0].mean()

    return loss


def sample_mini_batch(edge_index, batch_size, num_users, num_items, K):

    n_edges = edge_index.shape[1]
    device = edge_index.device
    replace_flag = batch_size > n_edges

    if not replace_flag:
        batch_index = torch.randperm(n_edges, device=device)[:batch_size]
    else:
        batch_index = torch.randint(0, n_edges, (batch_size,), device=device)
    
    pos_edge_index = edge_index[:, batch_index]

    if K == 1:
        edge_index_pn = negative_sampling(pos_edge_index, num_users, num_items)
        return edge_index_pn
    
    elif K > 1:
        edge_index_mpn = multi_negative_sampling(pos_edge_index, num_users, num_items, K)
        return edge_index_mpn


def negative_sampling(edge_index, num_users, num_items):
    user_indices = edge_index[0]
    item_indices = edge_index[1]
    L = edge_index.size(1)

    negative_samples = torch.randint(num_users, num_users + num_items, (L,), device=edge_index.device)

    mask = (negative_samples == item_indices)
    while mask.any():
        negative_samples[mask] = torch.randint(num_users, num_users + num_items, (mask.sum(),), device=edge_index.device)
        mask = (negative_samples == item_indices)

    edge_index_with_neg = torch.cat([edge_index, negative_samples.unsqueeze(0)], dim=0)
          
    return edge_index_with_neg


def multi_negative_sampling(edge_index, num_users, num_items, K):
    user_indices = edge_index[0] 
    pos_items = edge_index[1]
    B = edge_index.shape[1]

    neg_items = torch.randint(
        num_users, num_users + num_items, (K, B), device=edge_index.device
    )

    for k in range(K):
        mask = (neg_items[k] == pos_items)
        while mask.any():
            neg_items[k, mask] = torch.randint(
                num_users, num_users + num_items, (mask.sum(),), device=edge_index.device
            )
            mask = (neg_items[k] == pos_items)

    edge_index_with_neg = torch.cat(
        [user_indices.unsqueeze(0), pos_items.unsqueeze(0), neg_items], dim=0
    )

    return edge_index_with_neg


def contrastive_loss(model, edge_index_series, device, tau=0.5):

    edge_index_aug1, edge_index_aug2 = generate_sub_edge_list(edge_index_series, device)

    user_embs1, item_embs1 = model(edge_index_aug1)
    user_embs2, item_embs2 = model(edge_index_aug2)

    loss_users = info_nce_loss(user_embs1, user_embs2, tau)
    loss_items = info_nce_loss(item_embs1, item_embs2, tau)

    return loss_users + loss_items


def mini_batch_contrastive_loss_num(model, edge_index_series, device, 
                                 tau=0.5, num_sampled_users=512, num_sampled_items=512):

    edge_index_aug1, edge_index_aug2 = generate_sub_edge_list(edge_index_series, device)

    user_embs1, item_embs1 = model(edge_index_aug1)
    user_embs2, item_embs2 = model(edge_index_aug2)

    sampled_user_idx = torch.randperm(user_embs1.shape[0])[:num_sampled_users].to(device)
    sampled_item_idx = torch.randperm(item_embs1.shape[0])[:num_sampled_items].to(device)
                                     
    z1_users = user_embs1[sampled_user_idx]
    z2_users = user_embs2[sampled_user_idx]

    z1_items = item_embs1[sampled_item_idx]
    z2_items = item_embs2[sampled_item_idx]

    loss_users = info_nce_loss(z1_users, z2_users, tau)
    loss_items = info_nce_loss(z1_items, z2_items, tau)

    return loss_users + loss_items


def mini_batch_contrastive_loss_ratio(model, edge_index_series, device, 
                                 tau=0.5, user_sample_ratio=0.7, item_sample_ratio=0.7):

    edge_index_aug1, edge_index_aug2 = generate_sub_edge_list(edge_index_series, device)

    user_embs1, item_embs1 = model(edge_index_aug1)
    user_embs2, item_embs2 = model(edge_index_aug2)

    num_users = user_embs1.shape[0]
    num_items = item_embs1.shape[0]

    num_sampled_users = max(1, int(user_sample_ratio * num_users))
    num_sampled_items = max(1, int(item_sample_ratio * num_items))

    sampled_user_idx = torch.randperm(num_users)[:num_sampled_users].to(device)
    sampled_item_idx = torch.randperm(num_items)[:num_sampled_items].to(device)

    z1_users = user_embs1[sampled_user_idx]
    z2_users = user_embs2[sampled_user_idx]
    z1_items = item_embs1[sampled_item_idx]
    z2_items = item_embs2[sampled_item_idx]

    loss_users = info_nce_loss(z1_users, z2_users, tau)
    loss_items = info_nce_loss(z1_items, z2_items, tau)

    return loss_users + loss_items



def info_nce_loss(z1, z2, tau):
    
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    sim_matrix = torch.mm(z1, z2.T) / tau
    exp_sim = torch.exp(sim_matrix)

    pos_sim = torch.diag(exp_sim)
    contrastive_loss = -torch.mean(torch.log(pos_sim / exp_sim.sum(dim=1)))

    return contrastive_loss


def save_model(epoch, model, optimizer, loss, save_path):
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    
    
def recall_at_k(edge_list, user_embeddings, item_embeddings, K, device):
    
    edge_list = edge_list.to(device)
    user_embeddings = user_embeddings.to(device)
    item_embeddings = item_embeddings.to(device)
    
    users = edge_list[0]
    items = edge_list[1]
    
    num_users = user_embeddings.shape[0]    
    scores = torch.matmul(user_embeddings, item_embeddings.T)     
    top_k_items = torch.topk(scores, K, dim=1)[1] + num_users  
    
    total_recall = torch.tensor(0.0, device=device)
    
    for user_idx in range(num_users):
        
        relevant_items = items[users == user_idx].unique()  
        if len(relevant_items) == 0:
            continue 
        
        predicted_items = top_k_items[user_idx]        
        correct_predictions = torch.sum(torch.isin(predicted_items, relevant_items)).float()        
        recall = correct_predictions / min(len(relevant_items), K)
        total_recall += recall
    
    avg_recall = total_recall / num_users
    
    return avg_recall.item()


def ndcg_at_k(edge_list, user_embeddings, item_embeddings, K, device):
    
    edge_list = edge_list.to(device)
    user_embeddings = user_embeddings.to(device)
    item_embeddings = item_embeddings.to(device)    
    
    users = edge_list[0]
    items = edge_list[1]
    
    num_users = user_embeddings.shape[0]
    
    scores = torch.matmul(user_embeddings, item_embeddings.T) 
    top_k_items = torch.topk(scores, K, dim=1)[1] + num_users 
    total_ndcg = torch.tensor(0.0, device=device)
    
    for user_idx in range(num_users):
        
        relevant_items = items[users == user_idx].unique() 
        
        if len(relevant_items) == 0:
            continue  
        
        predicted_items = top_k_items[user_idx]        
        dcg = torch.tensor(0.0, device=device)
        for rank, item_idx in enumerate(predicted_items):
            relevance = torch.tensor(1.0, device=device) if item_idx in relevant_items else torch.tensor(0.0, device=device)
            dcg += relevance / torch.log2(torch.tensor(rank + 2.0, device=device))  # 랭크는 0부터 시작하므로 log2(rank+2)
        
        idcg = torch.tensor(0.0, device=device)
        ideal_relevant_count = min(len(relevant_items), K)
        for rank in range(ideal_relevant_count):
            idcg += 1.0 / torch.log2(torch.tensor(rank + 2.0, device=device)) 
        
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = torch.tensor(0.0, device=device)
        
        total_ndcg += ndcg    
    avg_ndcg = total_ndcg / num_users
    
    return avg_ndcg


def tensor_to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def tensor_to_float(x, digits=4):
    return round(x.item(), digits) if torch.is_tensor(x) else round(x, digits)


